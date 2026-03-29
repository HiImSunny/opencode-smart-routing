import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.schemas.openai_chat import ChatCompletionRequest
from app.core.feature_extractor import extract_features
from app.core.router_engine import route
from app.core.fallback_manager import execute_with_fallback, stream_with_fallback
from app.core.policy_loader import get_policy
from app.telemetry.models import TelemetryRecord
from app.telemetry.logger import log_record
from app.core.settings import LOG_DIR

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(http_request: Request, body: ChatCompletionRequest):
    policy = get_policy()
    adapters = http_request.app.state.adapters

    # 1. Extract routing features
    features = extract_features(body)

    # 2. Determine route class
    route_class = route(features, policy)

    # 3. Init telemetry record
    record = TelemetryRecord(
        route_class=route_class,
        input_chars=features.total_chars,
        has_image=features.has_image,
        message_count=features.message_count,
        stream=bool(body.stream),
    )

    t_start = time.monotonic()

    # 4. Execute (streaming or regular)
    if body.stream:
        async def event_generator():
            try:
                async for chunk in stream_with_fallback(body, route_class, policy, adapters, record):
                    yield chunk
            finally:
                record.latency_ms = (time.monotonic() - t_start) * 1000
                await log_record(record, LOG_DIR)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response, record = await execute_with_fallback(body, route_class, policy, adapters, record)
        record.latency_ms = (time.monotonic() - t_start) * 1000
    except RuntimeError as e:
        record.latency_ms = (time.monotonic() - t_start) * 1000
        await log_record(record, LOG_DIR)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        # Always log even on errors (record.status == "all_failed" in that case)
        pass

    await log_record(record, LOG_DIR)
    return response
