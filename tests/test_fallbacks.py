"""
Tests for the fallback manager logic.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from app.core.fallback_manager import execute_with_fallback
from app.schemas.router_policy import PolicyConfig, RoutingConfig, Thresholds, FallbackEntry
from app.schemas.openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)
from app.telemetry.models import TelemetryRecord
import time


# --- Helpers ---

def make_policy(*providers_and_models):
    """Create a policy with a test fallback chain."""
    chain = [FallbackEntry(provider=p, model=m) for p, m in providers_and_models]
    return PolicyConfig(
        routing=RoutingConfig(
            default_route="cloud_fast",
            thresholds=Thresholds(),
            keyword_rules={},
        ),
        fallback_chains={"test_route": chain},
    )


def make_response(model="gpt-test") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="test-id",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello"),
                finish_reason="stop",
            )
        ],
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Hi")],
    )


# --- Tests ---

@pytest.mark.asyncio
async def test_primary_success_fallback_index_zero():
    """When primary succeeds, fallback_index should be 0."""
    mock_adapter = AsyncMock()
    mock_adapter.chat = AsyncMock(return_value=make_response("model-a"))

    policy = make_policy(("provider_a", "model-a"), ("provider_b", "model-b"))
    adapters = {"provider_a": mock_adapter, "provider_b": AsyncMock()}
    record = TelemetryRecord()

    resp, record = await execute_with_fallback(make_request(), "test_route", policy, adapters, record)

    assert record.fallback_index == 0
    assert record.chosen_provider == "provider_a"
    assert record.chosen_model == "model-a"
    assert record.status == "success"
    assert len(record.errors) == 0


@pytest.mark.asyncio
async def test_primary_timeout_triggers_fallback():
    """When primary times out, should try fallback and succeed."""
    primary = AsyncMock()
    primary.chat = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    fallback = AsyncMock()
    fallback.chat = AsyncMock(return_value=make_response("model-b"))

    policy = make_policy(("primary", "model-a"), ("fallback_p", "model-b"))
    adapters = {"primary": primary, "fallback_p": fallback}
    record = TelemetryRecord()

    resp, record = await execute_with_fallback(make_request(), "test_route", policy, adapters, record)

    assert record.fallback_index == 1
    assert record.chosen_provider == "fallback_p"
    assert record.status == "success"
    assert len(record.errors) == 1
    assert "Timeout" in record.errors[0]["error"]


@pytest.mark.asyncio
async def test_all_fail_raises_runtime_error():
    """When all providers fail, RuntimeError should be raised and status set to all_failed."""
    adapter_a = AsyncMock()
    adapter_a.chat = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    adapter_b = AsyncMock()
    adapter_b.chat = AsyncMock(side_effect=ValueError("empty response"))

    policy = make_policy(("a", "model-a"), ("b", "model-b"))
    adapters = {"a": adapter_a, "b": adapter_b}
    record = TelemetryRecord()

    with pytest.raises(RuntimeError, match="All fallbacks exhausted"):
        await execute_with_fallback(make_request(), "test_route", policy, adapters, record)

    assert record.status == "all_failed"
    assert len(record.errors) == 2


@pytest.mark.asyncio
async def test_missing_provider_skipped():
    """If an adapter is not registered, it should be skipped."""
    fallback = AsyncMock()
    fallback.chat = AsyncMock(return_value=make_response("model-b"))

    policy = make_policy(("missing", "model-a"), ("registered", "model-b"))
    adapters = {"registered": fallback}  # "missing" adapter not in dict
    record = TelemetryRecord()

    resp, record = await execute_with_fallback(make_request(), "test_route", policy, adapters, record)

    assert record.chosen_provider == "registered"
    assert record.status == "success"


@pytest.mark.asyncio
async def test_empty_choices_triggers_fallback():
    """If adapter returns empty choices, should trigger fallback."""
    from app.schemas.openai_chat import ChatCompletionResponse
    import time

    empty_resp = ChatCompletionResponse(
        id="empty", created=int(time.time()), model="x", choices=[]
    )
    primary = AsyncMock()
    primary.chat = AsyncMock(return_value=empty_resp)

    fallback = AsyncMock()
    fallback.chat = AsyncMock(return_value=make_response("model-b"))

    policy = make_policy(("primary", "model-a"), ("fallback_p", "model-b"))
    adapters = {"primary": primary, "fallback_p": fallback}
    record = TelemetryRecord()

    resp, record = await execute_with_fallback(make_request(), "test_route", policy, adapters, record)

    assert record.fallback_index == 1
    assert record.status == "success"
