import httpx
import time
from typing import Tuple, Dict
from app.schemas.router_policy import PolicyConfig
from app.adapters.base import BaseAdapter
from app.adapters.openai_compatible_adapter import OpenAICompatibleAdapter
from app.telemetry.models import TelemetryRecord
from app.schemas.openai_chat import ChatCompletionRequest, ChatCompletionResponse


async def execute_with_fallback(
    request: ChatCompletionRequest,
    route_class: str,
    policy: PolicyConfig,
    adapters: Dict[str, BaseAdapter],
    record: TelemetryRecord,
) -> Tuple[ChatCompletionResponse, TelemetryRecord]:
    """
    Try each provider in the fallback chain in order.
    Records errors along the way and raises RuntimeError if all fail.
    """
    chain = policy.fallback_chains.get(route_class, [])
    if not chain:
        raise RuntimeError(f"No fallback chain defined for route class: {route_class}")

    retry_on_status = set(policy.retry.retry_on_status)

    for idx, entry in enumerate(chain):
        adapter = adapters.get(entry.provider)
        if adapter is None:
            err = f"No adapter registered for provider '{entry.provider}'"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            continue

        # Fast pre-flight: skip providers that are obviously unavailable
        # (e.g. placeholder URL, missing API key). OllamaAdapter checks /api/tags;
        # OpenAICompatibleAdapter checks key+URL in-memory — both are near-instant.
        if not await adapter.is_available():
            err = f"Provider '{entry.provider}' is not available (skipped)"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            continue

        try:
            t0 = time.monotonic()
            response = await adapter.chat(request, entry.model)
            record.latency_ms = (time.monotonic() - t0) * 1000

            # Validate non-empty choices
            if not response.choices:
                raise ValueError("Empty choices in response")

            record.chosen_model = entry.model
            record.chosen_provider = entry.provider
            record.fallback_index = idx
            record.status = "success"
            return response, record

        except httpx.TimeoutException as e:
            record.errors.append({
                "provider": entry.provider,
                "model": entry.model,
                "error": f"Timeout: {e}",
            })
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else 0
            record.errors.append({
                "provider": entry.provider,
                "model": entry.model,
                "error": f"HTTP {status_code}: {e}",
            })
            # Only trigger fallback on configured status codes
            if status_code not in retry_on_status:
                raise
        except ValueError as e:
            record.errors.append({
                "provider": entry.provider,
                "model": entry.model,
                "error": f"Validation: {e}",
            })
        except Exception as e:
            record.errors.append({
                "provider": entry.provider,
                "model": entry.model,
                "error": str(e),
            })

    record.status = "all_failed"
    raise RuntimeError(
        f"All fallbacks exhausted for route '{route_class}'. "
        f"Errors: {record.errors}"
    )


async def stream_with_fallback(
    request: ChatCompletionRequest,
    route_class: str,
    policy: PolicyConfig,
    adapters: Dict[str, BaseAdapter],
    record: TelemetryRecord,
):
    """
    Streaming version: find the first working provider then yield SSE chunks.
    Falls back to the next provider if the first is unavailable.
    """
    import sys
    chain = policy.fallback_chains.get(route_class, [])
    if not chain:
        raise RuntimeError(f"No fallback chain defined for route class: {route_class}")

    for idx, entry in enumerate(chain):
        adapter = adapters.get(entry.provider)
        if adapter is None:
            err = f"No adapter registered for provider '{entry.provider}'"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Skip {entry.provider}: {err}", file=sys.stderr)
            continue

        # Fast pre-flight: skip providers that are obviously unavailable
        if not await adapter.is_available():
            err = f"Provider '{entry.provider}' is not available (skipped)"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Skip {entry.provider}: not available", file=sys.stderr)
            continue

        try:
            record.chosen_model = entry.model
            record.chosen_provider = entry.provider
            record.fallback_index = idx
            record.status = "success"
            async for chunk in adapter.stream_chat(request, entry.model):
                yield chunk
            return  # success — stop iterating the chain
        except httpx.TimeoutException as e:
            err = f"Timeout after {entry.provider}: {e}"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Fallback from {entry.provider} ({entry.model}): {err}", file=sys.stderr)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else 0
            err = f"HTTP {status_code}: {e}"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Fallback from {entry.provider} ({entry.model}): {err}", file=sys.stderr)
            if status_code not in set(policy.retry.retry_on_status):
                raise  # non-retryable — bubble up immediately
        except ValueError as e:
            err = f"Config error for {entry.provider}: {e}"
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Fallback from {entry.provider} ({entry.model}): {err}", file=sys.stderr)
        except Exception as e:
            err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            record.errors.append({"provider": entry.provider, "model": entry.model, "error": err})
            print(f"[ROUTER] Fallback from {entry.provider} ({entry.model}): {err}", file=sys.stderr)

    record.status = "all_failed"
    raise RuntimeError(
        f"All streaming fallbacks exhausted for route '{route_class}'. "
        f"Errors: {record.errors}"
    )
