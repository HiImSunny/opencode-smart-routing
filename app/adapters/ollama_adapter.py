import httpx
import time
import uuid
import sys
from typing import AsyncIterator
from app.adapters.base import BaseAdapter
from app.schemas.openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

# Ollama local models have a limited context window (~8k–32k tokens).
# 1 token ≈ 4 chars; 32k tokens ≈ 128k chars. We cap at 120k to be safe.
MAX_CONTEXT_CHARS = 120_000


def _to_ollama_messages(request: ChatCompletionRequest) -> list[dict]:
    """Convert OpenAI messages to Ollama format, truncating if context is too large."""
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            # Build content string from parts (Ollama doesn't support multipart)
            text_parts = [p.text for p in msg.content if p.type == "text" and p.text]
            messages.append({"role": msg.role, "content": " ".join(text_parts)})

    # Truncate to fit Ollama's context window:
    # Keep the system message (if first) + most recent messages.
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > MAX_CONTEXT_CHARS:
        print(
            f"[OLLAMA] Context too large ({total_chars:,} chars > {MAX_CONTEXT_CHARS:,} limit). "
            f"Truncating from {len(messages)} to most recent messages.",
            file=sys.stderr,
        )
        # Always keep system prompt if present
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # Drop oldest non-system messages until within budget
        budget = MAX_CONTEXT_CHARS - sum(len(m["content"]) for m in system_msgs)
        kept = []
        for msg in reversed(non_system):
            if budget <= 0:
                break
            kept.insert(0, msg)
            budget -= len(msg["content"])

        messages = system_msgs + kept
        new_total = sum(len(m["content"]) for m in messages)
        print(
            f"[OLLAMA] After truncation: {len(messages)} messages, {new_total:,} chars.",
            file=sys.stderr,
        )

    return messages


class OllamaAdapter(BaseAdapter):
    _AVAILABILITY_TTL = 10.0  # seconds between health checks

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available_cache: bool | None = None
        self._available_checked_at: float = 0.0

    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = {
            "model": model,
            "messages": _to_ollama_messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        if request.top_p is not None:
            payload.setdefault("options", {})["top_p"] = request.top_p

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp = await resp
            if resp.status_code == 404:
                print(f"\n⚠️  [OLLAMA WARNING] Mô hình '{model}' CHƯA ĐƯỢC TẢI VỀ! Vui lòng mở Terminal và chạy lệnh: ollama pull {model}\n")
            resp.raise_for_status()

        data = resp.json()

        # Ollama response schema: {"message": {"role": "...", "content": "..."}, ...}
        msg_data = data.get("message", {})
        role = msg_data.get("role", "assistant")
        content = msg_data.get("content", "")

        # Token usage (Ollama provides eval_count / prompt_eval_count)
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

        return ChatCompletionResponse(
            id=f"ollama-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=role, content=content),
                    finish_reason=data.get("done_reason", "stop"),
                )
            ],
            usage=usage,
        )

    async def stream_chat(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[bytes]:
        """Yield SSE chunks by mapping Ollama's native stream to OpenAI format."""
        import json
        payload = {
            "model": model,
            "messages": _to_ollama_messages(request),
            "stream": True,
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        if request.top_p is not None:
            payload.setdefault("options", {})["top_p"] = request.top_p

        # Probe the connection first (before entering the async generator)
        # so exceptions propagate normally to the fallback manager.
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as resp:
                    if resp.status_code == 404:
                        print(
                            f"\n⚠️  [OLLAMA WARNING] Mô hình '{model}' CHƯA ĐƯỢC TẢI VỀ! "
                            f"Vui lòng mở Terminal và chạy lệnh: ollama pull {model}\n"
                        )
                    if resp.status_code != 200:
                        # Read body to get the error message before raising
                        body = await resp.aread()
                        raise httpx.HTTPStatusError(
                            f"Ollama returned HTTP {resp.status_code}: {body.decode(errors='replace')}",
                            request=resp.request,
                            response=resp,
                        )

                    chat_id = f"chatcmpl-{uuid.uuid4()}"
                    created_time = int(time.time())

                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)

                            # Check if Ollama signalled an error mid-stream
                            if "error" in chunk_data:
                                raise RuntimeError(
                                    f"Ollama stream error for model '{model}': {chunk_data['error']}"
                                )

                            content = chunk_data.get("message", {}).get("content", "")
                            is_done = chunk_data.get("done", False)

                            openai_chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": "stop" if is_done else None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode("utf-8")

                            if is_done:
                                yield b"data: [DONE]\n\n"
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama timeout for model '{model}' at {self.base_url} "
                f"(likely context too large or model overloaded): {type(e).__name__}"
            ) from e

    async def is_available(self) -> bool:
        now = time.monotonic()
        if self._available_cache is not None and (now - self._available_checked_at) < self._AVAILABILITY_TTL:
            return self._available_cache
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._available_cache = resp.status_code == 200
        except Exception:
            self._available_cache = False
        self._available_checked_at = time.monotonic()
        return self._available_cache

    async def get_installed_models(self) -> list[str]:
        """Fetch list of installed models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name") for m in data.get("models", [])]
                return []
        except Exception:
            return []
