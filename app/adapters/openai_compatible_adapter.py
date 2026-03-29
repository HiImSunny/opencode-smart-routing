import httpx
import time
import os
from typing import AsyncIterator
from app.adapters.base import BaseAdapter
from app.schemas.openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)


class OpenAICompatibleAdapter(BaseAdapter):
    def __init__(self, base_url: str, api_key_env: str | None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ChatCompletionRequest, model: str) -> dict:
        payload: dict = {
            "model": model,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str)
                    else [p.model_dump(exclude_none=True) for p in m.content],
                }
                for m in request.messages
            ],
            "stream": request.stream or False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop"] = request.stop
        return payload

    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        if not self.api_key:
            raise ValueError(f"API key not configured for provider {self.base_url}")
        
        payload = self._build_payload(request, model)
        payload["stream"] = False  # non-streaming path

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()

        data = resp.json()

        # Validate that choices are present
        choices_data = data.get("choices", [])
        if not choices_data:
            raise ValueError(f"Empty choices in response from {self.base_url}")

        choices = []
        for c in choices_data:
            msg_data = c.get("message", {})
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=Message(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{int(time.time())}"),
            object=data.get("object", "chat.completion"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
        )

    async def stream_chat(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[bytes]:
        """Yield SSE chunks as raw bytes for streaming pass-through."""
        if not self.api_key:
            raise ValueError(f"API key not configured for provider {self.base_url}")
            
        payload = self._build_payload(request, model)
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def is_available(self) -> bool:
        """Cloud adapters are available only if API key is configured and URL is valid."""
        is_dummy_url = "your-opencode-go-endpoint" in self.base_url
        return bool(self.api_key) and not is_dummy_url
