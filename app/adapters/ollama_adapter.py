import httpx
import time
import uuid
from typing import Any
from app.adapters.base import BaseAdapter
from app.schemas.openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)


def _to_ollama_messages(request: ChatCompletionRequest) -> list[dict]:
    """Convert OpenAI messages to Ollama format."""
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            # Build content string from parts (Ollama doesn't support multipart)
            text_parts = [p.text for p in msg.content if p.type == "text" and p.text]
            messages.append({"role": msg.role, "content": " ".join(text_parts)})
    return messages


class OllamaAdapter(BaseAdapter):
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
