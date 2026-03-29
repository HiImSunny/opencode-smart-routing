from abc import ABC, abstractmethod
from app.schemas.openai_chat import ChatCompletionRequest, ChatCompletionResponse


class BaseAdapter(ABC):
    @abstractmethod
    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        """Send a chat completion request and return a normalized response."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend is reachable and healthy."""
        ...
