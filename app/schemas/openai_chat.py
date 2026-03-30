from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union, Literal, Any, Dict


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[dict] = None


class ToolCall(BaseModel):
    """Represents a tool call made by the assistant."""
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]  # {"name": "...", "arguments": "..."}


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[ContentPart], None] = None
    name: Optional[str] = None
    # For assistant messages that include tool calls
    tool_calls: Optional[List[ToolCall]] = None
    # For tool-role messages (results returned to model)
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    # Tool calling fields (OpenCode uses these heavily)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Pass-through for any other fields OpenCode may send (e.g. reasoning_effort etc.)
    model_config = ConfigDict(extra="allow")


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
