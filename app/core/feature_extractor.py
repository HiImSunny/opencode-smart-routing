from dataclasses import dataclass
from typing import List
from app.schemas.openai_chat import ChatCompletionRequest, ContentPart


@dataclass
class RequestFeatures:
    total_chars: int
    message_count: int
    has_image: bool
    last_user_text: str
    all_text: str


def extract_features(request: ChatCompletionRequest) -> RequestFeatures:
    total_chars = 0
    has_image = False
    texts: List[str] = []

    for msg in request.messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
            texts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text" and part.text:
                    total_chars += len(part.text)
                    texts.append(part.text)
                elif part.type == "image_url":
                    has_image = True

    last_user_text = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                last_user_text = msg.content
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == "text":
                        last_user_text = part.text or ""
            break

    return RequestFeatures(
        total_chars=total_chars,
        message_count=len(request.messages),
        has_image=has_image,
        last_user_text=last_user_text.lower(),
        all_text=" ".join(texts).lower(),
    )
