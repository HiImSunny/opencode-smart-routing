from dataclasses import dataclass, field
from typing import Optional, List
import time
import uuid


@dataclass
class TelemetryRecord:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    route_class: str = ""
    chosen_provider: str = ""
    chosen_model: str = ""
    fallback_index: int = 0        # 0 = primary succeeded
    input_chars: int = 0
    has_image: bool = False
    message_count: int = 0
    latency_ms: float = 0.0
    status: str = ""               # success | all_failed
    errors: List[dict] = field(default_factory=list)
    stream: bool = False
