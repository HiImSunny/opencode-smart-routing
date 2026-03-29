from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class FallbackEntry(BaseModel):
    provider: str
    model: str


class Thresholds(BaseModel):
    local_fast_max_chars: int = 2000
    local_fast_max_messages: int = 6
    cloud_architecture_min_chars: int = 8000


class RoutingConfig(BaseModel):
    default_route: str = "cloud_fast"
    thresholds: Thresholds = Thresholds()
    keyword_rules: Dict[str, List[str]] = {}


class TimeoutsConfig(BaseModel):
    ollama_seconds: int = 60
    cloud_seconds: int = 30
    fallback_extra_seconds: int = 10


class RetryConfig(BaseModel):
    max_retries_per_provider: int = 2
    retry_on_status: List[int] = [429, 500, 502, 503, 504]


class CooldownConfig(BaseModel):
    enabled: bool = True
    window_seconds: int = 60
    failure_threshold: int = 3


class PolicyConfig(BaseModel):
    routing: RoutingConfig
    fallback_chains: Dict[str, List[FallbackEntry]]
    timeouts: TimeoutsConfig = TimeoutsConfig()
    retry: RetryConfig = RetryConfig()
    cooldown: CooldownConfig = CooldownConfig()


class ProviderConfig(BaseModel):
    base_url: str
    type: str  # "ollama" | "openai_compatible"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None


class ProvidersConfig(BaseModel):
    providers: Dict[str, ProviderConfig]
