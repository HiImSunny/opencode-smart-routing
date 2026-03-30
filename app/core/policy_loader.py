import yaml
import os
from datetime import datetime, timezone
from typing import Dict
from app.schemas.router_policy import PolicyConfig, ProvidersConfig, FallbackEntry

# Holds the loaded config objects, updated on reload
_policy: PolicyConfig | None = None
_providers: ProvidersConfig | None = None
_policy_loaded_at: str = ""


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_policy(policy_path: str, providers_path: str) -> None:
    global _policy, _providers, _policy_loaded_at

    raw_policy = _load_yaml(policy_path)
    raw_providers = _load_yaml(providers_path)

    _policy = PolicyConfig(**raw_policy)
    _providers = ProvidersConfig(**raw_providers)
    _policy_loaded_at = datetime.now(timezone.utc).isoformat()
    
    # Override ollama models from .env at startup
    from app.core.settings import OLLAMA_MODEL
    set_ollama_model(OLLAMA_MODEL)


def get_policy() -> PolicyConfig:
    if _policy is None:
        raise RuntimeError("Policy not loaded. Call load_policy() first.")
    return _policy


def get_providers() -> ProvidersConfig:
    if _providers is None:
        raise RuntimeError("Providers not loaded. Call load_policy() first.")
    return _providers


def get_policy_loaded_at() -> str:
    return _policy_loaded_at


def get_ollama_model() -> str:
    """Return the current Ollama model used in fallback chains."""
    if _policy is None:
        from app.core.settings import OLLAMA_MODEL
        return OLLAMA_MODEL
    for chain in _policy.fallback_chains.values():
        for entry in chain:
            if entry.provider == "ollama":
                return entry.model
    from app.core.settings import OLLAMA_MODEL
    return OLLAMA_MODEL


def set_ollama_model(model: str) -> int:
    """
    Replace the Ollama model in ALL fallback chain entries at runtime.
    Returns the number of chain entries updated.
    """
    if _policy is None:
        raise RuntimeError("Policy not loaded.")
    count = 0
    for chain in _policy.fallback_chains.values():
        for entry in chain:
            if entry.provider == "ollama":
                entry.model = model
                count += 1
    return count
