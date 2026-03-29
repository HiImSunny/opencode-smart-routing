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
