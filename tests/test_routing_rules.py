"""
Tests for the rule-based routing engine.
"""
import pytest
from app.core.feature_extractor import RequestFeatures
from app.core.router_engine import route
from app.schemas.router_policy import (
    PolicyConfig,
    RoutingConfig,
    Thresholds,
    FallbackEntry,
)

# --- Shared policy fixture ---

KEYWORD_RULES = {
    "cloud_vision": ["image", "screenshot", "figma", "ui mockup", "photo"],
    "cloud_debug": ["debug", "stacktrace", "traceback", "logs", "security", "vulnerability", "review"],
    "cloud_architecture": ["architecture", "refactor", "multi-file", "migration", "system design", "plan", "redesign"],
    "local_fast": ["rename", "fix typo", "docstring", "write test", "small fix"],
}


@pytest.fixture
def policy():
    return PolicyConfig(
        routing=RoutingConfig(
            default_route="cloud_fast",
            thresholds=Thresholds(
                local_fast_max_chars=2000,
                local_fast_max_messages=6,
                cloud_architecture_min_chars=8000,
            ),
            keyword_rules=KEYWORD_RULES,
        ),
        fallback_chains={
            "local_fast": [FallbackEntry(provider="ollama", model="qwen2.5-coder:7b")],
            "cloud_fast": [FallbackEntry(provider="minimax", model="MiniMax-Text-01")],
            "cloud_debug": [FallbackEntry(provider="minimax", model="MiniMax-M1")],
            "cloud_architecture": [FallbackEntry(provider="zhipu", model="glm-4-plus")],
            "cloud_vision": [FallbackEntry(provider="moonshot", model="moonshot-v1-8k")],
        },
    )


def make_features(
    total_chars=100,
    message_count=2,
    has_image=False,
    all_text="",
    last_user_text="",
) -> RequestFeatures:
    return RequestFeatures(
        total_chars=total_chars,
        message_count=message_count,
        has_image=has_image,
        all_text=all_text,
        last_user_text=last_user_text,
    )


# --- Test cases ---

def test_image_routes_to_cloud_vision(policy):
    features = make_features(has_image=True)
    assert route(features, policy) == "cloud_vision"


def test_vision_keyword_routes_to_cloud_vision(policy):
    features = make_features(all_text="please look at this screenshot and fix the layout")
    assert route(features, policy) == "cloud_vision"


def test_debug_keyword_routes_to_cloud_debug(policy):
    features = make_features(
        total_chars=500,
        message_count=3,
        all_text="debug this stacktrace please",
    )
    assert route(features, policy) == "cloud_debug"


def test_architecture_keyword_routes_to_cloud_architecture(policy):
    features = make_features(
        total_chars=500,
        all_text="help me with the system design and architecture of this service",
    )
    assert route(features, policy) == "cloud_architecture"


def test_very_long_prompt_routes_to_cloud_architecture(policy):
    features = make_features(total_chars=10_000, message_count=5, all_text="random long text")
    assert route(features, policy) == "cloud_architecture"


def test_local_keyword_short_prompt_routes_to_local_fast(policy):
    features = make_features(
        total_chars=200,
        message_count=2,
        all_text="please rename variable x to count",
    )
    assert route(features, policy) == "local_fast"


def test_local_keyword_long_prompt_routes_to_cloud_fast(policy):
    """Local keyword + long prompt → default (cloud_fast)."""
    features = make_features(
        total_chars=5000,
        message_count=10,
        all_text="rename something in this huge codebase " * 100,
    )
    assert route(features, policy) == "cloud_fast"


def test_short_prompt_no_keywords_routes_to_local_fast(policy):
    features = make_features(total_chars=100, message_count=2, all_text="hello world")
    assert route(features, policy) == "local_fast"


def test_medium_prompt_no_keywords_routes_to_cloud_fast(policy):
    features = make_features(total_chars=3000, message_count=5, all_text="hello world " * 300)
    assert route(features, policy) == "cloud_fast"


def test_image_takes_priority_over_debug_keyword(policy):
    """Vision rule (has_image) has highest priority."""
    features = make_features(has_image=True, all_text="debug this image stacktrace")
    assert route(features, policy) == "cloud_vision"
