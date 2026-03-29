"""
Integration-style tests for the FastAPI app endpoints.
"""
import pytest
import time
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.schemas.openai_chat import ChatCompletionResponse, Choice, Message, Usage
from app.schemas.router_policy import PolicyConfig, RoutingConfig, Thresholds, FallbackEntry
from app.core import policy_loader


def make_response(model="test-model") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="test-id",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Test reply"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


@pytest.fixture
def mock_policy():
    return PolicyConfig(
        routing=RoutingConfig(
            default_route="cloud_fast",
            thresholds=Thresholds(
                local_fast_max_chars=2000,
                local_fast_max_messages=6,
                cloud_architecture_min_chars=8000,
            ),
            keyword_rules={
                "cloud_vision": ["image"],
                "cloud_debug": ["debug"],
                "cloud_architecture": ["architecture"],
                "local_fast": ["rename"],
            },
        ),
        fallback_chains={
            "local_fast": [FallbackEntry(provider="mock", model="mock-model")],
            "cloud_fast": [FallbackEntry(provider="mock", model="mock-model")],
            "cloud_debug": [FallbackEntry(provider="mock", model="mock-model")],
            "cloud_architecture": [FallbackEntry(provider="mock", model="mock-model")],
            "cloud_vision": [FallbackEntry(provider="mock", model="mock-model")],
        },
    )


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.chat = AsyncMock(return_value=make_response())
    adapter.is_available = AsyncMock(return_value=True)
    return adapter


@pytest.fixture
def mock_providers():
    from app.schemas.router_policy import ProvidersConfig, ProviderConfig
    return ProvidersConfig(providers={
        "mock": ProviderConfig(base_url="http://mock", type="openai_compatible", api_key_env=None),
    })


@pytest.fixture
def app(mock_policy, mock_providers, mock_adapter, monkeypatch):
    # Patch policy loader to return our mock policy and providers
    monkeypatch.setattr(policy_loader, "_policy", mock_policy)
    monkeypatch.setattr(policy_loader, "_providers", mock_providers)
    monkeypatch.setattr(policy_loader, "_policy_loaded_at", "2026-01-01T00:00:00+00:00")

    test_app = create_app()
    # Bypass lifespan by setting adapters directly
    test_app.state.adapters = {"mock": mock_adapter}
    return test_app


@pytest.mark.asyncio
async def test_health_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_chat_completions_returns_openai_format(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "rename x to count"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_models_endpoint_returns_list(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


@pytest.mark.asyncio
async def test_router_status_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/router/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "route_classes" in data
    assert "providers" in data
    assert "policy_loaded_at" in data


@pytest.mark.asyncio
async def test_router_policy_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/router/policy")
    assert resp.status_code == 200
    data = resp.json()
    assert "routing" in data
    assert "fallback_chains" in data


@pytest.mark.asyncio
async def test_chat_completions_all_adapters_fail_returns_502(app, mock_adapter):
    """When all adapters fail, endpoint should return 502."""
    import httpx as _httpx
    mock_adapter.chat = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "rename x to count"}],
            },
        )
    assert resp.status_code == 502
