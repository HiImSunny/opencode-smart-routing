import asyncio
from fastapi import APIRouter, Request, HTTPException
from app.core.policy_loader import get_policy, get_providers, get_policy_loaded_at, load_policy
from app.core.settings import POLICY_FILE, PROVIDERS_FILE

router = APIRouter()


@router.get("/models")
async def list_models(http_request: Request):
    """Return all models from providers.yaml in OpenAI /v1/models format."""
    providers = get_providers()
    policy = get_policy()

    model_list = []
    seen = set()

    # Collect all models referenced in fallback chains
    for chain in policy.fallback_chains.values():
        for entry in chain:
            key = f"{entry.provider}/{entry.model}"
            if key not in seen:
                seen.add(key)
                model_list.append({
                    "id": entry.model,
                    "object": "model",
                    "owned_by": entry.provider,
                    "created": 0,
                })

    return {
        "object": "list",
        "data": model_list,
    }


@router.get("/router/status")
async def router_status(http_request: Request):
    """Return current router state and provider availability."""
    adapters = http_request.app.state.adapters

    # Check availability of each provider concurrently
    async def check(name, adapter):
        try:
            available = await adapter.is_available()
            return name, "available" if available else "unavailable"
        except Exception:
            return name, "error"

    results = await asyncio.gather(*[check(n, a) for n, a in adapters.items()])
    provider_status = dict(results)

    policy = get_policy()
    return {
        "route_classes": list(policy.fallback_chains.keys()),
        "providers": provider_status,
        "policy_loaded_at": get_policy_loaded_at(),
    }


@router.get("/router/policy")
async def router_policy():
    """Return the parsed policy as JSON."""
    policy = get_policy()
    return policy.model_dump()


@router.post("/router/reload")
async def reload_policy():
    """Hot-reload policy.yaml and providers.yaml without restarting."""
    try:
        load_policy(POLICY_FILE, PROVIDERS_FILE)
        return {"status": "reloaded", "policy_loaded_at": get_policy_loaded_at()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
