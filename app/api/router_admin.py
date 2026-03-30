import asyncio
from fastapi import APIRouter, Request, HTTPException
from app.core.policy_loader import (
    get_policy, get_providers, get_policy_loaded_at, load_policy,
    get_ollama_model, set_ollama_model,
)

from app.core.settings import POLICY_FILE, PROVIDERS_FILE, LOG_DIR
import os
import json
from datetime import datetime, timezone

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

    policy = get_policy()

    # Check availability of each provider concurrently
    async def check(name, adapter):
        result = {"status": "unavailable", "warnings": []}
        try:
            available = await adapter.is_available()
            if available:
                result["status"] = "available"
                # If ollama, check for missing needed models
                if name == "ollama" and hasattr(adapter, "get_installed_models"):
                    installed = await adapter.get_installed_models()
                    needed_models = {
                        entry.model
                        for chain in policy.fallback_chains.values()
                        for entry in chain
                        if entry.provider == "ollama"
                    }
                    missing = needed_models - set(installed)
                    if missing:
                        result["warnings"].append(f"Missing models: {', '.join(missing)}")
                        
        except Exception as e:
            result["status"] = "error"
            result["warnings"].append(str(e))
        return name, result

    results = await asyncio.gather(*[check(n, a) for n, a in adapters.items()])
    provider_status = dict(results)

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

@router.get("/telemetry/logs")
async def get_telemetry_logs(date: str = None, page: int = 1, limit: int = 50):
    """Return telemetry logs with pagination and date filter."""
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
    log_path = os.path.join(LOG_DIR, f"requests-{date}.jsonl")
    
    if not os.path.exists(log_path):
        return {"logs": [], "total": 0, "page": page, "limit": limit}
        
    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    total = len(lines)
    
    # Reverse the lines so newest is first
    lines = list(reversed(lines))
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    for line in lines[start_idx:end_idx]:
        try:
            logs.append(json.loads(line))
        except:
            pass
            
    return {"logs": logs, "total": total, "page": page, "limit": limit, "date": date}

@router.get("/telemetry/dates")
async def get_log_dates():
    """Return a list of available dates from the logs directory."""
    try:
        if not os.path.exists(LOG_DIR):
            return {"dates": []}
        files = os.listdir(LOG_DIR)
        dates = []
        for file in files:
            if file.startswith("requests-") and file.endswith(".jsonl"):
                date_str = file[9:-6]
                dates.append(date_str)
        dates.sort(reverse=True)
        return {"dates": dates}
    except Exception:
        return {"dates": []}


# ──────────────────────────────────────────────
#  Ollama Model Management
# ──────────────────────────────────────────────

@router.get("/router/ollama/models")
async def list_ollama_models(http_request: Request):
    """Return all models currently installed in the local Ollama instance."""
    adapters = http_request.app.state.adapters
    ollama = adapters.get("ollama")
    if ollama is None:
        raise HTTPException(status_code=404, detail="Ollama adapter not configured")
    try:
        models = await ollama.get_installed_models()
        current = get_ollama_model()
        return {"models": models, "current": current}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")


@router.get("/router/ollama/model")
async def get_current_ollama_model():
    """Return the model currently active for all Ollama fallback entries."""
    return {"model": get_ollama_model()}


@router.post("/router/ollama/model")
async def update_ollama_model(http_request: Request, body: dict):
    """
    Hot-swap the Ollama model in all fallback chains (no restart needed).
    Also invalidates the Ollama adapter's availability cache so the new
    model is picked up immediately.

    Body: {"model": "qwen2.5-coder:7b"}
    """
    model = (body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=422, detail="'model' field is required")

    try:
        updated = set_ollama_model(model)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Invalidate the availability cache so is_available() re-checks immediately
    adapters = http_request.app.state.adapters
    ollama = adapters.get("ollama")
    if ollama and hasattr(ollama, "_available_cache"):
        ollama._available_cache = None

    return {
        "status": "ok",
        "model": model,
        "chains_updated": updated,
        "message": f"Ollama model switched to '{model}' across {updated} chain(s). "
                   "Add OLLAMA_MODEL={model} to .env to persist after restart.",
    }

