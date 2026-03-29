from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.settings import POLICY_FILE, PROVIDERS_FILE, ROUTER_PORT
from app.core.policy_loader import load_policy, get_providers
from app.adapters.ollama_adapter import OllamaAdapter
from app.adapters.openai_compatible_adapter import OpenAICompatibleAdapter
from app.api import chat, health, router_admin


def build_adapters(providers_config) -> dict:
    """Instantiate adapters based on providers.yaml."""
    adapters = {}
    for name, cfg in providers_config.providers.items():
        if cfg.type == "ollama":
            adapters[name] = OllamaAdapter(base_url=cfg.base_url)
        elif cfg.type == "openai_compatible":
            adapters[name] = OpenAICompatibleAdapter(
                base_url=cfg.base_url,
                api_key_env=cfg.api_key_env,
            )
    return adapters


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load config and build adapters
    load_policy(POLICY_FILE, PROVIDERS_FILE)
    providers = get_providers()
    app.state.adapters = build_adapters(providers)
    yield
    # Shutdown: nothing to clean up for now


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenCode Smart Router",
        description="OpenAI-compatible routing proxy for OpenCode",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(router_admin.router)
    app.include_router(chat.router)
    
    app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

    return app


app = create_app()
