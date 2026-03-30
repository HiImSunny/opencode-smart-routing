import os
from dotenv import load_dotenv

load_dotenv()

ROUTER_PORT: int = int(os.getenv("ROUTER_PORT", "1234"))
LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
CONFIG_DIR: str = os.getenv("CONFIG_DIR", "./config")

POLICY_FILE: str = os.path.join(CONFIG_DIR, "policy.yaml")
PROVIDERS_FILE: str = os.path.join(CONFIG_DIR, "providers.yaml")

# Default Ollama model — can be overridden at runtime via the admin API
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

VERSION: str = "1.0.0"
