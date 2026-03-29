import os
from dotenv import load_dotenv

load_dotenv()

ROUTER_PORT: int = int(os.getenv("ROUTER_PORT", "1234"))
LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
CONFIG_DIR: str = os.getenv("CONFIG_DIR", "./config")

POLICY_FILE: str = os.path.join(CONFIG_DIR, "policy.yaml")
PROVIDERS_FILE: str = os.path.join(CONFIG_DIR, "providers.yaml")

VERSION: str = "1.0.0"
