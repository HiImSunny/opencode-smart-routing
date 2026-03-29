# opencode-smart-router

> OpenAI-compatible HTTP proxy for [OpenCode](https://opencode.ai) — routes requests to Ollama (local) or cloud LLM backends based on rule-based policy, with automatic fallback chains and full telemetry.

```
OpenCode
  └─► Smart Router (localhost:4000/v1)
        ├─ Feature Extractor  (chars, keywords, image?)
        ├─ Router Engine      (5 route classes)
        ├─ Fallback Manager   (chain with retry)
        ├─ Provider Adapters  (Ollama / OpenAI-compatible)
        └─ Telemetry Logger   (JSONL, daily rotation)
```

---

## Route Classes

| Route Class | Trigger | Primary Backend |
|---|---|---|
| `cloud_vision` | `has_image` or vision keywords | Moonshot / GLM-4V |
| `cloud_architecture` | architecture keywords or `chars ≥ 8000` | GLM-4-plus / MiniMax M1 |
| `cloud_debug` | debug / security keywords | MiniMax M1 / GLM-4-plus |
| `local_fast` | simple keywords + `chars ≤ 2000` + `msgs ≤ 6` | Ollama qwen2.5-coder:7b |
| `cloud_fast` | everything else | MiniMax Text-01 |

---

## Quick Start

### 1. Install dependencies

```bash
pip install poetry
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Start the router

```bash
# Linux/Mac
bash scripts/run_dev.sh

# Windows (PowerShell)
$env:ROUTER_PORT=4000; uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload
```

### 4. Configure OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "smart-router": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "local"
      },
      "models": {
        "auto": {}
      }
    }
  }
}
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main chat endpoint (OpenAI-compatible) |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all configured models |
| `GET` | `/router/status` | Provider availability & loaded policy |
| `GET` | `/router/policy` | Parsed policy as JSON |
| `POST` | `/router/reload` | Hot-reload `policy.yaml` & `providers.yaml` |

---

## Configuration

### `config/policy.yaml`

Controls routing thresholds, keyword rules, fallback chains, timeouts, and cooldown.

### `config/providers.yaml`

Defines backends (Ollama, MiniMax, ZhiPu, Moonshot). API keys are read from environment variables.

---

## Running Tests

```bash
poetry run pytest -v
```

## Smoke Test (against running server)

```bash
python scripts/smoke_test.py
```

---

## Telemetry

Every request writes a JSON record to `logs/requests-YYYY-MM-DD.jsonl`:

```json
{
  "request_id": "...",
  "timestamp": 1711704000.0,
  "route_class": "local_fast",
  "chosen_provider": "ollama",
  "chosen_model": "qwen2.5-coder:7b",
  "fallback_index": 0,
  "input_chars": 42,
  "has_image": false,
  "message_count": 2,
  "latency_ms": 1234.5,
  "status": "success",
  "errors": [],
  "stream": false
}
```

---

## Project Structure

```
opencode-smart-router/
├─ app/
│   ├─ main.py                    # FastAPI app + lifespan
│   ├─ api/
│   │   ├─ chat.py                # POST /v1/chat/completions
│   │   ├─ health.py              # GET /health
│   │   └─ router_admin.py       # Admin endpoints
│   ├─ core/
│   │   ├─ settings.py
│   │   ├─ policy_loader.py
│   │   ├─ feature_extractor.py
│   │   ├─ router_engine.py
│   │   └─ fallback_manager.py
│   ├─ adapters/
│   │   ├─ base.py
│   │   ├─ ollama_adapter.py
│   │   └─ openai_compatible_adapter.py
│   ├─ telemetry/
│   │   ├─ models.py
│   │   └─ logger.py
│   └─ schemas/
│       ├─ openai_chat.py
│       └─ router_policy.py
├─ config/
│   ├─ policy.yaml
│   └─ providers.yaml
├─ tests/
│   ├─ test_routing_rules.py
│   ├─ test_fallbacks.py
│   └─ test_openai_format.py
├─ scripts/
│   ├─ run_dev.sh
│   └─ smoke_test.py
├─ logs/
├─ .env.example
├─ pyproject.toml
└─ README.md
```
