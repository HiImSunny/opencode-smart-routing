"""
Smoke test — run against a live server at http://localhost:1234.
Usage: python scripts/smoke_test.py
"""
import httpx
import json
import sys

BASE = "http://localhost:1234"


def check(label: str, cond: bool, detail: str = ""):
    if cond:
        print(f"✅ {label}")
    else:
        print(f"❌ {label}" + (f": {detail}" if detail else ""))
        sys.exit(1)


# 1. Health
resp = httpx.get(f"{BASE}/health")
check("Health endpoint returns 200", resp.status_code == 200, resp.text)
data = resp.json()
check("Health returns status=ok", data.get("status") == "ok")

# 2. Models
resp = httpx.get(f"{BASE}/models")
check("/models returns 200", resp.status_code == 200, resp.text)
data = resp.json()
check("/models returns a list", isinstance(data.get("data"), list) and len(data["data"]) > 0)

# 3. Router status
resp = httpx.get(f"{BASE}/router/status")
check("/router/status returns 200", resp.status_code == 200, resp.text)
data = resp.json()
check("/router/status has route_classes", "route_classes" in data)

# 4. Router policy
resp = httpx.get(f"{BASE}/router/policy")
check("/router/policy returns 200", resp.status_code == 200, resp.text)

# 5. Chat completion (short prompt → local_fast)
resp = httpx.post(
    f"{BASE}/v1/chat/completions",
    json={
        "model": "auto",
        "messages": [{"role": "user", "content": "rename the variable x to count"}],
    },
    timeout=90,
)
check("POST /v1/chat/completions returns 200", resp.status_code == 200, resp.text)
data = resp.json()
check("Response has choices", "choices" in data and len(data["choices"]) > 0)
check("Response has model field", "model" in data)

model_used = data.get("model", "unknown")
reply = data["choices"][0]["message"]["content"][:80]
print(f"\n   Model used : {model_used}")
print(f"   Reply      : {reply}")

# 6. Reload
resp = httpx.post(f"{BASE}/router/reload")
check("POST /router/reload returns 200", resp.status_code == 200, resp.text)

print("\n🎉 All smoke tests passed!")
