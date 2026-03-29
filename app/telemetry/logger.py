import asyncio
import json
import os
import dataclasses
from datetime import datetime, timezone
from app.telemetry.models import TelemetryRecord

_lock = asyncio.Lock()


def _get_log_path(log_dir: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"requests-{today}.jsonl")


async def log_record(record: TelemetryRecord, log_dir: str) -> None:
    """Write a single telemetry record as a JSON line. Thread-safe via asyncio.Lock."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = _get_log_path(log_dir)

    record_dict = dataclasses.asdict(record)
    line = json.dumps(record_dict, ensure_ascii=False)

    async with _lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
