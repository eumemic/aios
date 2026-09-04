"""Cross-process worker watchdog health publication.

The API and worker are separate processes (and normally separate containers), so
worker-local gauges cannot make a deterministic HTTP health signal.  This small
JSON snapshot lives under the workspace root, which is already shared by both
services.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def watchdog_health_path() -> Path:
    """Return the worker/API shared watchdog snapshot path."""
    explicit = os.environ.get("AIOS_WORKER_WATCHDOG_HEALTH_FILE")
    if explicit:
        return Path(explicit)
    workspace = os.environ.get("AIOS_WORKSPACE_ROOT")
    if workspace:
        return Path(workspace) / ".aios-worker-watchdog-health.json"
    # Local/unit-test fallback. Production always configures AIOS_WORKSPACE_ROOT.
    return Path("/tmp/aios-worker-watchdog-health.json")


def publish_gc_health(last_success_at: datetime | None, consecutive_failures: int) -> None:
    """Atomically publish GC health for readers in the API process."""
    path = watchdog_health_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gc_consecutive_failures": consecutive_failures,
        "gc_last_success_at": last_success_at.isoformat() if last_success_at is not None else None,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_gc_health() -> dict[str, Any]:
    """Read a validated snapshot, explicitly distinguishing read failures."""
    try:
        payload = json.loads(watchdog_health_path().read_text(encoding="utf-8"))
        failures = payload["gc_consecutive_failures"]
        last_success = payload["gc_last_success_at"]
        if not isinstance(failures, int) or failures < 0:
            raise ValueError("invalid GC failure count")
        if last_success is not None and not isinstance(last_success, str):
            raise ValueError("invalid GC last-success value")
        return {
            "gc_health_status": "healthy" if failures == 0 else "failing",
            "gc_consecutive_failures": failures,
            "gc_last_success_at": last_success,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "gc_health_status": "unknown",
            "gc_consecutive_failures": None,
            "gc_last_success_at": None,
        }
