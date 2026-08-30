"""Container health probe for connectors built on :mod:`aios_connector_http`."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

DEFAULT_HEARTBEAT_PATH = Path("/var/run/aios-connector-alive")
DEFAULT_MAX_AGE_SECONDS = 30.0


def resolve_heartbeat_path() -> Path:
    """Return the probe path, using a writable location outside containers."""
    configured = os.environ.get("AIOS_CONNECTOR_HEARTBEAT_PATH")
    if configured:
        return Path(configured)
    if Path("/.dockerenv").exists():
        return DEFAULT_HEARTBEAT_PATH
    return Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / DEFAULT_HEARTBEAT_PATH.name


def heartbeat_max_age_seconds() -> float:
    """Return the age at which the container probe considers a heartbeat stale."""
    return float(
        os.environ.get("AIOS_CONNECTOR_HEARTBEAT_MAX_AGE_SECONDS", DEFAULT_MAX_AGE_SECONDS)
    )


def heartbeat_is_fresh(path: Path, *, max_age_seconds: float) -> bool:
    """Return whether ``path`` was touched within ``max_age_seconds``."""
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age <= max_age_seconds


def read_connection_health(path: Path) -> tuple[list[str], list[str]]:
    """Read connection-correlated transport state from a heartbeat."""
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return [], []
    if not isinstance(payload, dict):
        return [], []

    def ids(key: str) -> list[str]:
        values = payload.get(key, [])
        return [str(value) for value in values] if isinstance(values, list) else []

    return ids("healthy_connection_ids"), ids("unhealthy_connection_ids")


def main() -> None:
    path = resolve_heartbeat_path()
    max_age = heartbeat_max_age_seconds()
    healthy, unhealthy = read_connection_health(path)
    # Docker retains probe output in State.Health.Log.  The external reader
    # consumes this machine-readable line to attribute container health to the
    # affected connection rather than its healthy siblings.
    print(
        json.dumps(
            {
                "healthy_connection_ids": healthy,
                "unhealthy_connection_ids": unhealthy,
            },
            sort_keys=True,
        )
    )
    if unhealthy or not heartbeat_is_fresh(path, max_age_seconds=max_age):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
