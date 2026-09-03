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


def _parse_connection_health(path: Path) -> tuple[list[str], list[str]] | None:
    """Return validated connection state, or ``None`` for unreadable content."""
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None

    healthy = payload.get("healthy_connection_ids")
    unhealthy = payload.get("unhealthy_connection_ids")
    if not all(
        isinstance(values, list) and all(isinstance(value, str) for value in values)
        for values in (healthy, unhealthy)
    ):
        return None
    return healthy, unhealthy


def read_connection_health(path: Path) -> tuple[list[str], list[str]]:
    """Read connection-correlated transport state from a heartbeat.

    Invalid content remains represented as empty state for callers which only
    consume attribution.  The health probe uses the validity-aware parser and
    fails closed instead of confusing invalid content with valid empty state.
    """
    return _parse_connection_health(path) or ([], [])


def main() -> None:
    path = resolve_heartbeat_path()
    max_age = heartbeat_max_age_seconds()
    parsed = _parse_connection_health(path)
    valid = parsed is not None
    healthy, unhealthy = parsed or ([], [])
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
    if not valid or unhealthy or not heartbeat_is_fresh(path, max_age_seconds=max_age):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
