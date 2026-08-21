"""Container health probe for connectors built on :mod:`aios_connector_http`."""

from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_HEARTBEAT_PATH = Path("/var/run/aios-connector-alive")
DEFAULT_MAX_AGE_SECONDS = 30.0


def heartbeat_is_fresh(path: Path, *, max_age_seconds: float) -> bool:
    """Return whether ``path`` was touched within ``max_age_seconds``."""
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age <= max_age_seconds


def main() -> None:
    path = Path(os.environ.get("AIOS_CONNECTOR_HEARTBEAT_PATH", DEFAULT_HEARTBEAT_PATH))
    max_age = float(
        os.environ.get("AIOS_CONNECTOR_HEARTBEAT_MAX_AGE_SECONDS", DEFAULT_MAX_AGE_SECONDS)
    )
    if not heartbeat_is_fresh(path, max_age_seconds=max_age):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
