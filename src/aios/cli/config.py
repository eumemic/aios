"""CLI URL resolution.

The CLI reads ``AIOS_URL`` / ``AIOS_API_KEY`` / ``AIOS_API_PORT`` from the
process environment via typer's ``envvar=`` — there is deliberately no
second config surface. For ``.env`` loading, callers run
``set -a; source .env; set +a`` before invoking ``aios``, the same pattern
used for the operator commands (alembic, procrastinate).
"""

from __future__ import annotations

import os


def resolve_base_url(override: str | None) -> str:
    """Pick the effective API base URL.

    Priority: explicit flag > ``AIOS_URL`` env > ``http://127.0.0.1:{AIOS_API_PORT}``.
    The last fallback mirrors the playground pattern where the operator sets
    ``AIOS_API_PORT`` (e.g. 8090) but never ``AIOS_URL`` explicitly.
    """
    if override is not None:
        return override.rstrip("/")
    env_url = os.environ.get("AIOS_URL")
    if env_url:
        return env_url.rstrip("/")
    port = os.environ.get("AIOS_API_PORT", "8080")
    return f"http://127.0.0.1:{port}"
