"""SDK-level configuration helpers — env-resolution rules shared with the CLI."""

from __future__ import annotations

import os


def resolve_base_url(override: str | None = None) -> str:
    """Pick the effective API base URL.

    Priority: explicit override > ``AIOS_URL`` env > ``http://127.0.0.1:{AIOS_API_PORT}``.
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
