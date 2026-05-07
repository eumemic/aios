"""Convenience constructors for the aios SDK Client."""

from __future__ import annotations

import os

from aios.cli.config import resolve_base_url
from aios.sdk._generated import AuthenticatedClient


def client_from_env() -> AuthenticatedClient:
    """Build a Client from ``AIOS_URL`` (or ``AIOS_API_PORT``) and ``AIOS_API_KEY``.

    URL resolution mirrors the CLI: explicit env > ``http://127.0.0.1:{AIOS_API_PORT}``
    > ``http://127.0.0.1:8080``. Reuses
    :func:`aios.cli.config.resolve_base_url` so CLI and SDK never disagree
    about the resolved base URL.

    Raises ``RuntimeError`` if ``AIOS_API_KEY`` isn't set — the SDK uses
    Bearer auth on every endpoint, so an unset key is operator error rather
    than a usable default.
    """
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AIOS_API_KEY is not set. Export it, or pass token=... to "
            "AuthenticatedClient(...) directly."
        )
    return AuthenticatedClient(base_url=resolve_base_url(None), token=api_key)
