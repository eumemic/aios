"""Convenience constructors for the aios SDK Client."""

from __future__ import annotations

import os

from aios_sdk._generated import AuthenticatedClient
from aios_sdk.config import resolve_base_url


def client_from_env() -> AuthenticatedClient:
    """Build a Client from ``AIOS_URL`` (or ``AIOS_API_PORT``) and ``AIOS_API_KEY``.

    URL resolution mirrors the CLI: explicit env > ``http://127.0.0.1:{AIOS_API_PORT}``
    > ``http://127.0.0.1:8080``. The SDK and the CLI share
    :func:`aios_sdk.config.resolve_base_url` so they never disagree about
    the resolved base URL.

    Raises ``RuntimeError`` if ``AIOS_API_KEY`` isn't set — the SDK uses
    Bearer auth on every endpoint, so an unset key is operator error rather
    than a usable default. (The CLI's ``CliState.sdk_client()`` accepts an
    unset key on purpose — the ``aios status`` command needs to probe an
    unauthenticated ``/health`` endpoint and report auth-key state. Plugins
    and scripts have no equivalent need.)
    """
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AIOS_API_KEY is not set. Export it, or pass token=... to "
            "AuthenticatedClient(...) directly."
        )
    return AuthenticatedClient(base_url=resolve_base_url(), token=api_key)
