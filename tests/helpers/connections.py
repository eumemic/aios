"""Shared httpx helpers for e2e tests.

E2E tests routinely build ``httpx`` requests with a bearer token —
operator key (``aios_env["AIOS_API_KEY"]``) or runtime token from
``POST /v1/runtime-tokens``.  Two duplicated patterns recur:

* **Client construction** — ``httpx.AsyncClient(base_url=...,
  headers={"Authorization": f"Bearer {token}"})`` (sometimes with a
  ``transport=`` kwarg for in-process ``ASGITransport`` tests).
  Use :func:`authed_client`.

* **Per-request auth** — bare ``client.post(url,
  headers={"Authorization": f"Bearer {token}"})`` where the client
  is constructed elsewhere and auth varies per call.  Use
  :func:`bearer`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx


def authed_client(base_url: str, token: str, **kwargs: Any) -> httpx.AsyncClient:
    """Bearer-auth ``httpx.AsyncClient`` for e2e tests.

    Extra kwargs flow through to :class:`httpx.AsyncClient` — handy
    for in-process tests that pass ``transport=ASGITransport(...)``.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        **kwargs,
    )


def bearer(token: str) -> dict[str, str]:
    """Bearer-auth header dict for per-request use."""
    return {"Authorization": f"Bearer {token}"}


async def issue_runtime_token(api_key: str, base_url: str, connector: str) -> str:
    """Mint a fresh ``aios_runtime_*`` token scoped to one connector type.

    Used by every e2e test that runs a runtime container against a live
    aios — see ``tests/e2e/test_echo_http_connector.py``,
    ``tests/e2e/test_signal_connector.py``,
    ``tests/e2e/test_telegram_connector.py``.
    """
    async with authed_client(base_url, api_key) as c:
        r = await c.post("/v1/runtime-tokens", json={"connector": connector})
        r.raise_for_status()
        return str(r.json()["plaintext"])


async def create_connection(api_key: str, base_url: str, account: str) -> str:
    """Create a detached ``echo`` connection scoped to ``account`` (the external_account_id).

    Used by e2e tests that need a fresh connection_id without going
    through the auto-create-on-first-inbound supervisor path.
    """
    async with authed_client(base_url, api_key) as c:
        r = await c.post(
            "/v1/connections", json={"connector": "echo", "external_account_id": account}
        )
        r.raise_for_status()
        return str(r.json()["id"])


@contextlib.asynccontextmanager
async def asgi_client(pool: Any) -> AsyncIterator[httpx.AsyncClient]:
    """In-process ``httpx.AsyncClient`` wired to a fresh FastAPI app.

    The app's pool / crypto_box / db_url are populated from settings
    (which the caller's ``aios_env*`` fixture has already mocked); the
    procrastinate handle is a mock since e2e auth tests don't run jobs.
    Shared by tests that build the app themselves rather than going
    through a running uvicorn — see ``tests/e2e/test_account_auth.py``
    and ``tests/e2e/test_accounts_bootstrap.py``.
    """
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(base_url="http://testserver", transport=transport) as client:
        yield client


async def wait_for_health(url: str, *, deadline_s: float = 5.0) -> None:
    """Poll ``<url>/v1/health`` until it returns < 500 or ``deadline_s`` passes.

    Used by e2e fixtures that spawn a uvicorn aios in-process before
    yielding the URL — the polling lets the test wait through uvicorn's
    bring-up race without a fixed sleep.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    async with httpx.AsyncClient() as client:
        while True:
            with contextlib.suppress(httpx.HTTPError):
                r = await client.get(f"{url}/v1/health", timeout=0.5)
                if r.status_code < 500:
                    return
            if loop.time() >= deadline:
                raise TimeoutError(f"server at {url} not ready in {deadline_s}s")
            await asyncio.sleep(0.05)
