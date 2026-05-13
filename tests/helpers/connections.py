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
from typing import Any

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
