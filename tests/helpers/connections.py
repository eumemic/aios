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
import functools
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from unittest import mock

import httpx

if TYPE_CHECKING:
    import asyncpg
    from fastapi import FastAPI


@functools.cache
def _shared_app() -> FastAPI:
    """The one FastAPI app this test process serves — built on first use.

    ``create_app()`` costs ~0.3 s (router registration plus fastapi-mcp's
    OpenAPI build for the ``/mcp`` mount).  Paying that per test made app
    construction the dominant cost of the e2e suite, so every app-wiring
    fixture shares this instance via :func:`wired_app`.  Sharing is sound
    because tests touch nothing on the app beyond ``app.state``, and
    request handlers read settings per request (``get_settings`` is
    cache-cleared per test by ``aios_env_minimal``).  Under pytest-xdist
    each worker process gets its own instance.

    The load-bearing invariant: tests must NEVER mutate this app beyond
    ``app.state`` — no ``dependency_overrides``, no route/middleware
    registration.  Any of those would leak into every later test in the
    process.  A test that genuinely needs them should build its own
    ``FastAPI()`` (see ``tests/unit/test_sse_preflight.py``).
    """
    from aios.api.app import create_app

    return create_app()


def wired_app(pool: Any) -> FastAPI:
    """The shared app with this test's state bound.

    ``crypto_box`` / ``db_url`` come from settings (already mocked by the
    caller's ``aios_env*`` fixture); ``procrastinate`` is a fresh
    ``MagicMock`` per call so mock assertions never see another test's
    wake deferrals.
    """
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = _shared_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()
    # The real lifespan flips this True only after the boot-admission gate
    # (#1575) proves the DB safe; these helpers bind state directly without
    # running the lifespan, and the test DB is migrated to head with no
    # retired-token residue, so mark the process admitted so ``/ready`` and
    # the routes behave as they would post-gate.
    app.state.retirements_ok = True
    return app


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


async def mint_runtime_token_via_db(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account_id: str = "acc_test_stub",
    label: str | None = None,
) -> str:
    """Mint a runtime token by direct service call (not HTTP).

    Use this in tests that must NOT hit uvicorn before the test's first
    HTTP request — e.g. the SSE-first-open regression for #377. The
    standard :func:`issue_runtime_token` POSTs to ``/v1/runtime-tokens``,
    which itself warms up uvicorn (lazy lifespan + connection init) and
    defeats the bug repro. This helper inserts the token row directly
    against ``pool`` via the runtime-tokens service so the test's first
    network request to the server is the SSE GET under examination.
    """
    from aios.services import runtime_tokens as runtime_tokens_service

    _, plaintext = await runtime_tokens_service.issue(
        pool, account_id=account_id, connector=connector, label=label
    )
    return plaintext


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


async def admit_inbound_all(pool: Any, connection_id: str) -> None:
    """Set a connection's ``inbound_policy`` to ``AllowAll`` directly in the DB.

    The inbound-admission gate (#1500) flipped the connector inbound path
    fail-closed: a connection with ``inbound_policy IS NULL`` resolves to the
    server default ``DenyAll`` and every inbound is dropped with
    ``DENIED_BY_POLICY`` (HTTP 422) *before* any side effect. There is no
    operator endpoint to set the policy yet (deliberately deferred to the
    operator-surface PR), so e2e tests that create a fresh connection and then
    POST to ``/v1/connectors/runtime/inbound`` must seed an admitting policy
    out-of-band. This writes ``{"kind":"allow_all"}`` so the gate admits every
    ``chat_id`` for the connection.
    """
    await pool.execute(
        'UPDATE connections SET inbound_policy = \'{"kind":"allow_all"}\'::jsonb WHERE id = $1',
        connection_id,
    )


@contextlib.asynccontextmanager
async def asgi_client(pool: Any) -> AsyncIterator[httpx.AsyncClient]:
    """In-process ``httpx.AsyncClient`` wired to the shared FastAPI app.

    State binding is :func:`wired_app`'s; the procrastinate handle is a
    mock since e2e auth tests don't run jobs.  Shared by tests that build
    the app themselves rather than going through a running uvicorn — see
    ``tests/e2e/test_account_auth.py`` and
    ``tests/e2e/test_accounts_bootstrap.py``.
    """
    transport = httpx.ASGITransport(app=wired_app(pool))
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
