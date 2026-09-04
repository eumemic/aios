"""End-to-end regression for the lane_activate session-update 422 fix.

Drives the REAL ``PUT /v1/sessions/{id}`` route (the FastAPI body-validation
gate that produces the 422) against the testcontainer-Postgres harness, with
both the FIXED body (no ``archive_when_idle``) and the BODY THE BUG SHIPPED
(with ``archive_when_idle``). This is the strongest in-repo evidence the
production 422 is gone: the route binds ``body: SessionUpdate``
(``extra="forbid"``, no ``archive_when_idle`` field), so the buggy body is
rejected at the router before any service runs, and the fixed body is accepted
and converges the live session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

from aios.api.deps import require_bearer_auth
from aios.api.routers import sessions as sessions_router
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


def _build_app(pool: asyncpg.Pool[Any], crypto_box: CryptoBox) -> FastAPI:
    """A minimal FastAPI app mounting only the sessions router.

    ``require_bearer_auth`` is overridden to return the bootstrapped root
    account (``acc_test_stub`` seeded by the ``aios_env`` fixture) so the
    route's ``AccountIdDep`` resolves without a real bearer-token round-trip;
    the body-validation gate under test runs unchanged.
    """
    app = FastAPI()
    app.state.pool = pool
    app.state.crypto_box = crypto_box

    async def _override_auth() -> tuple[str, str, bool]:
        return ("acc_test_stub", "akey_test", True)

    app.dependency_overrides[require_bearer_auth] = _override_auth
    app.include_router(sessions_router.router)
    return app


@pytest.fixture
async def harness(
    aios_env: dict[str, str], migrated_db_url: str, _reset_db_state: Any
) -> AsyncIterator[tuple[FastAPI, str, str]]:
    """A real sessions-router app + a seeded live session to PUT against.

    Seeds (agent, environment, session) on the bootstrapped root account with
    ``title="t"``, ``archive_when_idle=False``, ``vault_ids=[]`` — the shape the
    lane lock would converge. Yields ``(app, session_id, account_id)``.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            account_id = "acc_test_stub"
            agent = await agents_service.create_agent(
                pool,
                account_id=account_id,
                name="lane-harness-agent",
                model="openrouter/test",
                system="",
                tools=[],
                description=None,
                metadata={},
                window_min=50_000,
                window_max=150_000,
            )
            env = await environments_service.create_environment(
                pool, account_id=account_id, name="lane-harness-env"
            )
            session = await queries.insert_session(
                conn,
                account_id=account_id,
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                title="t",
                metadata={},
                archive_when_idle=False,
            )
        crypto_box = CryptoBox.from_base64(aios_env["AIOS_VAULT_KEY"])
        app = _build_app(pool, crypto_box)
        try:
            yield app, session.id, account_id
        finally:
            await httpx.AsyncClient(transport=httpx.ASGITransport(app=app)).aclose()
    finally:
        await pool.close()


async def test_fixed_body_converges_via_real_route(
    harness: tuple[FastAPI, str, str],
) -> None:
    """The FIXED PUT body (no archive_when_idle) is accepted: 200, title updated."""
    app, session_id, _account_id = harness
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.put(
            f"/v1/sessions/{session_id}", json={"title": "t-new", "vault_ids": []}
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "t-new"


async def test_buggy_body_with_archive_when_idle_is_rejected_422(
    harness: tuple[FastAPI, str, str],
) -> None:
    """The body the bug shipped (with archive_when_idle) is rejected at the route."""
    app, session_id, _account_id = harness
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.put(
            f"/v1/sessions/{session_id}",
            json={"title": "t-new", "vault_ids": [], "archive_when_idle": False},
        )
    assert resp.status_code == 422, resp.text
    # The rejection is the extra-forbidden gate on SessionUpdate, not a
    # service-layer error: the response body names the forbidden key.
    body = resp.json()
    detail = str(body)
    assert "archive_when_idle" in detail
