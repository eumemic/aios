"""E2E HTTP contract test: re-approving an already-active inbound grant must
return HTTP 409 (not 500) through the real FastAPI exception handler stack.

This drives the real ``POST /v1/connections/{id}/inbound-grants/approve`` and
``/revoke`` routes via the real ``aios.api.routers.connections.router`` mounted
on a minimal FastAPI app, with the real ``install_exception_handlers`` so the
``ConflictError`` → 409 envelope is exercised exactly as in production. The
``AccountIdDep`` and ``PoolDep`` dependencies are overridden to bind a real
migrated DB pool and a fixed account_id (the bearer-auth path is orthogonal to
the bug, which lives in the query CTE).

Procedure E of the test plan for the re-approve-already-active fix.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

from aios.api.deps import get_account_id, get_pool
from aios.api.routers.connections import router as connections_router
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import install_exception_handlers
from aios.models.inbound_policy import RequireApproval

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_reapprove_http"


@pytest.fixture
async def app_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[FastAPI, asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                   VALUES ($1, NULL, TRUE, 'reapprove-http')""",
                _ACCOUNT,
            )
            connection = await queries.insert_connection(
                conn,
                account_id=_ACCOUNT,
                connector="signal",
                external_account_id="reapprove-http",
                metadata={},
            )
            await queries.set_connection_inbound_policy(
                conn,
                connection.id,
                account_id=_ACCOUNT,
                policy=RequireApproval(approved=[]),
            )
        app = FastAPI()
        install_exception_handlers(app)
        app.state.pool = pool
        app.include_router(connections_router)
        app.dependency_overrides[get_pool] = lambda: pool
        app.dependency_overrides[get_account_id] = lambda: _ACCOUNT
        yield app, pool, connection.id
    finally:
        await pool.close()


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_reapprove_already_active_returns_409_envelope(
    app_pool: tuple[FastAPI, asyncpg.Pool[Any], str],
) -> None:
    """approve → revoke → re-approve → re-approve: the 4th approve must return
    HTTP 409 with the aios error envelope, NOT HTTP 500."""
    app, pool, connection_id = app_pool
    async with pool.acquire() as conn:
        await queries.upsert_pending_inbound_grant(
            conn, account_id=_ACCOUNT, connection_id=connection_id, chat_id="alice"
        )
    async with await _client(app) as client:
        approve_url = f"/v1/connections/{connection_id}/inbound-grants/approve"
        revoke_url = f"/v1/connections/{connection_id}/inbound-grants/revoke"
        body = {"chat_id": "alice"}

        r1 = await client.post(approve_url, json=body)
        assert r1.status_code == 200, r1.text
        r2 = await client.post(revoke_url, json=body)
        assert r2.status_code == 200, r2.text
        r3 = await client.post(approve_url, json=body)
        assert r3.status_code == 200, r3.text
        r4 = await client.post(approve_url, json=body)
        assert r4.status_code == 409, (r4.status_code, r4.text)
        envelope = r4.json()
        assert "error" in envelope
        assert envelope["error"]["type"] == "conflict"
        assert "pending/revoked" in envelope["error"]["message"]


async def test_approve_already_active_no_revoked_history_returns_409_envelope(
    app_pool: tuple[FastAPI, asyncpg.Pool[Any], str],
) -> None:
    """pending → active → approve again (no revoked history): HTTP 409 envelope."""
    app, pool, connection_id = app_pool
    async with pool.acquire() as conn:
        await queries.upsert_pending_inbound_grant(
            conn, account_id=_ACCOUNT, connection_id=connection_id, chat_id="alice"
        )
    async with await _client(app) as client:
        approve_url = f"/v1/connections/{connection_id}/inbound-grants/approve"
        body = {"chat_id": "alice"}

        r1 = await client.post(approve_url, json=body)
        assert r1.status_code == 200, r1.text
        r2 = await client.post(approve_url, json=body)
        assert r2.status_code == 409, (r2.status_code, r2.text)
        envelope = r2.json()
        assert envelope["error"]["type"] == "conflict"


async def test_concurrent_reapprove_on_already_active_no_500_no_double_active(
    migrated_db_url: str,
    _reset_db_state: None,
) -> None:
    """Concurrent approve calls on an already-active chat (with revoked history)
    must each resolve to a 409 ConflictError — never a 500 — and never leave two
    active rows. The ``FOR UPDATE`` on ``connections`` plus the CTE-level guard
    keep the approve statement atomic; one request serializes behind the other.

    Procedure F of the test plan (concurrency / idempotency probe).
    """
    import asyncio

    from aios.errors import ConflictError
    from aios.services import connections as connections_service

    pool = await create_pool(migrated_db_url, min_size=4, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                   VALUES ($1, NULL, TRUE, 'conc-reapprove')""",
                _ACCOUNT,
            )
            connection = await queries.insert_connection(
                conn,
                account_id=_ACCOUNT,
                connector="signal",
                external_account_id="conc-reapprove",
                metadata={},
            )
            await queries.set_connection_inbound_policy(
                conn,
                connection.id,
                account_id=_ACCOUNT,
                policy=RequireApproval(approved=[]),
            )
            await queries.upsert_pending_inbound_grant(
                conn, account_id=_ACCOUNT, connection_id=connection.id, chat_id="alice"
            )
        cid = connection.id
        await connections_service.approve_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT)
        await connections_service.revoke_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT)
        await connections_service.approve_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT)

        # Fire several concurrent approves while already active.
        results = await asyncio.gather(
            *[
                connections_service.approve_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT),
                connections_service.approve_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT),
                connections_service.approve_inbound_grant(pool, cid, "alice", account_id=_ACCOUNT),
            ],
            return_exceptions=True,
        )
        for r in results:
            # Every concurrent approve must resolve to ConflictError (409), never
            # an asyncpg UniqueViolationError (which would surface as 500) or any
            # other exception.
            assert isinstance(r, ConflictError), (
                f"expected ConflictError, got {type(r).__name__}: {r!r}"
            )
        async with pool.acquire() as conn:
            active = await conn.fetchval(
                """SELECT count(*) FROM inbound_grants
                    WHERE connection_id=$1 AND chat_id='alice' AND account_id=$2 AND status='active'""",
                cid,
                _ACCOUNT,
            )
            assert active == 1, f"expected exactly one active row, got {active}"
    finally:
        await pool.close()
