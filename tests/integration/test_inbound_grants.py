"""Stateful coverage for the runtime inbound-approval ledger."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.models.inbound_policy import RequireApproval
from aios.services import connections as connections_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def grants_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                   VALUES ('grant-root', NULL, TRUE, 'root'),
                          ('grant-a', 'grant-root', FALSE, 'a'),
                          ('grant-b', 'grant-root', FALSE, 'b')"""
            )
        yield pool
    finally:
        await pool.close()


async def _connection(pool: asyncpg.Pool[Any]) -> str:
    async with pool.acquire() as conn:
        connection = await queries.insert_connection(
            conn,
            account_id="grant-a",
            connector="signal",
            external_account_id="grant-test",
            metadata={},
        )
        await queries.set_connection_inbound_policy(
            conn,
            connection.id,
            account_id="grant-a",
            policy=RequireApproval(approved=[]),
        )
    return connection.id


async def test_approve_revoke_reapprove_preserves_revoked_history(
    grants_pool: asyncpg.Pool[Any],
) -> None:
    connection_id = await _connection(grants_pool)
    async with grants_pool.acquire() as conn:
        await queries.upsert_pending_inbound_grant(
            conn,
            account_id="grant-a",
            connection_id=connection_id,
            chat_id="alice",
        )
        # Repeated denied messages are idempotent while the grant is live.
        await queries.upsert_pending_inbound_grant(
            conn,
            account_id="grant-a",
            connection_id=connection_id,
            chat_id="alice",
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM inbound_grants WHERE connection_id=$1 AND chat_id='alice'",
                connection_id,
            )
            == 1
        )

    first = await connections_service.approve_inbound_grant(
        grants_pool, connection_id, "alice", account_id="grant-a"
    )
    _, _, session_a = await seed_agent_env_session(
        grants_pool, account_id="grant-a", prefix="grant-a"
    )
    async with grants_pool.acquire() as conn:
        await queries.insert_chat_session(
            conn,
            account_id="grant-a",
            connection_id=connection_id,
            chat_id="alice",
            session_id=session_a.id,
        )
    revoked = await connections_service.revoke_inbound_grant(
        grants_pool, connection_id, "alice", account_id="grant-a"
    )
    async with grants_pool.acquire() as conn:
        assert (
            await queries.lookup_chat_session(conn, connection_id, "alice", account_id="grant-a")
            is None
        )
        await queries.insert_chat_session(
            conn,
            account_id="grant-a",
            connection_id=connection_id,
            chat_id="alice",
            session_id=session_a.id,
        )
    second = await connections_service.approve_inbound_grant(
        grants_pool, connection_id, "alice", account_id="grant-a"
    )
    assert first.id == revoked.id
    assert second.id != revoked.id

    async with grants_pool.acquire() as conn:
        assert (
            await queries.lookup_chat_session(conn, connection_id, "alice", account_id="grant-a")
            is None
        )
        rows = await conn.fetch(
            """SELECT id, status FROM inbound_grants
                WHERE connection_id=$1 AND chat_id='alice' ORDER BY created_at, id""",
            connection_id,
        )
        assert sorted(row["status"] for row in rows) == ["active", "revoked"]
        assert any(row["id"] == revoked.id and row["status"] == "revoked" for row in rows)
        policy = await conn.fetchval(
            "SELECT inbound_policy FROM connections WHERE id=$1", connection_id
        )
        assert policy == {"kind": "require_approval", "approved": ["alice"]}


async def test_pending_list_validates_connection_and_gc_only_reaps_pending(
    grants_pool: asyncpg.Pool[Any],
) -> None:
    connection_id = await _connection(grants_pool)
    with pytest.raises(NotFoundError):
        await connections_service.list_pending_inbound_grants(
            grants_pool, "missing", account_id="grant-a"
        )
    with pytest.raises(NotFoundError):
        await connections_service.list_pending_inbound_grants(
            grants_pool, connection_id, account_id="grant-b"
        )

    async with grants_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO inbound_grants
                   (account_id, connection_id, chat_id, status, created_at)
               VALUES ('grant-a',$1,'old-pending','pending',now()-interval '2 days'),
                      ('grant-a',$1,'old-active','active',now()-interval '2 days'),
                      ('grant-a',$1,'old-revoked','revoked',now()-interval '2 days')""",
            connection_id,
        )
        assert await queries.reap_pending_inbound_grants(conn, ttl=timedelta(days=1)) == 1
        rows = await conn.fetch(
            "SELECT chat_id, status FROM inbound_grants WHERE connection_id=$1",
            connection_id,
        )
        statuses = {row["chat_id"]: row["status"] for row in rows}
        assert statuses == {"old-active": "active", "old-revoked": "revoked"}
        await queries.upsert_pending_inbound_grant(
            conn,
            account_id="grant-a",
            connection_id=connection_id,
            chat_id="old-pending",
        )
    pending = await connections_service.list_pending_inbound_grants(
        grants_pool, connection_id, account_id="grant-a"
    )
    assert [grant.chat_id for grant in pending] == ["old-pending"]


async def test_reparent_moves_every_grant_status(grants_pool: asyncpg.Pool[Any]) -> None:
    connection_id = await _connection(grants_pool)
    async with grants_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO inbound_grants (account_id, connection_id, chat_id, status)
               VALUES ('grant-a',$1,'pending','pending'),
                      ('grant-a',$1,'active','active'),
                      ('grant-a',$1,'revoked','revoked')""",
            connection_id,
        )
        await queries.reparent_connection(conn, connection_id, destination_account_id="grant-b")
        accounts = await conn.fetch(
            "SELECT DISTINCT account_id FROM inbound_grants WHERE connection_id=$1",
            connection_id,
        )
        assert [row["account_id"] for row in accounts] == ["grant-b"]
