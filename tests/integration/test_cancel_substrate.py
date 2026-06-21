"""Integration tests for the cancel-supervision side-table (cancel-design §2).

DB-backed round-trips over the durable primitive the recursive ``cancel_invocation``
cascade is built on: the session-side ``session_cancel_markers`` exit-marker. The cascade
logic that drives it (propagation, the leaf, the seed) is covered by test_session_cancel_leaf;
here we pin the marker's idempotency + harvest semantics in isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_cancel_substrate"


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'cancel-substrate')",
                _ACCOUNT,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="cancel-sub"
        )
        yield pool, session.id
    finally:
        await pool.close()


async def test_session_cancel_marker_idempotent_and_harvest(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """The session exit-marker is ON-CONFLICT-idempotent (re-propagation is a no-op) and
    ``harvested_at`` removes it from the unharvested (sweep-visible) set."""
    pool, session_id = pool_and_session
    async with pool.acquire() as conn:
        assert (
            await queries.insert_session_cancel_marker(
                conn, session_id=session_id, request_id="req_1", account_id=_ACCOUNT
            )
            is True
        )
        # Second seed of the same edge is a no-op (re-propagation must not double-mark).
        assert (
            await queries.insert_session_cancel_marker(
                conn, session_id=session_id, request_id="req_1", account_id=_ACCOUNT
            )
            is False
        )

        marker = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_1"
        )
        assert marker is not None and marker.harvested_at is None
        unharvested = await queries.list_unharvested_session_cancel_markers(conn, session_id)
        assert [m.request_id for m in unharvested] == ["req_1"]

        await queries.mark_session_cancel_marker_harvested(
            conn, session_id=session_id, request_id="req_1"
        )
        harvested = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_1"
        )
        assert harvested is not None and harvested.harvested_at is not None
        assert await queries.list_unharvested_session_cancel_markers(conn, session_id) == []
