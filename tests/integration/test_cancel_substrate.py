"""Integration tests for the cancel-supervision side-tables (cancel-design §0/§9).

DB-backed round-trips over the durable primitives the recursive ``cancel_invocation``
cascade is built on: the ``cancel_intents`` tombstone (+ its §9 monotone quiescence
counter) and the session-side ``session_cancel_markers`` exit-marker. The cascade logic
that drives these (propagation, the leaves, the tool) lands in 6d-6g; here we pin the
substrate's idempotency + counter semantics in isolation.
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


async def test_cancel_intent_idempotent_with_monotone_counter(
    pool_and_session: tuple[asyncpg.Pool[Any], str],
) -> None:
    """The tombstone is idempotent (re-cancel never resets the counter) and ``quiesced_at``
    latches monotonically the first time ``outstanding`` reaches 0."""
    pool, _session = pool_and_session
    key = dict(servicer_kind="run", servicer_id="wfr_x", request_id="req_1")
    async with pool.acquire() as conn:
        intent = await queries.insert_cancel_intent(conn, account_id=_ACCOUNT, **key)
        assert intent.outstanding == 1 and intent.quiesced_at is None

        # A node marks 2 children → += (2 - 1). Then a RE-cancel must be a no-op:
        # the counter keeps its accumulated value, never resets to the seed.
        await queries.adjust_cancel_outstanding(conn, delta=+1, **key)
        again = await queries.insert_cancel_intent(conn, account_id=_ACCOUNT, **key)
        assert again.outstanding == 2  # not reset to 1

        # Drive to 0 → quiesced latches.
        await queries.adjust_cancel_outstanding(conn, delta=-1, **key)
        zeroed = await queries.adjust_cancel_outstanding(conn, delta=-1, **key)
        assert zeroed is not None and zeroed.outstanding == 0 and zeroed.quiesced_at is not None
        first_quiesce = zeroed.quiesced_at

        # A late delta does NOT un-quiesce (monotone) — the cascade is already complete.
        bumped = await queries.adjust_cancel_outstanding(conn, delta=+1, **key)
        assert bumped is not None and bumped.outstanding == 1
        assert bumped.quiesced_at == first_quiesce

        # A delta against a missing tombstone is a no-op returning None.
        gone = await queries.adjust_cancel_outstanding(
            conn, servicer_kind="run", servicer_id="wfr_nope", request_id="r", delta=-1
        )
        assert gone is None


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
        # Second seed of the same edge is a no-op (the §9 counter must not double-count).
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
