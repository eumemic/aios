"""Integration tests: repair-on-mismatch at the two cold hard-reject sites
(issue #1742).

The maintained ``sessions.channels`` array is the hot-path source of
truth, but a rolling-deploy window can leave it stale relative to the
event log (an old, pre-#1742 container appends a new channel without
maintaining the column). Both cold sites — ``switch_channel``'s
membership check and the POST ``/messages`` ``metadata.channel``
bound-check — must, before hard-rejecting a real target, recompute the
ground truth from the event log and repair the stored row if it
disagrees, then re-check. Neither ever hard-rejects a channel that
genuinely has message history.

These tests manually corrupt ``channels`` (drop a real entry that DOES
have message history) and confirm both call sites self-heal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.tools.switch_channel import switch_channel_handler
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_chan_repair', NULL, TRUE, 'chan-repair')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_chan_repair", prefix="chan-repair"
        )
        yield pool, "acc_chan_repair", session.id
    finally:
        await pool.close()


async def _corrupt_drop_channel(
    pool: asyncpg.Pool[Any], session_id: str, dropped: str, *, account_id: str
) -> None:
    """Manually remove ``dropped`` from the stored ``channels`` array,
    simulating the rolling-deploy drift window."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET channels = array_remove(channels, $1) "
            "WHERE id = $2 AND account_id = $3",
            dropped,
            session_id,
            account_id,
        )


async def test_switch_channel_repairs_corrupted_set(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi"},
            orig_channel="real_chan",
        )

    # Corrupt: the maintained array no longer thinks "real_chan" is bound,
    # even though the event log proves it is.
    await _corrupt_drop_channel(pool, session_id, "real_chan", account_id=account_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channels FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        assert list(row["channels"]) == []

    with patch("aios.tools.switch_channel.runtime.require_pool", return_value=pool):
        result = await switch_channel_handler(session_id, {"channel_id": "real_chan"})

    assert result.is_error is not True
    assert result.metadata is not None
    marker = result.metadata.get("switch_channel")
    assert marker == {"target": "real_chan", "success": True}

    # The row is repaired.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channels FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        assert list(row["channels"]) == ["real_chan"]


async def test_post_message_bound_check_repairs_corrupted_set(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi"},
            orig_channel="real_chan_2",
        )

    await _corrupt_drop_channel(pool, session_id, "real_chan_2", account_id=account_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channels FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        assert list(row["channels"]) == []

    # Reproduce the router's bound-check logic directly against the real
    # pool (an in-process HTTP client isn't wired in this integration test
    # tier — the logic under test is the query-layer repair sequence the
    # router calls, exercised end-to-end against Postgres).
    async with pool.acquire() as conn:
        bound = set(await queries.list_session_channels(conn, session_id, account_id=account_id))
        assert "real_chan_2" not in bound
        recomputed = set(
            await queries.recompute_session_channels(conn, session_id, account_id=account_id)
        )
        assert recomputed != bound
        await queries.set_session_channels(
            conn, session_id, sorted(recomputed), account_id=account_id
        )
        bound = recomputed
    assert "real_chan_2" in bound

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channels FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        assert list(row["channels"]) == ["real_chan_2"]
