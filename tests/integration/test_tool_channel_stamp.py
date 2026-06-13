"""Integration tests: the tool-role event's ``channel`` column is stamped
from the parent assistant's ``focal_channel_at_arrival`` — whether the value
is supplied by the live dispatch path (``tool_parent_channel=<stamp>``) or
re-derived by the pre-transaction lookup (cold paths: operator, connector,
ghost-repair, confirmed re-dispatch). Both must agree (issue #862).

The drift case proves the STORED ``focal_channel_at_arrival`` always comes
from the locked RETURNING value even when the token-count pre-read saw a
different focal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import events as events_mod
from aios.db.queries import sessions as session_queries
from aios.db.queries.events import _lookup_tool_parent_channel
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
                VALUES ('acc_chan', NULL, TRUE, 'chan-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_chan", prefix="chan-test"
        )
        yield pool, "acc_chan", session.id
    finally:
        await pool.close()


async def _append_parent_assistant(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, tool_call_id: str
) -> str | None:
    """Append an assistant message carrying a tool_call; return its stamped
    ``focal_channel_at_arrival`` (from the locked RETURNING value)."""
    ev = await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        orig_channel=None,
    )
    return ev.focal_channel_at_arrival


async def _stored_channel(
    conn: asyncpg.Connection[Any], session_id: str, tool_call_id: str
) -> str | None:
    channel: str | None = await conn.fetchval(
        "SELECT channel FROM events WHERE session_id = $1 "
        "AND data->>'tool_call_id' = $2 AND data->>'role' = 'tool'",
        session_id,
        tool_call_id,
    )
    return channel


async def test_live_builtin_passes_stamp(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Hot path: the dispatcher hands the parent stamp via
    ``tool_parent_channel`` — stored tool channel == that stamp, no lookup."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, "tg:42", account_id=account_id
        )
        parent_stamp = await _append_parent_assistant(conn, account_id, session_id, "tc_1")
        assert parent_stamp == "tg:42"

        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_1", "name": "bash", "content": "ok"},
            orig_channel=None,
            tool_parent_channel=parent_stamp,
        )
        assert await _stored_channel(conn, session_id, "tc_1") == "tg:42"


async def test_cold_path_lookup_matches(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Cold path: NO ``tool_parent_channel`` → the pre-tx lookup derives the
    parent stamp. Covers operator/connector/ghost-repair/confirmed-re-dispatch
    — they all share this default-sentinel lookup."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, "tg:99", account_id=account_id
        )
        parent_stamp = await _append_parent_assistant(conn, account_id, session_id, "tc_2")

        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_2", "name": "bash", "content": "ok"},
            orig_channel=None,
        )
        assert await _stored_channel(conn, session_id, "tc_2") == parent_stamp == "tg:99"


async def test_passed_stamp_equals_lookup(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """For the same parent, the value passed on the hot path equals the value
    the cold-path lookup would derive."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, "signal:abc", account_id=account_id
        )
        parent_stamp = await _append_parent_assistant(conn, account_id, session_id, "tc_3")
        looked_up = await _lookup_tool_parent_channel(
            conn, session_id, "tc_3", account_id=account_id
        )
        assert parent_stamp == looked_up == "signal:abc"


async def test_non_connector_none_stamp(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Non-connector parent: focal None. Both hot (pass None) and cold
    (lookup → None) store channel=None for the tool event."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # focal_channel defaults to NULL — no set_session_focal_channel call.
        parent_stamp = await _append_parent_assistant(conn, account_id, session_id, "tc_hot")
        assert parent_stamp is None

        # Hot path: pass None explicitly.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_hot", "name": "bash", "content": "ok"},
            orig_channel=None,
            tool_parent_channel=None,
        )
        assert await _stored_channel(conn, session_id, "tc_hot") is None

        # Cold path: omit → lookup → None.
        await _append_parent_assistant(conn, account_id, session_id, "tc_cold")
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_cold", "name": "bash", "content": "ok"},
            orig_channel=None,
        )
        assert await _stored_channel(conn, session_id, "tc_cold") is None


async def test_user_append_stamp_from_lock_despite_focal_switch(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift case (#862): the token-count pre-read of focal happens BEFORE the
    row lock, so a ``switch_channel`` committing in between can make the
    pre-read stale. The STORED ``focal_channel_at_arrival`` must still be the
    locked RETURNING value, exactly.

    A truly-concurrent interleave isn't deterministically reachable without a
    hook, so we SIMULATE it: monkeypatch the pre-read helper to return channel
    A while the actual session-row focal is set to channel B at lock time.
    Append a USER message; assert the stored stamp == B (the lock value),
    proving the stored stamp is unaffected by the stale token-count pre-read.
    """
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # Real session-row focal = B (what the lock will RETURN).
        await session_queries.set_session_focal_channel(
            conn, session_id, "chan_B", account_id=account_id
        )

    # Pre-read helper lies and returns A (the "stale" pre-lock read).
    async def _fake_focal(_conn: Any, _sid: str, *, account_id: str) -> str | None:
        return "chan_A"

    monkeypatch.setattr(session_queries, "get_session_focal_channel", _fake_focal)

    async with pool.acquire() as conn:
        ev = await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi"},
            orig_channel="chan_B",
        )
    # Stored stamp is the locked RETURNING value B, not the pre-read A.
    assert ev.focal_channel_at_arrival == "chan_B"


async def test_non_message_tool_role_skips_parent_lookup(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 4 guard (#862): the pre-lock tool-channel resolution must gate on
    ``kind == "message"`` before inspecting ``role``. A NON-message event that
    happens to carry ``data["role"] == "tool"`` (e.g. a span) must NOT trigger
    the parent-assistant JSONB lookup — that would be a spurious DB round-trip
    and a behavior change from the old ``_derive_event_channel`` (which
    early-returned for ``kind != "message"`` before ever reading ``role``).
    """
    pool, account_id, session_id = pool_and_session

    spy = AsyncMock(return_value=None)
    monkeypatch.setattr(events_mod, "_lookup_tool_parent_channel", spy)

    async with pool.acquire() as conn:
        ev = await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="span",
            data={"role": "tool", "tool_call_id": "tc_span"},
            orig_channel=None,
        )

    # The lookup is never awaited for a non-message kind, and the derived
    # channel is NULL (non-message events carry no channel).
    spy.assert_not_awaited()
    assert ev.channel is None


async def test_supplied_channel_skips_parent_lookup(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot path (#991 criterion #5): when ``tool_parent_channel`` is supplied
    (the live builtin/MCP dispatch path, or the ``append_tool_result`` path that
    feeds the channel from its single name-lookup scan), the precompute must NOT
    invoke ``_lookup_tool_parent_channel`` — at most ONE parent ``@>`` scan."""
    pool, account_id, session_id = pool_and_session

    spy = AsyncMock(return_value="should-not-be-used")
    monkeypatch.setattr(events_mod, "_lookup_tool_parent_channel", spy)

    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, "tg:supplied", account_id=account_id
        )
        await _append_parent_assistant(conn, account_id, session_id, "tc_supplied")
        ev = await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={
                "role": "tool",
                "tool_call_id": "tc_supplied",
                "name": "bash",
                "content": "ok",
            },
            orig_channel=None,
            tool_parent_channel="tg:supplied",
        )

    spy.assert_not_awaited()
    assert ev.channel == "tg:supplied"
