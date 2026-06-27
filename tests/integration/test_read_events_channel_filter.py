"""Integration tests: the channel/chat_type filter on ``read_events`` (#1613).

These exercise the authoritative channel slice the relay/cockpit need — and
directly reproduce the dropped-DM bug (an inbound on channel B previously
mis-attributed to a send started on channel A) and prove it excluded.

Drives:
  * ``inbound(A) + inbound(B) interleaved → send`` — assert ?channel=A returns
    the A turn and EXCLUDES the B inbound.
  * ``inbound(A) → switch_channel(B) → send`` — the send's tool result inherits
    the parent assistant's focal A, so ?channel=A still returns the send.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import sessions as session_queries
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

CHAN_A = "signal/bot/6c21718f-f095-483f-8cd6-610137d581aa"  # a DM (UUID)
CHAN_B = "signal/bot/abcDEF123_-=="  # a group (base64)


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
                VALUES ('acc_filt', NULL, TRUE, 'filter-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_filt", prefix="filter-test"
        )
        yield pool, "acc_filt", session.id
    finally:
        await pool.close()


async def _inbound(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, channel: str, text: str
) -> None:
    await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={"role": "user", "content": text, "metadata": {"channel": channel}},
        orig_channel=channel,
    )


async def _assistant_with_tool(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, tool_call_id: str
) -> str | None:
    ev = await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={
            "role": "assistant",
            "content": "sending",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "signal_send", "arguments": "{}"},
                }
            ],
        },
        orig_channel=None,
    )
    return ev.focal_channel_at_arrival


async def _tool_result(
    conn: asyncpg.Connection[Any], account_id: str, session_id: str, tool_call_id: str
) -> None:
    await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="message",
        data={
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "signal_send",
            "content": "sent",
        },
        orig_channel=None,
    )


async def test_channel_filter_excludes_cross_channel_inbound(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Dropped-DM reproduction (interleaved drive): inbound(A) + inbound(B) →
    send in A. ?channel=A returns the A inbound + the send (assistant + result)
    and EXCLUDES the B inbound that previously caused the misattribution."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # Inbound on A, then a cross-channel inbound on B interleaves.
        await session_queries.set_session_focal_channel(
            conn, session_id, CHAN_A, account_id=account_id
        )
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        await _inbound(conn, account_id, session_id, CHAN_B, "from B")
        # Assistant turn is focal A → send to A.
        await _assistant_with_tool(conn, account_id, session_id, "tc_send_A")
        await _tool_result(conn, account_id, session_id, "tc_send_A")

    async with pool.acquire() as conn:
        a_events = await queries.read_events(
            conn, session_id, channels=[CHAN_A], account_id=account_id
        )
        b_events = await queries.read_events(
            conn, session_id, channels=[CHAN_B], account_id=account_id
        )

    a_channels = {e.channel for e in a_events}
    assert a_channels == {CHAN_A}, a_channels
    # The A slice contains the inbound, the assistant send, and its tool result.
    a_roles = sorted(e.data.get("role") for e in a_events)
    assert a_roles == ["assistant", "tool", "user"], a_roles
    # The cross-channel B inbound is EXCLUDED from the A slice (the bug).
    assert all(e.data.get("content") != "from B" for e in a_events)

    # The B slice is exactly the one B inbound.
    assert {e.channel for e in b_events} == {CHAN_B}
    assert [e.data.get("content") for e in b_events] == ["from B"]


async def test_channel_filter_send_after_switch_attributes_to_origin(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Switch drive: inbound(A) → switch focal to B → send. The send's tool
    result inherits the PARENT assistant's focal A, so ?channel=A returns the
    send and ?channel=B excludes it."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await session_queries.set_session_focal_channel(
            conn, session_id, CHAN_A, account_id=account_id
        )
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        # Assistant turn arrives while focal is A.
        stamp = await _assistant_with_tool(conn, account_id, session_id, "tc_after_switch")
        assert stamp == CHAN_A
        # Focal switches to B AFTER the assistant turn, before the result lands.
        await session_queries.set_session_focal_channel(
            conn, session_id, CHAN_B, account_id=account_id
        )
        await _tool_result(conn, account_id, session_id, "tc_after_switch")

    async with pool.acquire() as conn:
        a_events = await queries.read_events(
            conn, session_id, channels=[CHAN_A], account_id=account_id
        )
        b_events = await queries.read_events(
            conn, session_id, channels=[CHAN_B], account_id=account_id
        )

    # The tool result attributes to A (parent's focal), not B.
    a_roles = sorted(e.data.get("role") for e in a_events)
    assert a_roles == ["assistant", "tool", "user"], a_roles
    assert all(e.channel == CHAN_A for e in a_events)
    # No event leaked into the B slice.
    assert b_events == []


async def test_channel_filter_or_semantics(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Multiple channels ⇒ OR semantics; both A and B inbound returned."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        await _inbound(conn, account_id, session_id, CHAN_B, "from B")

    async with pool.acquire() as conn:
        both = await queries.read_events(
            conn, session_id, channels=[CHAN_A, CHAN_B], account_id=account_id
        )
    assert {e.channel for e in both} == {CHAN_A, CHAN_B}


async def test_no_filter_is_unchanged(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Regression: omitting ``channels`` returns ALL rows incl. NULL-channel
    switch_channel/lifecycle rows — byte-identical to before #1613."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _inbound(conn, account_id, session_id, CHAN_A, "from A")
        # A switch_channel event carries no channel (NULL).
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="span",
            data={"name": "switch_channel"},
            orig_channel=None,
        )
        await _inbound(conn, account_id, session_id, CHAN_B, "from B")

    async with pool.acquire() as conn:
        unfiltered = await queries.read_events(conn, session_id, account_id=account_id)
        a_only = await queries.read_events(
            conn, session_id, channels=[CHAN_A], account_id=account_id
        )

    # Unfiltered includes the NULL-channel span; the channel filter excludes it.
    assert any(e.channel is None for e in unfiltered)
    assert len(unfiltered) == 3
    assert all(e.channel == CHAN_A for e in a_only)
    assert len(a_only) == 1


async def test_chat_type_filter(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``chat_type=dm`` returns only DM-channel rows (CHAN_A is a UUID/DM);
    ``chat_type=group`` returns only the base64/group rows (CHAN_B)."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await _inbound(conn, account_id, session_id, CHAN_A, "dm msg")
        await _inbound(conn, account_id, session_id, CHAN_B, "group msg")

    async with pool.acquire() as conn:
        dm = await queries.read_events(conn, session_id, chat_type="dm", account_id=account_id)
        group = await queries.read_events(
            conn, session_id, chat_type="group", account_id=account_id
        )

    assert [e.channel for e in dm] == [CHAN_A]
    assert [e.channel for e in group] == [CHAN_B]


async def test_chat_type_filter_paginates_to_limit(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The chat_type post-filter preserves LIMIT semantics: interleave many
    group rows around a few DM rows and assert ?chat_type=dm collects up to
    ``limit`` DM rows even though they are sparse in the log."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # 1 DM, then 5 group, then 1 DM, ... so DM rows are sparse.
        for i in range(3):
            await _inbound(conn, account_id, session_id, CHAN_A, f"dm {i}")
            for j in range(5):
                await _inbound(conn, account_id, session_id, CHAN_B, f"grp {i}-{j}")

    async with pool.acquire() as conn:
        dm = await queries.read_events(
            conn, session_id, chat_type="dm", limit=2, account_id=account_id
        )
    assert len(dm) == 2
    assert all(e.channel == CHAN_A for e in dm)


async def test_channel_filter_uses_partial_index(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The channel filter must hit ``events_session_channel_seq_idx`` — not a
    full seq-scan of the log (acceptance: no scan of the whole log)."""
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        for i in range(50):
            await _inbound(conn, account_id, session_id, CHAN_A, f"a{i}")
            await _inbound(conn, account_id, session_id, CHAN_B, f"b{i}")
        plan = await conn.fetch(
            "EXPLAIN SELECT * FROM events "
            "WHERE session_id = $1 AND account_id = $2 AND channel = ANY($3) "
            "ORDER BY seq ASC LIMIT 200",
            session_id,
            account_id,
            [CHAN_A],
        )
    plan_text = "\n".join(r["QUERY PLAN"] for r in plan)
    assert "events_session_channel_seq_idx" in plan_text, plan_text
