"""Integration test: the #1742 sessions.channels invariant.

For any session, at all times ``set(sessions.channels) ==
{e.channel : e.kind='message' AND e.channel IS NOT NULL}``, and
``list_session_channels`` returns exactly that set, sorted.

This appends a scripted mix of user / assistant(after a focal switch) /
tool (both the live ``tool_parent_channel``-supplied path and the cold
``_lookup_tool_parent_channel`` path) / non-message / NULL-channel events
across >=2 channels, and after EACH append asserts:

* ``list_session_channels`` (the maintained ``sessions.channels`` array,
  sorted) == ``recompute_session_channels`` (the DISTINCT-scan ground
  truth) — the correctness invariant the whole issue exists to hold;
* duplicates are never re-appended (the array never grows for a channel
  already present);
* the result is sorted.
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
                VALUES ('acc_chan_inv', NULL, TRUE, 'chan-invariant')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_chan_inv", prefix="chan-invariant"
        )
        yield pool, "acc_chan_inv", session.id
    finally:
        await pool.close()


async def _assert_invariant(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    listed = await queries.list_session_channels(conn, session_id, account_id=account_id)
    recomputed = await queries.recompute_session_channels(conn, session_id, account_id=account_id)
    assert listed == recomputed, f"{listed!r} != {recomputed!r}"
    assert listed == sorted(listed), f"{listed!r} is not sorted"
    return listed


async def test_scripted_append_mix_holds_invariant(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = pool_and_session

    async with pool.acquire() as conn:
        # 1. User message on channel A.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi from A"},
            orig_channel="chan_A",
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A"]

        # 2. Focal switch to A so the next assistant message stamps from it,
        # then an assistant reply (channel derived from focal_channel).
        await session_queries.set_session_focal_channel(
            conn, session_id, "chan_A", account_id=account_id
        )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": "hello back", "tool_calls": []},
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A"]  # duplicate — must not re-append

        # 3. User message on a second channel B.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi from B"},
            orig_channel="chan_B",
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B"]

        # 4. Switch focal to B, assistant message requesting a tool call.
        await session_queries.set_session_focal_channel(
            conn, session_id, "chan_B", account_id=account_id
        )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={
                "role": "assistant",
                "content": "running tool",
                "tool_calls": [
                    {
                        "id": "tc_live",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B"]

        # 5. Tool result via the LIVE dispatch path (tool_parent_channel
        # supplied directly) — channel B, already present.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_live", "name": "bash", "content": "ok"},
            orig_channel=None,
            tool_parent_channel="chan_B",
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B"]

        # 6. Switch focal to a third channel C, assistant requests another
        # tool call, and the tool result arrives via the COLD lookup path
        # (no tool_parent_channel supplied — the default ``...`` sentinel
        # triggers ``_lookup_tool_parent_channel``).
        await session_queries.set_session_focal_channel(
            conn, session_id, "chan_C", account_id=account_id
        )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={
                "role": "assistant",
                "content": "running another tool",
                "tool_calls": [
                    {
                        "id": "tc_cold",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B", "chan_C"]

        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_cold", "name": "bash", "content": "ok"},
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B", "chan_C"]

        # 7. Non-message span event — must not perturb the set.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="span",
            data={"event": "model_request_end"},
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B", "chan_C"]

        # 8. A message event with NO identifiable channel (no orig_channel,
        # no focal at arrival) — clear focal first, must not perturb the set.
        await session_queries.set_session_focal_channel(
            conn, session_id, None, account_id=account_id
        )
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": "no channel", "tool_calls": []},
            orig_channel=None,
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B", "chan_C"]

        # 9. A repeat user message on chan_A — must not re-append / duplicate.
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi again from A"},
            orig_channel="chan_A",
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == ["chan_A", "chan_B", "chan_C"]

        # 10. A fourth channel D, inserted "out of sorted order" relative to
        # insertion (D > C alphabetically so this also exercises the sort
        # step trivially; add a lexicographically-earlier channel to prove
        # ``list_session_channels`` sorts rather than returns insertion order).
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi from AA"},
            orig_channel="chan_AA",
        )
        listed = await _assert_invariant(conn, session_id, account_id=account_id)
        assert listed == sorted(["chan_A", "chan_B", "chan_C", "chan_AA"])
