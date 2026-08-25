"""Integration tests for #2254: the awaiting read-model's history scan is
bounded by ``sessions.open_tool_call_floor_seq`` (migration 0136), and the
former two-scan + in-Python diff is a single SQL anti-join.

Discrimination proof, both directions (a guard only ever seen to permit is
indistinguishable from one that permits everything):

* ``test_bound_is_applied`` FAILS IF THE BOUND IS REMOVED — with the floor
  raised above an open call's seq, that call must vanish from the batch.  (A
  floor the sweep maintains never sits above a genuinely open call — invariant
  of #1746 — so hiding it here is safe to assert on; the test forges the floor
  precisely to make the bound observable.)
* ``test_open_call_at_floor_surfaces`` FAILS IF THE BOUND IS TOO AGGRESSIVE —
  the bound is ``>=``: an open call whose assistant event sits EXACTLY at the
  floor must still surface.

Plus behaviour-unchanged (floor 0 = unbounded, the migration default) and
anti-join parity (resolved calls stay hidden, intra-turn order preserved).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.models.sessions import Session
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


def _assistant(*tool_call_ids: str, name: str = "bash") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tcid,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for tcid in tool_call_ids
        ],
    }


async def _set_floor(pool: asyncpg.Pool[Any], session_id: str, floor_seq: int) -> None:
    """Forge ``open_tool_call_floor_seq`` directly.

    Production writes go through the sweep's single GREATEST-only statement
    (guarded by ``test_open_tool_call_floor_seq_single_writer``); tests forge
    the column to place the floor exactly where each discrimination case
    needs it.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET open_tool_call_floor_seq = $1 WHERE id = $2",
            floor_seq,
            session_id,
        )


@pytest.fixture
async def two_open_turns(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, Session, int, int]]:
    """Yield ``(pool, account_id, session, a1_seq, a2_seq)`` for a log of
    A1(tc_old) → user → A2(tc_new), with NO tool_result for either."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_awaiting_floor_bound"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "awaiting-floor-bound-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="awaiting-floor-bound",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            a1 = await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data=_assistant("tc_old"),
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data={"role": "user", "content": "are you still there?"},
            )
            a2 = await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data=_assistant("tc_new"),
            )
        yield pool, account_id, session, a1.seq, a2.seq
    finally:
        await pool.close()


async def _unresolved_ids(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> list[str]:
    async with pool.acquire() as conn:
        batch = await queries.list_unresolved_tool_calls_batch(
            conn, [session_id], account_id=account_id
        )
    return [e["tool_call_id"] for e in batch.get(session_id, [])]


class TestAwaitingFloorSeqBound:
    async def test_default_floor_is_unbounded(
        self, two_open_turns: tuple[asyncpg.Pool[Any], str, Session, int, int]
    ) -> None:
        """Migration default (floor 0): behaviour unchanged — every open call
        from the whole history surfaces, #741 span semantics intact."""
        pool, account_id, session, _a1_seq, _a2_seq = two_open_turns
        assert await _unresolved_ids(pool, session.id, account_id=account_id) == [
            "tc_old",
            "tc_new",
        ]

    async def test_bound_is_applied(
        self, two_open_turns: tuple[asyncpg.Pool[Any], str, Session, int, int]
    ) -> None:
        """FAILS IF THE BOUND IS REMOVED: floor above A1 hides tc_old."""
        pool, account_id, session, _a1_seq, a2_seq = two_open_turns
        await _set_floor(pool, session.id, a2_seq)
        assert await _unresolved_ids(pool, session.id, account_id=account_id) == ["tc_new"], (
            "open_tool_call_floor_seq did not bound the awaiting scan — a "
            "forged floor above tc_old's assistant seq must exclude it (#2254)"
        )

    async def test_open_call_at_floor_surfaces(
        self, two_open_turns: tuple[asyncpg.Pool[Any], str, Session, int, int]
    ) -> None:
        """FAILS IF THE BOUND IS TOO AGGRESSIVE: the bound is ``>=`` — an open
        call sitting EXACTLY at the floor (the sweep's normal steady state:
        the floor IS the oldest open call's seq) must still surface."""
        pool, account_id, session, a1_seq, _a2_seq = two_open_turns
        await _set_floor(pool, session.id, a1_seq)
        assert await _unresolved_ids(pool, session.id, account_id=account_id) == [
            "tc_old",
            "tc_new",
        ], (
            "an open call at seq == open_tool_call_floor_seq was hidden — the "
            "bound must be inclusive (>=), or the sweep's own floor placement "
            "(floor = oldest open call's seq, #1746) would hide real work"
        )

    async def test_anti_join_parity_and_order(
        self, two_open_turns: tuple[asyncpg.Pool[Any], str, Session, int, int]
    ) -> None:
        """The SQL anti-join matches the old two-scan + set-diff semantics:
        a resolved call disappears, unresolved intra-turn order is preserved."""
        pool, account_id, session, _a1_seq, _a2_seq = two_open_turns
        async with pool.acquire() as conn:
            # A third assistant turn with TWO calls; resolve only the first.
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data=_assistant("tc_a", "tc_b"),
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data={"role": "tool", "tool_call_id": "tc_a", "content": "done"},
            )
        assert await _unresolved_ids(pool, session.id, account_id=account_id) == [
            "tc_old",
            "tc_new",
            "tc_b",
        ]
