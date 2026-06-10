"""Regression suite for aios #752 — age-bound ghost-repair for abandoned
client-side tool calls (the wake-no-progress loop, a regression from #750).

Background
----------
Migration 0066 / #750 added the maintained scalar ``open_tool_call_count`` to
``sessions`` and wired it into the wake predicate
(``sweep.CANDIDATE_ROWS_SQL`` ∧ its read-path twin
``queries._SESSION_ACTIVE_EXPR``): a session with ``open_tool_call_count > 0``
is a permanent wake candidate.

The incident
------------
A session (``metals-factchecker``) accumulated two unresolved ``search_issues``
tool calls (no ``role='tool'`` result, no in-flight task) ~10 days earlier.
``search_issues`` was emitted as a BARE name (no ``mcp__`` prefix) and is not in
the tool registry, so it is a CLIENT-result-pending tool — one the harness never
dispatches because the CLIENT runs it and returns the result.
``sweep._was_dispatched`` returns ``False`` for it (the non-MCP, not-in-registry
branch), so ``find_and_repair_ghosts`` would NEVER resolve it. Those two calls
held ``open_tool_call_count = 2`` forever, so the sweep woke the session every
~32s, the agent emitted a no-op monologue (no tool_calls), and re-woke —
~111 Opus calls/hour for hours (``woken_sessions: 1, repaired_ghosts: 0``).

The fix (#752)
--------------
A new setting ``client_tool_call_max_age_seconds`` (default 24h) extends
ghost-repair: an unresolved, not-in-flight, client-result-pending tool call
whose ASSISTANT turn is older than the bound is treated as ABANDONED — a
synthetic timeout/abandoned error result is appended (the SAME append path the
dispatched-ghost repair uses, so ``open_tool_call_count`` decrements). The call
resolves → the wake predicate goes false → the session quiesces, and the agent
gets an explicit "tool call abandoned" signal instead of silent looping.

CRITICAL discriminator: only CLIENT-result-pending calls (non-MCP,
not-in-registry) are age-errored. CONFIRMATION-pending calls (``always_ask``
tools awaiting a ``tool_confirmed`` event) are EXCLUDED — those wait on the
USER, not a client, and erroring them would kill a slow human-in-the-loop
confirmation.

The invariant this suite pins
-----------------------------
- An ABANDONED client-result-pending call (older than the bound) IS repaired,
  resolving ``open_tool_call_count`` and quiescing the session.
- A RECENT client-result-pending call (within the bound) is NOT repaired.
- A confirmation-pending ``always_ask`` call (any age) is NOT repaired.
- ``CANDIDATE_ROWS_SQL`` / ``_SESSION_ACTIVE_EXPR`` are UNCHANGED — the fix is
  purely on the resolution side; the existing predicate naturally stops waking
  once the call resolves.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import sweep
from aios.harness.sweep import find_and_repair_ghosts, find_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.models.agents import ToolSpec
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

# A client tool call left unresolved for ~10 days — the incident's age. Well
# past the default ``client_tool_call_max_age_seconds`` (24h).
ABANDONED_AGE = dt.timedelta(days=10)
# A client tool call dispatched a few minutes ago — comfortably inside the 24h
# bound, a legitimately in-flight client call.
RECENT_AGE = dt.timedelta(minutes=5)


def _assistant(tool_call_ids: list[str], name: str) -> dict[str, Any]:
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


async def _backdate_assistant_turn(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    account_id: str,
    tool_call_id: str,
    age: dt.timedelta,
) -> None:
    """Rewrite the ``created_at`` of the assistant turn that issued
    ``tool_call_id`` so the call's age is ``age``. ``append_event`` stamps
    ``created_at = now()`` (the table default); the fix keys on the assistant
    turn's ``created_at`` (when the call was emitted), so this is load-bearing.
    """
    old = dt.datetime.now(dt.UTC) - age
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE events SET created_at = $1 "
            "WHERE session_id = $2 AND account_id = $3 "
            "AND kind = 'message' AND role = 'assistant' "
            "AND data->'tool_calls' @> jsonb_build_array("
            "    jsonb_build_object('id', $4::text))",
            old,
            session_id,
            account_id,
            tool_call_id,
        )


async def _seed(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    prefix: str,
    tools: list[ToolSpec],
) -> str:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, NULL, TRUE, $2)",
            account_id,
            prefix,
        )
    _agent, _env, session = await seed_agent_env_session(
        pool,
        account_id=account_id,
        prefix=prefix,
        tools=tools,
    )
    return session.id


@pytest.fixture
async def session_with_abandoned_client_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for the incident shape:

        A1[tc_abandoned (search_issues)]  ← assistant turn backdated ~10 days,
                                            NO result, NO confirmation

    ``search_issues`` is a BARE-name tool the agent does NOT declare as a
    registered/MCP tool (the agent has only ``bash``), so it resolves to the
    CLIENT-result-pending branch of ``_was_dispatched`` — exactly the metals
    incident. ``open_tool_call_count`` is maintained at 1 by ``append_event``,
    so the session is a permanent wake candidate until the call resolves.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_abandoned_client_call"
        sid = await _seed(
            pool,
            account_id=account_id,
            prefix="abandoned-client-call",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=sid,
                kind="message",
                data=_assistant(["tc_abandoned"], name="search_issues"),
            )
        await _backdate_assistant_turn(
            pool,
            session_id=sid,
            account_id=account_id,
            tool_call_id="tc_abandoned",
            age=ABANDONED_AGE,
        )
        yield pool, account_id, sid
    finally:
        await pool.close()


@pytest.fixture
async def session_with_recent_client_call(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a client call still INSIDE
    the bound — a legitimately in-flight client call that must NOT be errored.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_recent_client_call"
        sid = await _seed(
            pool,
            account_id=account_id,
            prefix="recent-client-call",
            tools=[ToolSpec(type="bash")],
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=sid,
                kind="message",
                data=_assistant(["tc_recent"], name="search_issues"),
            )
        await _backdate_assistant_turn(
            pool,
            session_id=sid,
            account_id=account_id,
            tool_call_id="tc_recent",
            age=RECENT_AGE,
        )
        yield pool, account_id, sid
    finally:
        await pool.close()


@pytest.fixture
async def session_with_old_unconfirmed_always_ask(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for the discriminator guard:

        A1[tc_pending (bash, always_ask)]  ← assistant turn backdated ~10 days,
                                             NO ``tool_confirmed``, NO result

    ``bash`` is a REGISTERED ``always_ask`` tool with no confirmation yet, so it
    is CONFIRMATION-pending (waiting on the USER), NOT client-result-pending.
    Even though it is old, it MUST NOT be errored — erroring it would kill a
    slow human-in-the-loop confirmation.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_old_unconfirmed_always_ask"
        sid = await _seed(
            pool,
            account_id=account_id,
            prefix="old-unconfirmed-always-ask",
            tools=[ToolSpec(type="bash", permission="always_ask")],
        )
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=sid,
                kind="message",
                data=_assistant(["tc_pending"], name="bash"),
            )
        await _backdate_assistant_turn(
            pool,
            session_id=sid,
            account_id=account_id,
            tool_call_id="tc_pending",
            age=ABANDONED_AGE,
        )
        yield pool, account_id, sid
    finally:
        await pool.close()


async def _open_tool_call_count(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT open_tool_call_count FROM sessions WHERE id = $1", session_id
        )
        return count


async def _tool_results(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events "
            "WHERE session_id = $1 AND kind = 'message' AND role = 'tool' "
            "AND data->>'tool_call_id' = $2",
            session_id,
            tool_call_id,
        )
    return [queries.parse_jsonb(r["data"]) for r in rows]


class TestAbandonedClientCallRepaired:
    """The keystone: an abandoned client-result-pending call older than the
    bound IS repaired, resolving the wake predicate.

    Pre-fix (master) this test is RED — ``find_and_repair_ghosts`` returns ``[]``
    (``_was_dispatched`` is False, so the call is never repaired) and
    ``find_sessions_needing_inference`` keeps returning the session forever
    (``open_tool_call_count > 0`` with nothing to dispatch = wake-no-progress).
    """

    async def test_abandoned_call_is_repaired_and_session_quiesces(
        self,
        session_with_abandoned_client_call: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, account_id, session_id = session_with_abandoned_client_call
        registry = TaskRegistry()

        # Precondition: the call holds open_tool_call_count > 0, so the session
        # is a wake candidate. Pre-fix it has NO unreacted stimulus and nothing
        # in-flight — the wake produces a no-op monologue and re-arms (the loop).
        assert await _open_tool_call_count(pool, session_id) == 1
        before = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in before, (
            "precondition: an unresolved client call must make the session a "
            "wake candidate (open_tool_call_count > 0) — this is the loop"
        )

        # Ghost repair MUST now resolve the abandoned call (RED on master).
        repaired = await find_and_repair_ghosts(pool, registry, session_id=session_id)
        assert (session_id, "tc_abandoned") in repaired, (
            "abandoned client-result-pending call (~10 days old) must be repaired "
            f"by ghost-repair once the age bound applies; got {repaired}. On "
            "master this is empty — _was_dispatched is False so the call is never "
            "resolved, and the session loops forever (#155 / regression from #750)"
        )

        # A synthetic, clearly-worded error result resolves the call.
        results = await _tool_results(pool, session_id, "tc_abandoned")
        assert len(results) == 1, f"exactly one synthetic result expected; got {results}"
        assert results[0].get("is_error") is True
        assert "abandoned" in (results[0].get("content") or "").lower()

        # open_tool_call_count decremented to 0. The synthetic tool result is now
        # a genuine unreacted stimulus, so the session DOES need inference once —
        # this is real progress (the agent reacts to the abandoned-tool error),
        # NOT the no-op loop. The fix turns "permanent candidate, nothing to do"
        # into "react once, then quiesce".
        assert await _open_tool_call_count(pool, session_id) == 0
        after_repair = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id in after_repair, (
            "after repair the session reacts ONCE to the synthetic error — the "
            "unreacted tool result is real progress, unlike the pre-fix no-op loop"
        )

        # Simulate the agent reacting to the error (an assistant turn with no new
        # tool_calls, ``reacting_to`` the latest stimulus). With open_tool_call_count
        # at 0 and the stimulus reacted-to, the wake predicate is now false: the
        # session quiesces. No infinite loop.
        async with pool.acquire() as conn:
            last_seq = await conn.fetchval(
                "SELECT last_event_seq FROM sessions WHERE id = $1", session_id
            )
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "The search_issues call was abandoned; moving on.",
                    "reacting_to": last_seq,
                },
            )
        quiesced = await find_sessions_needing_inference(pool, registry, session_id=session_id)
        assert session_id not in quiesced, (
            "after the agent reacts the session must quiesce — open_tool_call_count "
            "is 0 and the stimulus is reacted-to, so the wake predicate is false. "
            "The wake-no-progress loop is broken."
        )


class TestRecentClientCallNotRepaired:
    """A client call still INSIDE the bound is a legitimately in-flight client
    call — it must keep waiting, not be errored."""

    async def test_recent_call_not_errored(
        self,
        session_with_recent_client_call: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, _account_id, session_id = session_with_recent_client_call
        registry = TaskRegistry()

        repaired = await find_and_repair_ghosts(pool, registry, session_id=session_id)
        assert repaired == [], (
            "a client-result-pending call only ~5 minutes old is well inside the "
            f"24h bound and must NOT be errored; got {repaired}"
        )
        assert await _tool_results(pool, session_id, "tc_recent") == []
        assert await _open_tool_call_count(pool, session_id) == 1


class TestConfirmationPendingExcluded:
    """The discriminator guard: ``always_ask`` confirmation-pending calls wait
    on the USER, not a client, and are EXCLUDED from age-error at ANY age."""

    async def test_old_unconfirmed_always_ask_not_errored(
        self,
        session_with_old_unconfirmed_always_ask: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, _account_id, session_id = session_with_old_unconfirmed_always_ask
        registry = TaskRegistry()

        repaired = await find_and_repair_ghosts(pool, registry, session_id=session_id)
        assert repaired == [], (
            "an unconfirmed always_ask call is confirmation-pending (waiting on "
            "the USER), NOT client-result-pending — it must NOT be age-errored "
            f"even at ~10 days old; got {repaired}. Erroring it would kill a slow "
            "human-in-the-loop confirmation."
        )
        assert await _tool_results(pool, session_id, "tc_pending") == []
        assert await _open_tool_call_count(pool, session_id) == 1


class TestCandidatePredicateUnchanged:
    """The fix must NOT touch ``CANDIDATE_ROWS_SQL`` — it stays byte-identical to
    its read-path twin ``queries._SESSION_ACTIVE_EXPR`` (modulo table alias).
    The resolution side changes; the predicate does not."""

    def test_candidate_sql_mirrors_session_active_expr(self) -> None:
        import re

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        candidate = norm(sweep.CANDIDATE_ROWS_SQL)
        active = norm(queries._SESSION_ACTIVE_EXPR)

        # Both must key on the same two disjuncts; the fix adds NO new clause.
        # (modulo table alias: the candidate uses ``s.``, the active expr uses
        # ``sessions.``)
        assert "s.last_stimulus_seq > s.last_reacted_seq" in candidate
        assert "s.open_tool_call_count > 0" in candidate
        assert "sessions.last_stimulus_seq > sessions.last_reacted_seq" in active
        assert "sessions.open_tool_call_count > 0" in active
