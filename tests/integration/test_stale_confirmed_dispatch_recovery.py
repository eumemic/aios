"""Regression suite for aios #746 — age-guard the confirmed-tool dispatch path.

Background
----------
Commit 775554e ("fix(harness): confirmed-tool dispatch hardening — window-edge
#737 / awaiting #741 / sweep-predicate unification #740") replaced the OLD
*windowed* confirmed-tool dispatch with a NEW *unwindowed* recovery query
``queries.list_confirmed_unresolved_tool_calls`` (``src/aios/db/queries/__init__.py``),
called by ``_dispatch_confirmed_tools`` (``src/aios/harness/loop.py``).  Its
detection twin is ``sweep.CONFIRMED_ROWS_SQL`` (the case-(c) wake predicate).

Both predicates matched a ``tool_confirmed``/``allow`` lifecycle event whose
``tool_call_id`` has no paired ``role='tool'`` result — *unwindowed, with no age
bound* (``created_at`` was never consulted).  So an operator-confirmed
``always_ask`` tool call that never dispatched was re-dispatched on the next
worker-restart / resume regardless of age (re-running a weeks-stale,
side-effecting tool).

Incident (#744, the connector sibling): on a worker restart, this recovered 17
``signal_send`` calls confirmed-but-undispatched since ~2.5 weeks earlier and
dispatched them with stale content (synchronous connector RPC at dispatch time),
so the messages went out fresh-stamped but stale-content.

The fix (#746)
--------------
A new setting ``confirmed_dispatch_max_age_seconds`` (default 1h, parallel to
``connector_backfill_max_age_seconds`` from #744) bounds BOTH predicates on the
CONFIRM event's ``created_at`` — the ``tool_confirmed`` lifecycle row, NOT the
assistant turn.  An operator can confirm an OLD proposal, which is a FRESH
intent to dispatch, so the dispatchable-since timestamp is when confirmation was
granted.  Skip-not-expire: stale calls are simply excluded; no synthetic
results, no log mutation.

CRITICAL: detection (sweep) and dispatch (resolver) apply the IDENTICAL age
clause, so they resolve the same condition — a mismatch surfaces a session for
wake that dispatch can't resolve (or vice-versa), the wake-with-no-progress loop
(#155 symptom).  ``TestConfirmDispatchSweepSync`` pins that invariant directly.

The invariant this suite pins
-----------------------------
A confirmed-unresolved tool call whose CONFIRMATION is far older than any sane
staleness window MUST NOT be auto-dispatched (or surfaced for wake) on recovery
— but a FRESH confirmation MUST be, even when the underlying assistant proposal
is old (the bound is on confirm-time, not assistant-time).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import sweep
from aios.harness.loop import _dispatch_confirmed_tools
from aios.harness.task_registry import TaskRegistry
from aios.models.agents import ToolSpec
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

# A confirmation that has sat unresolved for ~3 weeks. The incident's stale
# ``signal_send`` calls were ~2.5 weeks old; 21 days is comfortably past the
# default ``confirmed_dispatch_max_age_seconds`` (1h) staleness window.
STALE_AGE = dt.timedelta(days=21)


def _assistant(tool_call_ids: list[str], name: str = "bash") -> dict[str, Any]:
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


def _allow(tool_call_id: str) -> dict[str, Any]:
    return {"event": "tool_confirmed", "result": "allow", "tool_call_id": tool_call_id}


@pytest.fixture
async def session_with_stale_connector_send(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session whose log is:

        A1[tc_stale (signal_send)]  ← confirmed (allow) ~3 weeks ago, NO result
        user("are you still there?")
        A2[tc_fresh (signal_send)]  ← confirmed (allow), result present
        allow(tc_stale)             ← backdated ~3 weeks
        allow(tc_fresh)
        tool_result(tc_fresh)

    A1 is deliberately NOT the latest assistant (A2 is), so ``tc_stale`` is the
    #737 window-edge / non-latest-assistant case the unwindowed resolver was
    built to recover.  ``tc_stale`` is a CONNECTOR send (``signal_send``) and is
    aged so its confirmation predates the staleness window.  ``tc_fresh`` is the
    control: confirmed AND resolved, so it must never re-dispatch (invariant #4)
    — it is here to prove the fixture's recovery surface is real.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_stale_confirmed_dispatch"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "stale-confirmed-dispatch-test",
            )
        # ``signal_send`` is a connector tool; the resolver matches whatever
        # tool_call dict sits on the assistant (it does not filter by tool
        # name), so a real ToolSpec is unnecessary — mirror the existing test
        # and seed a plain bash ToolSpec for the agent scaffold.
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="stale-confirmed-dispatch",
            tools=[ToolSpec(type="bash")],
        )
        sid = session.id

        async def append(kind: str, data: dict[str, Any]) -> None:
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn, account_id=account_id, session_id=sid, kind=kind, data=data
                )

        # A1 carries the stale connector send (confirmed, never resolved); A2 is
        # the LATEST assistant and carries a resolved connector send — so the
        # stale tool's parent is deliberately not the latest assistant.
        await append("message", _assistant(["tc_stale"], name="signal_send"))
        await append("message", {"role": "user", "content": "are you still there?"})
        await append("message", _assistant(["tc_fresh"], name="signal_send"))
        await append("lifecycle", _allow("tc_stale"))
        await append("lifecycle", _allow("tc_fresh"))
        await append(
            "message",
            {
                "role": "tool",
                "tool_call_id": "tc_fresh",
                "content": "sent",
                "name": "signal_send",
            },
        )

        # Backdate the stale call's age. ``append_event`` stamps
        # ``created_at = now()`` (the table default); rewrite it on the rows
        # that carry the call's age — the A1 assistant that issued the
        # ``signal_send`` AND its ``tool_confirmed``/``allow`` lifecycle event —
        # so the confirmation is ~3 weeks old, well past the window.  The fix
        # (#746) keys on the CONFIRM event's ``created_at``, so backdating the
        # lifecycle row is load-bearing here; the assistant backdate is kept too
        # so the fixture also reflects a genuinely old proposal.
        old = dt.datetime.now(dt.UTC) - STALE_AGE
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE events SET created_at = $1 "
                "WHERE session_id = $2 AND account_id = $3 "
                "AND ("
                "  (kind = 'message' AND role = 'assistant' "
                "   AND data->'tool_calls' @> jsonb_build_array("
                "       jsonb_build_object('id', 'tc_stale'::text)))"
                "  OR (kind = 'lifecycle' AND data->>'event' = 'tool_confirmed' "
                "      AND data->>'tool_call_id' = 'tc_stale')"
                ")",
                old,
                sid,
                account_id,
            )

        yield pool, account_id, sid
    finally:
        await pool.close()


@pytest.fixture
async def session_with_fresh_confirm_on_old_proposal(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for the over-reach guard:

        A1[tc_revived (signal_send)]  ← assistant turn backdated ~3 weeks
        allow(tc_revived)             ← confirmation created NOW, NO result

    An operator left an ``always_ask`` proposal sitting for ~3 weeks, then
    confirmed it NOW.  The bound is on the CONFIRM event's ``created_at``, not
    the assistant turn, so this MUST still dispatch — the confirmation is a
    fresh intent.  Backdating ONLY the assistant turn (not the lifecycle row)
    is the whole point of the fixture.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_fresh_confirm_old_proposal"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "fresh-confirm-old-proposal-test",
            )
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="fresh-confirm-old-proposal",
            tools=[ToolSpec(type="bash")],
        )
        sid = session.id

        async def append(kind: str, data: dict[str, Any]) -> None:
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn, account_id=account_id, session_id=sid, kind=kind, data=data
                )

        await append("message", _assistant(["tc_revived"], name="signal_send"))
        await append("lifecycle", _allow("tc_revived"))

        # Backdate ONLY the assistant turn — NOT the ``tool_confirmed`` event.
        # The confirmation stays stamped at ``now()`` (fresh), so the bound (on
        # the confirm event) must NOT exclude it even though the proposal is old.
        old = dt.datetime.now(dt.UTC) - STALE_AGE
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE events SET created_at = $1 "
                "WHERE session_id = $2 AND account_id = $3 "
                "AND kind = 'message' AND role = 'assistant' "
                "AND data->'tool_calls' @> jsonb_build_array("
                "    jsonb_build_object('id', 'tc_revived'::text))",
                old,
                sid,
                account_id,
            )

        yield pool, account_id, sid
    finally:
        await pool.close()


class TestStaleConfirmedDispatchExcluded:
    async def test_stale_confirm_excluded_by_resolver(
        self,
        session_with_stale_connector_send: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The raw resolver, when passed the age bound, drops the ~3-week-old
        confirmed ``signal_send`` (regression for #746 at the query layer).

        Passing ``max_age_seconds`` explicitly here exercises the new clause;
        the production dispatch path (the test below) reads the setting itself.
        """
        pool, account_id, session_id = session_with_stale_connector_send
        async with pool.acquire() as conn:
            dispatchable = await queries.list_confirmed_unresolved_tool_calls(
                conn,
                session_id,
                account_id=account_id,
                max_age_seconds=get_settings().confirmed_dispatch_max_age_seconds,
            )
        ids = [tc["id"] for tc in dispatchable]
        assert ids == [], (
            "stale confirmed signal_send (confirmed ~3 weeks ago) must be "
            f"excluded once the age bound is applied; got {ids}"
        )

        # Unbounded (the default), the resolver still recovers it — the bound is
        # opt-in and the only difference, so the old recovery surface is intact.
        async with pool.acquire() as conn:
            unbounded = await queries.list_confirmed_unresolved_tool_calls(
                conn, session_id, account_id=account_id
            )
        assert [tc["id"] for tc in unbounded] == ["tc_stale"], (
            "without the age bound the resolver must still recover the stale "
            "call (the bound is the only behavioral change)"
        )

    async def test_stale_connector_send_excluded(
        self,
        session_with_stale_connector_send: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """INVARIANT (#746 / #744): a confirmed-unresolved CONNECTOR send whose
        confirmation is far older than the staleness window MUST NOT be
        auto-dispatched on recovery.  This is the brought-in repro's contract —
        it was RED on master and is GREEN with the fix.
        """
        pool, account_id, session_id = session_with_stale_connector_send

        pending = await _dispatch_confirmed_tools(
            pool,
            session_id,
            account_id=account_id,
            task_registry=TaskRegistry(),
        )
        pending_ids = [tc["id"] for tc in pending]

        assert "tc_stale" not in pending_ids, (
            "STALE CONNECTOR SEND RECOVERED (bug #744/#746): a signal_send "
            f"confirmed ~{STALE_AGE.days} days ago was returned for auto-dispatch "
            "and would be transmitted with stale content on this worker restart. "
            "The age guard on the confirm event must exclude it. "
            f"dispatchable={pending_ids}"
        )


class TestFreshConfirmOnOldProposalStillDispatches:
    """Over-reach guard: the bound is on confirm-time, NOT assistant-time."""

    async def test_fresh_confirm_old_proposal_is_dispatched(
        self,
        session_with_fresh_confirm_on_old_proposal: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A FRESH confirmation of an OLD (~3-week) assistant proposal MUST
        still dispatch — confirming an old proposal is a fresh intent to act,
        and the age bound keys on the ``tool_confirmed`` event, not the
        assistant turn.  Guards against the bound over-reaching onto the
        assistant timestamp (which would silently drop legitimately-revived
        proposals).
        """
        pool, account_id, session_id = session_with_fresh_confirm_on_old_proposal

        pending = await _dispatch_confirmed_tools(
            pool,
            session_id,
            account_id=account_id,
            task_registry=TaskRegistry(),
        )
        pending_ids = [tc["id"] for tc in pending]

        assert pending_ids == ["tc_revived"], (
            "a fresh confirmation of a ~3-week-old proposal must dispatch (the "
            "bound is on the confirm event, not the assistant turn); got "
            f"{pending_ids} — if empty, the bound wrongly keys on assistant-time"
        )
        assert pending[0]["function"]["name"] == "signal_send"


class TestConfirmDispatchSweepSync:
    """Detection (sweep ``CONFIRMED_ROWS_SQL``) and dispatch
    (``list_confirmed_unresolved_tool_calls``) MUST agree on the same age
    bound, or the worker wakes a session it then can't make progress on (the
    #155 wake-with-no-progress symptom)."""

    async def test_stale_confirm_surfaced_by_neither(
        self,
        session_with_stale_connector_send: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A session whose only confirmed-unresolved call is stale (confirm >
        threshold) is surfaced for wake by NEITHER detection nor dispatch."""
        pool, account_id, session_id = session_with_stale_connector_send
        max_age = get_settings().confirmed_dispatch_max_age_seconds

        # Detection: the sweep's CONFIRMED_ROWS_SQL, scoped to this session,
        # must NOT return it (its only confirmed-unresolved call is stale).
        async with pool.acquire() as conn:
            detected = await conn.fetch(
                sweep.CONFIRMED_ROWS_SQL.format(scope_clause="AND s.id = $1", age_param="$2"),
                session_id,
                max_age,
            )
        assert [r["session_id"] for r in detected] == [], (
            "detection (CONFIRMED_ROWS_SQL) surfaced a session whose only "
            "confirmed-unresolved call is weeks-stale — it would wake the "
            "worker with no dispatchable call (wake-no-progress, #155)"
        )

        # Dispatch: the resolver, with the same bound, must agree (no call).
        async with pool.acquire() as conn:
            dispatchable = await queries.list_confirmed_unresolved_tool_calls(
                conn, session_id, account_id=account_id, max_age_seconds=max_age
            )
        assert dispatchable == [], (
            "dispatch resolver disagreed with detection on the stale call — "
            "the two predicates must stay in sync"
        )

    async def test_fresh_confirm_surfaced_by_both(
        self,
        session_with_fresh_confirm_on_old_proposal: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The inverse: a FRESH confirmation (on an old proposal) IS surfaced by
        BOTH detection and dispatch — they stay in sync on the positive case
        too, so a legitimately-dispatchable call still triggers a wake AND
        resolves to a tool_call."""
        pool, account_id, session_id = session_with_fresh_confirm_on_old_proposal
        max_age = get_settings().confirmed_dispatch_max_age_seconds

        async with pool.acquire() as conn:
            detected = await conn.fetch(
                sweep.CONFIRMED_ROWS_SQL.format(scope_clause="AND s.id = $1", age_param="$2"),
                session_id,
                max_age,
            )
        assert [r["session_id"] for r in detected] == [session_id], (
            "detection must surface the session with a fresh confirmation "
            "(even on an old proposal) so the worker wakes to dispatch it"
        )

        async with pool.acquire() as conn:
            dispatchable = await queries.list_confirmed_unresolved_tool_calls(
                conn, session_id, account_id=account_id, max_age_seconds=max_age
            )
        assert [tc["id"] for tc in dispatchable] == ["tc_revived"], (
            "dispatch must resolve the same fresh-confirmed call detection "
            "surfaced — the two predicates stay in sync on the positive case"
        )
