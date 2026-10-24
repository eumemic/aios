"""Integration tests for #2393: the awaiting read-model's lifecycle-marker
lookup is bounded by the unresolved calls, not by lifecycle volume.

``list_unresolved_tool_calls_batch`` used to select every
``tool_confirmed``/``allow`` and ``tool_requested`` marker of every page
session (``session_id = ANY(page)``), riding
``events_session_lifecycle_seq_idx`` and heap-filtering every lifecycle row.
On prod that hit the statement timeout and returned 500 on ``GET /v1/sessions``.
The lookup now probes exactly the ``(session_id, tool_call_id)`` pairs that
``_unresolved_tool_calls`` returned, served by ``events_tool_marker_idx``
(migration 0195).

Discrimination:

* ``test_marker_query_receives_and_returns_only_unresolved_pairs`` FAILS IF THE
  RESTRICTION IS DROPPED.  The fixture has many sessions, each with many
  markers for RESOLVED calls plus lifecycle noise.  The marker query must be
  handed only the unresolved ids, and it must return only their markers.
* ``test_awaiting_flags_unchanged`` pins identical output: the allow/requested
  bits for unresolved calls, a ``tool_confirmed``/``deny`` that is NOT an allow,
  a marker carried on a DIFFERENT session for the same tool_call_id (it must not
  leak across sessions), and a foreign-account marker row (it must not count).
* ``test_marker_lookup_plan_uses_marker_index`` pins the plan shape: an index
  probe on ``events_tool_marker_idx``, not the lifecycle-wide scan.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.models.sessions import Session
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_awaiting_marker_bound"
_OTHER_ACCOUNT = "acc_awaiting_marker_other"
_N_SESSIONS = 12
_N_RESOLVED_TURNS = 150  # per session: assistant + tool result + 2 markers + 3 noise


def _assistant(*tool_call_ids: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": t, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            for t in tool_call_ids
        ],
    }


class _RecordingConn:
    """Delegates to a real asyncpg connection, recording every ``fetch``."""

    def __init__(self, conn: asyncpg.Connection[Any]) -> None:
        self._conn = conn
        self.fetches: list[tuple[str, tuple[Any, ...], list[asyncpg.Record]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        rows: list[asyncpg.Record] = await self._conn.fetch(sql, *args)
        self.fetches.append((sql, args, rows))
        return rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


async def _bulk_history(conn: asyncpg.Connection[Any], session_id: str, account_id: str) -> None:
    """Seed ``_N_RESOLVED_TURNS`` resolved turns directly (``append_event`` per
    row would make the fixture slow).  Each turn has an assistant call, its
    tool result, a ``tool_requested`` marker, a ``tool_confirmed``/``allow``
    marker, and three lifecycle noise rows."""
    base = await conn.fetchval("SELECT last_event_seq FROM sessions WHERE id = $1", session_id)
    n = _N_RESOLVED_TURNS * 7
    await conn.execute(
        """
        INSERT INTO events (id, session_id, seq, kind, data, role, account_id)
        SELECT 'evt_mb_' || $1 || '_' || g,
               $1,
               $3 + g,
               CASE WHEN g % 7 IN (1, 2) THEN 'message' ELSE 'lifecycle' END,
               CASE g % 7
                 WHEN 1 THEN jsonb_build_object(
                   'role', 'assistant', 'content', '',
                   'tool_calls', jsonb_build_array(jsonb_build_object(
                     'id', 'tc_done_' || $1 || '_' || (g / 7), 'type', 'function',
                     'function', jsonb_build_object('name', 'bash', 'arguments', '{}'))))
                 WHEN 2 THEN jsonb_build_object(
                   'role', 'tool', 'tool_call_id', 'tc_done_' || $1 || '_' || (g / 7),
                   'content', 'ok')
                 WHEN 3 THEN jsonb_build_object(
                   'event', 'tool_requested', 'tool_call_id', 'tc_done_' || $1 || '_' || (g / 7))
                 WHEN 4 THEN jsonb_build_object(
                   'event', 'tool_confirmed', 'result', 'allow',
                   'tool_call_id', 'tc_done_' || $1 || '_' || (g / 7))
                 WHEN 5 THEN '{"event": "request_opened", "request_id": "r"}'::jsonb
                 WHEN 6 THEN '{"event": "request_response", "request_id": "r"}'::jsonb
                 ELSE '{"event": "turn_ended", "stop_reason": "end_turn"}'::jsonb
               END,
               CASE g % 7 WHEN 1 THEN 'assistant' WHEN 2 THEN 'tool' END,
               $2
          FROM generate_series(1, $4::int) AS g
        """,
        session_id,
        account_id,
        base,
        n,
    )
    await conn.execute(
        "UPDATE sessions SET last_event_seq = last_event_seq + $2 WHERE id = $1",
        session_id,
        n,
    )


async def _lifecycle(conn: asyncpg.Connection[Any], session: Session, data: dict[str, Any]) -> None:
    await queries.append_event(
        conn, account_id=_ACCOUNT, session_id=session.id, kind="lifecycle", data=data
    )


@pytest.fixture
async def many_sessions(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], list[Session]]]:
    """``_N_SESSIONS`` sessions with a heavy resolved history each.

    Open calls:
    * s0: ``tc_req`` (has tool_requested) and ``tc_allow`` (has tool_confirmed/allow).
    * s1: ``tc_deny`` (tool_confirmed/deny only, NOT an allow), and ``tc_shared``
      whose markers live ONLY on s2.
    * s2: ``tc_plain`` (no markers), plus markers for ``tc_shared`` (s1's call).
    * s3: ``tc_foreign`` whose only marker row carries a FOREIGN account_id.
    * the rest: no open calls.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, $1)",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, $2, FALSE, $1)",
                _OTHER_ACCOUNT,
                _ACCOUNT,
            )
        sessions: list[Session] = []
        for i in range(_N_SESSIONS):
            _a, _e, s = await seed_agent_env_session(
                pool, account_id=_ACCOUNT, prefix=f"marker-bound-{i}", tools=[ToolSpec(type="bash")]
            )
            sessions.append(s)
        async with pool.acquire() as conn:
            for s in sessions:
                await _bulk_history(conn, s.id, _ACCOUNT)
            s0, s1, s2, s3 = sessions[:4]
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=s0.id,
                kind="message",
                data=_assistant("tc_req", "tc_allow"),
            )
            await _lifecycle(conn, s0, {"event": "tool_requested", "tool_call_id": "tc_req"})
            await _lifecycle(
                conn, s0, {"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc_allow"}
            )
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=s1.id,
                kind="message",
                data=_assistant("tc_deny", "tc_shared"),
            )
            await _lifecycle(
                conn, s1, {"event": "tool_confirmed", "result": "deny", "tool_call_id": "tc_deny"}
            )
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=s2.id,
                kind="message",
                data=_assistant("tc_plain"),
            )
            await _lifecycle(conn, s2, {"event": "tool_requested", "tool_call_id": "tc_shared"})
            await _lifecycle(
                conn,
                s2,
                {"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc_shared"},
            )
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=s3.id,
                kind="message",
                data=_assistant("tc_foreign"),
            )
            # A marker row on s3 under a foreign account_id: the account
            # predicate must keep it out (forged directly; append_event always
            # writes the session's own account).
            seq = await conn.fetchval(
                "UPDATE sessions SET last_event_seq = last_event_seq + 1 "
                "WHERE id = $1 RETURNING last_event_seq",
                s3.id,
            )
            await conn.execute(
                "INSERT INTO events (id, session_id, seq, kind, data, account_id) "
                "VALUES ('evt_mb_foreign', $1, $2, 'lifecycle', $3::jsonb, $4)",
                s3.id,
                seq,
                json.dumps(
                    {"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc_foreign"}
                ),
                _OTHER_ACCOUNT,
            )
            await conn.execute("ANALYZE events")
        yield pool, sessions
    finally:
        await pool.close()


_EXPECTED_FLAGS = {
    # (session index, tool_call_id): (has_allow_lifecycle, has_requested_lifecycle)
    (0, "tc_req"): (False, True),
    (0, "tc_allow"): (True, False),
    (1, "tc_deny"): (False, False),
    (1, "tc_shared"): (False, False),
    (2, "tc_plain"): (False, False),
    (3, "tc_foreign"): (False, False),
}


def _walk(node: dict[str, Any]) -> Any:
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


def _marker_fetch(
    rec: _RecordingConn,
) -> tuple[str, tuple[Any, ...], list[asyncpg.Record]]:
    hits = [f for f in rec.fetches if "kind = 'lifecycle'" in f[0]]
    assert len(hits) == 1, "expected exactly one lifecycle-marker fetch"
    return hits[0]


class TestAwaitingMarkerLookupBounded:
    async def test_awaiting_flags_unchanged(
        self, many_sessions: tuple[asyncpg.Pool[Any], list[Session]]
    ) -> None:
        pool, sessions = many_sessions
        async with pool.acquire() as conn:
            batch = await queries.list_unresolved_tool_calls_batch(
                conn, [s.id for s in sessions], account_id=_ACCOUNT
            )
        index_of = {s.id: i for i, s in enumerate(sessions)}
        got = {
            (index_of[sid], e["tool_call_id"]): (
                e["has_allow_lifecycle"],
                e["has_requested_lifecycle"],
            )
            for sid, entries in batch.items()
            for e in entries
        }
        assert got == _EXPECTED_FLAGS
        # Intra-session order unchanged (seq, then tool_calls ordinal).
        assert [e["tool_call_id"] for e in batch[sessions[0].id]] == ["tc_req", "tc_allow"]
        assert [e["tool_call_id"] for e in batch[sessions[1].id]] == ["tc_deny", "tc_shared"]

    async def test_marker_query_receives_and_returns_only_unresolved_pairs(
        self, many_sessions: tuple[asyncpg.Pool[Any], list[Session]]
    ) -> None:
        """FAILS IF THE ``tool_call_id`` RESTRICTION IS DROPPED.  Every session
        carries ``2 * _N_RESOLVED_TURNS`` markers for RESOLVED calls, and none
        of them may reach the marker query's input or its output."""
        pool, sessions = many_sessions
        async with pool.acquire() as conn:
            rec = _RecordingConn(conn)
            await queries.list_unresolved_tool_calls_batch(
                cast(Any, rec), [s.id for s in sessions], account_id=_ACCOUNT
            )
        _sql, args, rows = _marker_fetch(rec)
        unresolved = {tcid for (_i, tcid) in _EXPECTED_FLAGS}
        # Input: the only tool_call_ids handed to the query are the unresolved ones.
        passed_ids = {a for arg in args if isinstance(arg, list) for a in arg} - {
            s.id for s in sessions
        }
        assert passed_ids == unresolved
        # Output: only markers of unresolved calls come back.  On the
        # unrestricted shape this returns 2 * _N_RESOLVED_TURNS * _N_SESSIONS
        # (+ the s2 tc_shared pair) rows.
        returned = {(r["session_id"], r["tool_call_id"], r["event"]) for r in rows}
        s0, s1, s2, _s3 = sessions[:4]
        assert returned == {
            (s0.id, "tc_req", "tool_requested"),
            (s0.id, "tc_allow", "tool_confirmed"),
        }, f"marker query returned {len(rows)} rows; expected only unresolved pairs"
        assert all(not r["tool_call_id"].startswith("tc_done_") for r in rows)
        # s1's tc_shared markers live on s2.  They are not s1's, and s2 has no
        # open tc_shared call, so they must not be fetched at all.
        assert (s2.id, "tc_shared", "tool_requested") not in returned
        assert s1.id not in {r["session_id"] for r in rows}

    async def test_no_unresolved_skips_marker_query(
        self, many_sessions: tuple[asyncpg.Pool[Any], list[Session]]
    ) -> None:
        """Sessions with no open calls never pay for the marker lookup."""
        pool, sessions = many_sessions
        quiet = [s.id for s in sessions[4:]]
        async with pool.acquire() as conn:
            rec = _RecordingConn(conn)
            batch = await queries.list_unresolved_tool_calls_batch(
                cast(Any, rec), quiet, account_id=_ACCOUNT
            )
        assert batch == {}
        assert not [f for f in rec.fetches if "kind = 'lifecycle'" in f[0]]

    async def test_marker_lookup_plan_uses_marker_index(
        self, many_sessions: tuple[asyncpg.Pool[Any], list[Session]]
    ) -> None:
        """Generic-plan EXPLAIN of the exact marker SQL (asyncpg prepares
        statements, so the generic plan is the one that matters) probes
        ``events_tool_marker_idx``, not the lifecycle-wide index."""
        pool, sessions = many_sessions
        async with pool.acquire() as conn:
            rec = _RecordingConn(conn)
            await queries.list_unresolved_tool_calls_batch(
                cast(Any, rec), [s.id for s in sessions], account_id=_ACCOUNT
            )
            sql, args, _rows = _marker_fetch(rec)
            async with conn.transaction():
                await conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
                plan = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *args)
        if isinstance(plan, str):
            plan = json.loads(plan)
        nodes = list(_walk(plan[0]["Plan"]))
        events_scans = [n for n in nodes if n.get("Relation Name") == "events"]
        assert events_scans, plan
        assert all(n.get("Index Name") == "events_tool_marker_idx" for n in events_scans), plan
        assert all(n["Node Type"] != "Seq Scan" for n in events_scans), plan
