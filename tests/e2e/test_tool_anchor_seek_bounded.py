"""Perf guard: the tool-result stimulus anchor seek is O(distance), not
O(session) (issue #2575).

#2391 made the windower's retain-the-tail clamp anchor a tool-result stimulus
on its ISSUING assistant (``_latest_cumulative_tokens(stimulus_only=True)``).
The lookup was a flat ``ORDER BY seq DESC LIMIT 1`` with the JSONB
``tool_calls @> [{id}]`` containment beside ``cumulative_tokens IS NOT NULL``.
The planner estimated rows=1 for the containment and chose a Bitmap Heap Scan
over ``events_session_cumtokens_idx`` -- every message row of the session.
On Ultron (1.78M events) that cost ~5 s per tool hop on the per-step path.

The fix (``events.TOOL_ANCHOR_SQL``) walks ``seq`` backward from the tool
row's own seq through the tool-calling-assistant partial index, under a
literal LIMIT cap, and filters the containment on that bounded set.

Oracle: deterministic actual row counts from ``EXPLAIN (ANALYZE, FORMAT
JSON)`` (rows emitted + rows removed by filter / recheck on every ``events``
scan node), never wall-clock. Every SQL statement ``read_windowed_events``
issues on a >=50k-message session whose newest stimulus is a tool result is
captured and replayed under EXPLAIN ANALYZE; none may touch more than
``_ROW_BUDGET`` rows or use a Bitmap Heap Scan. The not-vacuous probe runs the
pre-fix SQL on the same slate and requires the oracle to go RED.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import asyncpg
import pytest

from aios.db.queries import events as events_q
from tests.conftest import needs_docker
from tests.e2e.test_read_path_complexity import seed_large_session

pytestmark = [pytest.mark.docker, needs_docker]

_ACCOUNT_ID = "acc_test_stub"
_SESSION_ID = "sess_tool_anchor_2575"
_N_MESSAGES = 50_000
_DELTA = 10
_TOOL_CALL_ID = "tc_anchor_2575"

# Upper bound on rows any single read on this path may touch. The anchor walk
# is capped at TOOL_ANCHOR_WALK_CAP; the retained window here is a handful of
# rows. An O(session) plan touches ~_N_MESSAGES.
_ROW_BUDGET = events_q.TOOL_ANCHOR_WALK_CAP + 256

# The pre-#2575 form of the anchor seek (the mutation the guard must catch).
_SQL_PRE_2575_ANCHOR = (
    "SELECT cumulative_tokens FROM events "
    "WHERE session_id = $1 AND kind = 'message' "
    "  AND data->>'role' = 'assistant' "
    "  AND data ? 'tool_calls' "
    "  AND data->'tool_calls' @> jsonb_build_array("
    "    jsonb_build_object('id', $2::text)) "
    "  AND cumulative_tokens IS NOT NULL "
    "ORDER BY seq DESC LIMIT 1"
)


async def _seed(pool: asyncpg.Pool[Any]) -> tuple[int, int]:
    """Seed ``_N_MESSAGES`` rows, then a matching assistant(tool_calls) +
    tool-result pair whose parent sits a few rows back from the tool row (the
    production shape: the match is ~20 rows back). Returns
    ``(assistant_seq, tool_seq)``."""
    await seed_large_session(
        pool,
        _ACCOUNT_ID,
        _N_MESSAGES,
        window_tokens=_N_MESSAGES * _DELTA // 2,
        session_id=_SESSION_ID,
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": _TOOL_CALL_ID,
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
    }
    rows: list[tuple[str, dict[str, Any]]] = [("assistant", assistant)]
    # A few non-matching tool-calling assistants between parent and result,
    # so the walk has real distance to cover.
    for i in range(20):
        rows.append(
            (
                "assistant",
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc_between_{i}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            )
        )
    rows.append(("tool", {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "ok"}))
    async with pool.acquire() as conn:
        base = _N_MESSAGES
        cum = _N_MESSAGES * _DELTA
        for offset, (role, data) in enumerate(rows, start=1):
            await conn.execute(
                "INSERT INTO events (id, session_id, seq, kind, data, created_at, "
                "cumulative_tokens, cumulative_messages, cumulative_text_mass, "
                "cumulative_tool_result_mass, cumulative_thinking_mass, "
                "cumulative_tool_use_mass, role, account_id) "
                "VALUES ($1, $2, $3, 'message', $4::jsonb, now(), $5, $6, 0, 0, 0, $5, $7, $8)",
                f"ev_anchor_2575_{offset}",
                _SESSION_ID,
                base + offset,
                json.dumps(data),
                cum + offset * _DELTA,
                base + offset,
                role,
                _ACCOUNT_ID,
            )
        await conn.execute(
            "UPDATE sessions SET last_event_seq = $2 WHERE id = $1",
            _SESSION_ID,
            base + len(rows),
        )
        await conn.execute("ANALYZE events")
    return base + 1, base + len(rows)


@pytest.fixture
async def seeded(aios_env: dict[str, str]) -> AsyncIterator[tuple[asyncpg.Pool[Any], int, int]]:
    from aios.db.pool import create_pool

    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=4)
    try:
        assistant_seq, tool_seq = await _seed(pool)
        yield pool, assistant_seq, tool_seq
    finally:
        await pool.close()


async def _explain_analyze(conn: asyncpg.Connection[Any], sql: str, *args: Any) -> dict[str, Any]:
    result = await conn.fetchval(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", *args)
    if isinstance(result, str):
        result = json.loads(result)
    return cast(dict[str, Any], result[0]["Plan"])


def _events_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("Relation Name") == "events":
            found.append(node)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return found


def _rows_touched(node: dict[str, Any]) -> int:
    loops = max(int(node.get("Actual Loops", 1)), 1)
    per_loop = (
        float(node.get("Actual Rows", 0))
        + float(node.get("Rows Removed by Filter", 0))
        + float(node.get("Rows Removed by Index Recheck", 0))
    )
    return int(per_loop * loops)


def _offenders(plan: dict[str, Any]) -> list[str]:
    """Plan-shape + row-budget violations over every ``events`` scan node."""
    out: list[str] = []
    for node in _events_nodes(plan):
        if node.get("Node Type") == "Bitmap Heap Scan":
            out.append(f"Bitmap Heap Scan on events (index {node.get('Index Name')!r})")
        touched = _rows_touched(node)
        if touched > _ROW_BUDGET:
            out.append(
                f"{node.get('Node Type')} on events via {node.get('Index Name')!r} "
                f"touched {touched} rows > budget {_ROW_BUDGET}"
            )
    return out


class _RecordingConn:
    """Delegating asyncpg connection proxy that records every read issued."""

    def __init__(self, conn: asyncpg.Connection[Any]) -> None:
        self._conn = conn
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return await self._conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return await self._conn.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return await self._conn.fetchval(sql, *args)


class TestToolAnchorSeekBounded:
    async def test_anchor_is_the_issuing_assistant(
        self, seeded: tuple[asyncpg.Pool[Any], int, int]
    ) -> None:
        """Semantics preserved (#2391): the anchor is the issuing assistant's
        ``cumulative_tokens``, not the tool row's."""
        pool, assistant_seq, _tool_seq = seeded
        async with pool.acquire() as conn:
            expected = await conn.fetchval(
                "SELECT cumulative_tokens FROM events WHERE session_id = $1 AND seq = $2",
                _SESSION_ID,
                assistant_seq,
            )
            anchor = await events_q._latest_cumulative_tokens(conn, _SESSION_ID, stimulus_only=True)
        assert anchor == expected

    async def test_anchor_seek_plan_is_bounded(
        self, seeded: tuple[asyncpg.Pool[Any], int, int]
    ) -> None:
        pool, _assistant_seq, tool_seq = seeded
        async with pool.acquire() as conn:
            plan = await _explain_analyze(
                conn, events_q.TOOL_ANCHOR_SQL, _SESSION_ID, _TOOL_CALL_ID, tool_seq
            )
            # Worst case: no parent at all -> the walk stops at the cap.
            miss_plan = await _explain_analyze(
                conn, events_q.TOOL_ANCHOR_SQL, _SESSION_ID, "tc_no_such_call", tool_seq
            )
        assert not _offenders(plan), (_offenders(plan), plan)
        assert not _offenders(miss_plan), (_offenders(miss_plan), miss_plan)

    async def test_read_windowed_events_issues_only_bounded_plans(
        self, seeded: tuple[asyncpg.Pool[Any], int, int]
    ) -> None:
        """Every statement ``read_windowed_events`` issues on a 50k-message
        session whose newest stimulus is a tool result stays within the row
        budget and never takes a Bitmap Heap Scan over ``events``."""
        pool, assistant_seq, _tool_seq = seeded
        async with pool.acquire() as conn:
            rec = _RecordingConn(conn)
            windowed = await events_q.read_windowed_events(
                cast("asyncpg.Connection[Any]", rec),
                _SESSION_ID,
                account_id=_ACCOUNT_ID,
                window_min=0,
                window_max=40,
                model=f"fake/anchor-{uuid.uuid4().hex[:8]}",
                overhead_local=0,
            )
            assert rec.calls, "read_windowed_events issued no SQL"
            violations: dict[str, list[str]] = {}
            for sql, args in rec.calls:
                plan = await _explain_analyze(conn, sql, *args)
                bad = _offenders(plan)
                if bad:
                    violations[sql] = bad
        assert not violations, violations
        assert any(sql == events_q.TOOL_ANCHOR_SQL for sql, _ in rec.calls)
        # The clamp still retains the issuing assistant (the #2391 anchor).
        assert windowed.events and windowed.events[0].seq <= assistant_seq

    async def test_pre_2575_unbounded_query_is_red(
        self, seeded: tuple[asyncpg.Pool[Any], int, int]
    ) -> None:
        """Not-vacuous: the reverted (unbounded) anchor seek trips the oracle
        on the same slate. A miss (parent absent, the worst case) has to
        consider every tool-calling assistant in the session."""
        pool, _assistant_seq, _tool_seq = seeded
        async with pool.acquire() as conn:
            plan = await _explain_analyze(
                conn, _SQL_PRE_2575_ANCHOR, _SESSION_ID, "tc_no_such_call"
            )
        assert _offenders(plan), plan
