"""Perf regression tests for sweep queries (issue #140).

The sweep path ran correlated subqueries against ``events`` that computed
``MAX(reacting_to)`` per session inside a per-row SubPlan — an N+1 pattern
that on JN's live data made one sweep pass cost ~7.5s on a 10k-event
table. The fix in this PR hoists the aggregation to a CTE and swaps
``data->>'role'`` for the normalized ``role`` column (from migration 0022).

These tests encode the fix as a **structural** invariant: after
``EXPLAIN (FORMAT JSON)``, no node in the plan tree is a SubPlan scanning
``events``. Deterministic, no wall-clock assertion, no flake on slow CI.
A regression from a well-meaning refactor ("why is this a CTE?") fails
this test immediately.

The budget smoke test (``@pytest.mark.slow``) is a secondary fence: on
the same seeded fixture, assert the rewritten queries actually complete
under a buffer-hits budget. Catches plan-choice regressions the
structural check might miss (e.g., Postgres version that reshapes the
plan in some other quadratic direction).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.harness.sweep import CANDIDATE_ROWS_SQL, UNREACTED_ROWS_SQL
from tests.conftest import needs_docker
from tests.support import find_subplans_over_events

# ─── fixture: pathological session ───────────────────────────────────────────


_N_SESSIONS = 3
_N_ASSISTANT_PER = 400
_N_UNREACTED_PER = 30


async def _seed_pathological(pool: asyncpg.Pool[Any]) -> list[str]:
    """Seed the shape that triggers the pre-#140 correlated-subquery
    pathology: multiple sessions, each with many assistant messages
    carrying ``reacting_to``, plus unreacted user messages at the tail.
    Returns the seeded session ids.
    """
    session_ids: list[str] = [f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)]

    async with pool.acquire() as conn:
        # Dependencies for the session FK chain.
        await conn.execute(
            """
            INSERT INTO agents (id, name, model)
            VALUES ('agt_perf', 'perf', 'openrouter/x')
            ON CONFLICT (id) DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO environments (id, name)
            VALUES ('env_perf', 'env_perf')
            ON CONFLICT (id) DO NOTHING
            """
        )

        for sid in session_ids:
            await conn.execute(
                """
                INSERT INTO sessions (id, agent_id, environment_id, status, workspace_volume_path)
                VALUES ($1, 'agt_perf', 'env_perf', 'idle', '/tmp/ws_' || $1)
                ON CONFLICT (id) DO NOTHING
                """,
                sid,
            )
            # Seed assistant messages. Each has reacting_to pointing back to
            # the previous user event (seq = 2*i for i-th assistant).
            rows = []
            seq = 1
            for i in range(_N_ASSISTANT_PER):
                # user msg
                rows.append(
                    (
                        f"ev_u_{sid}_{i}",
                        sid,
                        seq,
                        "message",
                        json.dumps({"role": "user", "content": f"u{i}"}),
                        "user",
                    )
                )
                seq += 1
                # assistant reply referring back to that user msg
                rows.append(
                    (
                        f"ev_a_{sid}_{i}",
                        sid,
                        seq,
                        "message",
                        json.dumps(
                            {
                                "role": "assistant",
                                "content": f"a{i}",
                                "reacting_to": seq - 1,
                            }
                        ),
                        "assistant",
                    )
                )
                seq += 1
            # A handful of **unreacted** user messages at the tail —
            # candidates the sweep must notice.
            for j in range(_N_UNREACTED_PER):
                rows.append(
                    (
                        f"ev_tail_u_{sid}_{j}",
                        sid,
                        seq,
                        "message",
                        json.dumps({"role": "user", "content": f"tail{j}"}),
                        "user",
                    )
                )
                seq += 1
            await conn.executemany(
                "INSERT INTO events (id, session_id, seq, kind, data, role) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6) "
                "ON CONFLICT (id) DO NOTHING",
                rows,
            )

        # ANALYZE so the planner has fresh stats matching the fixture.
        await conn.execute("ANALYZE events")
        await conn.execute("ANALYZE sessions")

    return session_ids


@pytest.fixture
async def seeded_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    from aios.db.pool import create_pool

    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=4)
    await _seed_pathological(pool)
    try:
        yield pool
    finally:
        await pool.close()


# ─── plan tree helpers ───────────────────────────────────────────────────────


async def _explain(pool: asyncpg.Pool[Any], sql: str, *args: Any) -> dict[str, Any]:
    async with pool.acquire() as conn:
        result = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *args)
    # asyncpg decodes JSON columns to Python lists/dicts already.
    if isinstance(result, str):
        result = json.loads(result)
    return result[0]["Plan"]


# ─── structural tests (primary, always-on) ───────────────────────────────────


@needs_docker
class TestNoCorrelatedSubplanOverEvents:
    """The central invariant: no sweep query may re-scan ``events`` inside
    a correlated SubPlan. That shape was the N+1 pathology behind #140."""

    async def test_candidate_rows_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        plan = await _explain(
            seeded_pool, CANDIDATE_ROWS_SQL.format(scope_clause="", cte_scope_clause="")
        )
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in find_sessions_needing_inference candidate query: "
            f"{len(found)} correlated subplan(s) over events. "
            f"See PR #145 — this query must hoist MAX(reacting_to) via a CTE."
        )

    async def test_unreacted_rows_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        session_ids = [f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)]
        plan = await _explain(seeded_pool, UNREACTED_ROWS_SQL, session_ids)
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in _filter_incomplete_batches unreacted query: "
            f"{len(found)} correlated subplan(s) over events. Same CTE fix applies."
        )


# ─── budget smoke (secondary, slow-marker) ───────────────────────────────────


@needs_docker
@pytest.mark.slow
class TestSweepQueryBudget:
    """Buffer-hit budget as a backstop for plan-choice regressions the
    structural check might miss. Buffer hits are far more stable across
    hardware than wall-clock."""

    async def test_candidate_rows_buffer_budget(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        async with seeded_pool.acquire() as conn:
            result = await conn.fetchval(
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {CANDIDATE_ROWS_SQL.format(scope_clause='', cte_scope_clause='')}"
            )
        if isinstance(result, str):
            result = json.loads(result)
        root = result[0]["Plan"]
        hits = _total_buffer_hits(root)
        # Fixture seeds ~2600 events; linear-scan cost is ~2.6k hits. We
        # allow generous headroom; the pre-fix path burnt ~1.8M here.
        assert hits < 50_000, f"candidate_rows uses {hits} buffer hits — N+1 regression?"


def _total_buffer_hits(plan_node: dict[str, Any]) -> int:
    total = int(plan_node.get("Shared Hit Blocks", 0) or 0)
    for child in plan_node.get("Plans", []):
        total += _total_buffer_hits(child)
    return total
