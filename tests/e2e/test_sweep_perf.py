"""Perf regression tests for sweep queries (issue #140).

The sweep path originally ran correlated subqueries against ``events`` that
computed ``MAX(reacting_to)`` per session inside a per-row SubPlan — an N+1
pattern that on JN's live data made one sweep pass cost ~7.5s on a 10k-event
table. ``CANDIDATE_ROWS_SQL`` and ``ERRORED_SESSIONS_SQL`` now use scalar
columns maintained on the ``sessions`` row (migration 0066) — simple column
arithmetic with no event-log scan at all.

These tests encode the fix as a **structural** invariant: after
``EXPLAIN (FORMAT JSON)``, no node in the plan tree is a SubPlan scanning
``events``. Deterministic, no wall-clock assertion, no flake on slow CI.
A regression from a well-meaning refactor fails this test immediately.

The budget smoke test (``@pytest.mark.slow``) is a secondary fence: on
the same seeded fixture, assert the rewritten queries actually complete
under a buffer-hits budget. Catches plan-choice regressions the
structural check might miss (e.g., Postgres version that reshapes the
plan in some other quadratic direction).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import asyncpg
import pytest

from aios.db.queries import _SESSION_STATUS_EXPR
from aios.db.queries.events import UNHARVESTED_MODEL_DISPATCH_PARKS_SQL
from aios.harness.sweep import (
    BATCH_RESULT_ROWS_SQL,
    CANDIDATE_ROWS_SQL,
    ERRORED_SESSIONS_SQL,
    FAST_PATH_PENDING_WORK_SQL,
    GHOST_ASST_SQL,
    GHOST_SPAN_START_SQL,
    OPEN_CANDIDATES_ASST_SQL,
    REFERENCED_ASST_BATCH_SQL,
    UNREACTED_ROWS_SQL,
)
from tests.conftest import needs_docker
from tests.support import find_seq_scans_over_events, find_subplans_over_events

pytestmark = pytest.mark.docker

# ─── fixture: pathological session ───────────────────────────────────────────


_N_SESSIONS = 3
_N_ASSISTANT_PER = 400
_N_UNREACTED_PER = 30

# Tool-call-heavy sessions. The sess_perf_* sessions carry no tool_calls, so the
# expensive branch of ``_SESSION_ACTIVE_EXPR`` — the CROSS JOIN LATERAL over
# ``tool_calls`` + the NOT EXISTS anti-join over tool results — never runs on
# them (the OR short-circuits on the unreacted-stimulus branch). These sessions
# are fully resolved AND fully reacted (a final assistant message reacts to the
# tail), so the active derivation can't short-circuit: it must scan every
# tool_call and confirm each has a result. That full-scan worst case is what the
# read-time budget test guards — a missing index / quadratic re-scan here would
# blow the buffer budget (the per-row-correlated active expr can't be locked
# structurally with ``find_subplans_over_events`` the way the sweep's bulk CTE
# queries are: its SubPlans over events are inherent and keyset+LIMIT-bounded).
_N_TC_SESSIONS = 3
_N_TC_ASST_PER = 30  # assistant messages carrying tool_calls
_N_TC_PER_ASST = 4  # tool_calls per assistant message

# One DEEP, fully-resolved tool-call session (#840). Its 500 assistant turns x
# 4 tool_calls = 2000 fully-paired tool calls give the cross-session ghost scan
# a large resolved history to walk if it were *not* bounded by
# ``sessions.open_tool_call_count > 0``. Because every call has a matching tool
# result, the all-sessions backfill UPDATE below sets its
# ``open_tool_call_count = 0``, so the bounded ``GHOST_ASST_SQL`` must skip it
# entirely and return zero rows.
_N_TC_DEEP_ASST = 500


def _tc_session_rows(session_id: str, n_asst: int) -> list[tuple[Any, ...]]:
    """Build the event rows for one fully-resolved, fully-reacted tool-call
    session: ``n_asst`` assistant messages each carrying ``_N_TC_PER_ASST``
    tool_calls, every call paired with a tool result, plus a final assistant
    message reacting to the tail (so there is no unreacted stimulus). The
    ``sess_tc_*`` and ``sess_tc_deep`` fixtures differ ONLY in id and depth;
    everything else about the row shape is shared here.
    """
    rows: list[tuple[Any, ...]] = []
    seq = 1
    for i in range(n_asst):
        tcids = [f"tc_{session_id}_{i}_{k}" for k in range(_N_TC_PER_ASST)]
        rows.append(
            (
                f"ev_a_{session_id}_{i}",
                session_id,
                seq,
                "message",
                json.dumps(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc,
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                            for tc in tcids
                        ],
                        "reacting_to": seq - 1,
                    }
                ),
                "assistant",
            )
        )
        seq += 1
        for tc in tcids:
            rows.append(
                (
                    f"ev_t_{session_id}_{i}_{tc}",
                    session_id,
                    seq,
                    "message",
                    json.dumps({"role": "tool", "tool_call_id": tc, "content": "ok"}),
                    "tool",
                )
            )
            seq += 1
    # Final assistant message reacts to every tool result, so there is no
    # unreacted stimulus — the active OR can't short-circuit and must evaluate
    # the (fully-resolved) tool_call branch.
    rows.append(
        (
            f"ev_a_{session_id}_final",
            session_id,
            seq,
            "message",
            json.dumps({"role": "assistant", "content": "done", "reacting_to": seq - 1}),
            "assistant",
        )
    )
    return rows


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
            INSERT INTO agents (id, name, model, account_id)
            VALUES ('agt_perf', 'perf', 'openrouter/x', 'acc_test_stub')
            ON CONFLICT (id) DO NOTHING
            """
        )
        await conn.execute(
            """
            INSERT INTO environments (id, name, account_id)
            VALUES ('env_perf', 'env_perf', 'acc_test_stub')
            ON CONFLICT (id) DO NOTHING
            """
        )

        for sid in session_ids:
            await conn.execute(
                """
                INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
                VALUES ($1, 'agt_perf', 'env_perf', '/tmp/ws_' || $1, 'acc_test_stub')
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
                "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_test_stub') "
                "ON CONFLICT (id) DO NOTHING",
                rows,
            )

        # Tool-call-heavy sessions (see module comment): fully resolved + fully
        # reacted, so the active derivation runs its tool_call branch to
        # completion (no short-circuit) and the session still derives idle.
        for t in range(_N_TC_SESSIONS):
            tsid = f"sess_tc_{t:03d}"
            await conn.execute(
                "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
                "VALUES ($1, 'agt_perf', 'env_perf', '/tmp/ws_' || $1, 'acc_test_stub') "
                "ON CONFLICT (id) DO NOTHING",
                tsid,
            )
            rows = _tc_session_rows(tsid, _N_TC_ASST_PER)
            await conn.executemany(
                "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_test_stub') "
                "ON CONFLICT (id) DO NOTHING",
                rows,
            )

        # One DEEP, fully-resolved tool-call session (#840): same shape as the
        # ``sess_tc_*`` loop above but ``_N_TC_DEEP_ASST`` assistant turns deep.
        # Every tool_call is paired with a tool result, so the all-sessions
        # backfill UPDATE below derives its ``open_tool_call_count = 0`` — making
        # it the bait for the bounded ghost scan: an unbounded GHOST_ASST_SQL
        # would walk all 500 assistant-with-tool_calls rows; the bounded one must
        # skip it. Inserted with the SAME FK pattern (agt_perf/env_perf,
        # acc_test_stub) as the ``sess_tc_*`` sessions.
        deep_sid = "sess_tc_deep"
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
            "VALUES ($1, 'agt_perf', 'env_perf', '/tmp/ws_' || $1, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            deep_sid,
        )
        rows = _tc_session_rows(deep_sid, _N_TC_DEEP_ASST)
        await conn.executemany(
            "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            rows,
        )

        # One small session with a GENUINELY UNRESOLVED tool call (#840): a user
        # message, then an assistant message carrying exactly one tool_call
        # (``tc_open_0``) with NO matching tool-role result. This is NOT built
        # from ``_tc_session_rows`` (which produces fully-RESOLVED sessions);
        # the rows are written inline so the open call stays open. The
        # all-sessions backfill UPDATE below (no id filter) derives its
        # ``open_tool_call_count = 1``, so the bounded ``GHOST_ASST_SQL`` MUST
        # return this session — the pass-through half of the bidirectional fence.
        # Same FK pattern (agt_perf/env_perf, acc_test_stub) as the sess_tc_*.
        open_sid = "sess_tc_open"
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
            "VALUES ($1, 'agt_perf', 'env_perf', '/tmp/ws_' || $1, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            open_sid,
        )
        open_rows: list[tuple[Any, ...]] = [
            (
                f"ev_u_{open_sid}_0",
                open_sid,
                1,
                "message",
                json.dumps({"role": "user", "content": "do a thing"}),
                "user",
            ),
            (
                f"ev_a_{open_sid}_0",
                open_sid,
                2,
                "message",
                json.dumps(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_open_0",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                        "reacting_to": 1,
                    }
                ),
                "assistant",
            ),
        ]
        await conn.executemany(
            "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            open_rows,
        )

        # Backfill the scalar columns on sessions to match the seeded events.
        # The production path maintains these in append_event; this test inserts
        # events directly, so it must reconcile the sessions rows manually.
        # The backfill SQL below intentionally mirrors migration 0066's
        # backfill logic — keep them in sync if either changes.
        await conn.execute(
            """
            UPDATE sessions s
               SET last_event_seq = COALESCE((
                       SELECT MAX(e.seq) FROM events e WHERE e.session_id = s.id
                   ), 0),
                   last_user_seq = COALESCE((
                       SELECT MAX(e.seq) FROM events e
                        WHERE e.session_id = s.id AND e.kind = 'message' AND e.role = 'user'
                   ), 0),
                   last_stimulus_seq = COALESCE((
                       SELECT MAX(e.seq) FROM events e
                        WHERE e.session_id = s.id AND e.kind = 'message' AND e.role <> 'assistant'
                   ), 0),
                   last_error_seq = COALESCE((
                       SELECT MAX(e.seq) FROM events e
                        WHERE e.session_id = s.id
                          AND e.kind = 'lifecycle' AND e.data->>'stop_reason' = 'error'
                   ), 0),
                   last_reacted_seq = COALESCE((
                       SELECT MAX(COALESCE((e.data->>'reacting_to')::bigint, e.seq))
                         FROM events e
                        WHERE e.session_id = s.id
                          AND e.kind = 'message' AND e.role = 'assistant'
                   ), 0),
                   open_tool_call_count = COALESCE((
                       SELECT COUNT(*)
                         FROM events ate
                        CROSS JOIN LATERAL jsonb_array_elements(ate.data->'tool_calls') tc
                        WHERE ate.session_id = s.id
                          AND ate.kind = 'message' AND ate.role = 'assistant'
                          AND ate.data ? 'tool_calls'
                          AND NOT EXISTS (
                              SELECT 1 FROM events tr
                               WHERE tr.session_id = s.id
                                 AND tr.kind = 'message' AND tr.role = 'tool'
                                 AND tr.data->>'tool_call_id' = tc->>'id'
                          )
                   ), 0)
            """
        )

        # A single unharvested ``model_workflow_park`` span (#1707). This makes
        # the migration-0131 park partial index non-empty on the seeded fixture,
        # so the cross-session crash-recovery scan
        # (``find_unharvested_model_dispatch_parks``) has a highly-selective index
        # to seek into. With the index empty, a cost-based planner can (version-
        # dependently, e.g. PG16) still pick a ``Seq Scan on events`` because the
        # empty partial index and a full scan cost ~the same; one real matching
        # row makes the partial index unambiguously cheaper than scanning the
        # whole event log, on every supported Postgres version. Deliberately left
        # unharvested (no ``model_workflow_harvest``/``harvest_end`` span for its
        # ``run_id``) so it survives both ``NOT EXISTS`` anti-joins and the scan
        # returns it — exercising the exact production plan the test EXPLAINs.
        await conn.execute(
            "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ('ev_park_sess_perf_000', 'sess_perf_000', 100000, 'span', "
            "$1::jsonb, NULL, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            json.dumps({"event": "model_workflow_park", "run_id": "run_park_perf_000"}),
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
    return cast(dict[str, Any], result[0]["Plan"])


# ─── structural tests (primary, always-on) ───────────────────────────────────


@needs_docker
class TestNoCorrelatedSubplanOverEvents:
    """The central invariant: no sweep query may re-scan ``events`` inside
    a correlated SubPlan. That shape was the N+1 pathology behind #140."""

    async def test_candidate_rows_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        plan = await _explain(seeded_pool, CANDIDATE_ROWS_SQL.format(scope_clause=""))
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in find_sessions_needing_inference candidate query: "
            f"{len(found)} correlated subplan(s) over events. "
            f"CANDIDATE_ROWS_SQL should use session scalar columns, not event-log scans."
        )

    async def test_unreacted_rows_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        session_ids = [f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)]
        plan = await _explain(seeded_pool, UNREACTED_ROWS_SQL, session_ids)
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in _filter_incomplete_batches unreacted query: "
            f"{len(found)} correlated subplan(s) over events. Same CTE fix applies."
        )

    async def test_errored_sessions_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """``ERRORED_SESSIONS_SQL`` derives the parked-errored set the sweep
        subtracts on every pass (replacing the old ``status = 'errored'``
        column filter). Lock its two-MAX-CTE shape so it can't regress into a
        correlated subquery over ``events``."""
        plan = await _explain(seeded_pool, ERRORED_SESSIONS_SQL.format(scope_clause=""))
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in errored-session derivation: "
            f"{len(found)} correlated subplan(s) over events. This query must "
            f"use maintained scalar columns on sessions (migration 0066), not event-log scans."
        )

    async def test_span_start_rows_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """``GHOST_SPAN_START_SQL`` drives the two-branch ghost-recovery
        synthesis (#685).  Lock its single-pass shape so a future
        refactor that adds a correlated subquery (e.g., excluding
        already-confirmed-but-redispatched tcids) fails here instead
        of regressing sweep latency."""
        session_ids = [f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)]
        # Arbitrary tcid set — the planner cares about the predicate shape,
        # not the actual values.
        tcids = [f"tc_{i}" for i in range(10)]
        plan = await _explain(seeded_pool, GHOST_SPAN_START_SQL, session_ids, tcids)
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in find_and_repair_ghosts span-marker query: "
            f"{len(found)} correlated subplan(s) over events. See #685."
        )

    async def test_ghost_asst_is_not_n_plus_1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """``GHOST_ASST_SQL`` is the cross-session entry point of ghost
        repair, run unscoped (``scope_clause=""``) on every periodic sweep
        pass via ``find_and_repair_ghosts``.  Its bound — ``s.open_tool_call_count
        > 0``, a maintained scalar on ``sessions`` (migration 0066) — must stay
        a plain seq-scan + hash-join over that column, never regress into a
        correlated subquery over ``events`` (e.g. an ``EXISTS`` re-deriving the
        open-call set per row).  Such a rewrite would still return 0 rows on a
        fully-resolved fixture — passing the row-budget test — while silently
        reintroducing the N+1 scan over ``events`` on every sweep (#840)."""
        plan = await _explain(seeded_pool, GHOST_ASST_SQL.format(scope_clause=""))
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in find_and_repair_ghosts cross-session ghost scan: "
            f"{len(found)} correlated subplan(s) over events. GHOST_ASST_SQL must "
            f"stay bounded by the maintained sessions.open_tool_call_count scalar "
            f"(migration 0066), not an event-log subquery. See #840."
        )

    async def test_open_candidates_asst_is_not_n_plus_1(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        """The empty-unreacted/no-in-flight dispatch-narrowing branch runs
        ``OPEN_CANDIDATES_ASST_SQL`` on every periodic sweep that reaches it.
        Keep its ``open_tool_call_count`` gate a maintained-scalar join rather
        than a per-session event-log derivation."""
        session_ids = [
            *(f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)),
            *(f"sess_tc_{i:03d}" for i in range(_N_TC_SESSIONS)),
            "sess_tc_deep",
            "sess_tc_open",
        ]
        plan = await _explain(seeded_pool, OPEN_CANDIDATES_ASST_SQL, session_ids)
        found = find_subplans_over_events(plan)
        assert not found, (
            f"N+1 regression in _filter_incomplete_batches empty_no_inflight "
            f"dispatch narrowing: {len(found)} correlated subplan(s) over events. "
            f"OPEN_CANDIDATES_ASST_SQL must use the maintained "
            f"sessions.open_tool_call_count scalar, not an event-log subquery."
        )

    async def test_unharvested_park_scan_uses_index_not_seq_scan(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        """``find_unharvested_model_dispatch_parks`` runs cross-session
        (``scope_clause=""``) on every 30s periodic sweep. Before #1707 its
        ``kind='span' AND data->>'event'='model_workflow_park'`` predicate had no
        supporting index, so the outer scan and both ``NOT EXISTS`` anti-joins
        fell to ``Seq Scan on events`` — cost proportional to the whole event log,
        fleet-wide. Migration 0131 adds the park partial index plus the two
        harvest companions; assert the planner picks index scans and no node in
        the plan tree is a ``Seq Scan on events``.

        ANALYZE the fixture first so the planner has stats: the partial indexes
        are tiny (few or zero matching rows on the seeded fixture), so a
        cost-based planner prefers them over a full seq scan once it knows the
        table's true row count.
        """
        async with seeded_pool.acquire() as conn:
            await conn.execute("ANALYZE events")
        plan = await _explain(
            seeded_pool, UNHARVESTED_MODEL_DISPATCH_PARKS_SQL.format(scope_clause="")
        )
        found = find_seq_scans_over_events(plan)
        assert not found, (
            f"find_unharvested_model_dispatch_parks seq-scans events: "
            f"{len(found)} Seq Scan node(s) on events. The park/harvest partial "
            f"indexes (migration 0131) must serve this cross-session sweep. See #1707."
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
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {CANDIDATE_ROWS_SQL.format(scope_clause='')}"
            )
        if isinstance(result, str):
            result = json.loads(result)
        root = result[0]["Plan"]
        hits = _total_buffer_hits(root)
        # The fixture seeds a few thousand events, so a healthy linear-scan plan
        # costs a few thousand buffer hits. We allow generous headroom (the
        # budget is sized to the plan *shape*, not the exact event count, so it
        # won't go stale as the fixture grows); the pre-fix N+1 path burnt ~1.8M.
        assert hits < 50_000, f"candidate_rows uses {hits} buffer hits — N+1 regression?"

    async def test_open_candidates_asst_buffer_budget(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """Exercise the exact assistant fetch used by ``empty_no_inflight``
        against a mixed candidate set containing deep resolved history and one
        genuinely open call. Two invariants, both required:

        - **Cardinality (correctness)** — the scalar gate
          (``s.open_tool_call_count > 0``) must suppress every fully-resolved
          session, so the query returns rows ONLY for ``sess_tc_open``. Without
          this assertion the budget check is a HOLLOW fence: deleting the gate
          leaks all 500 assistant rows of the resolved ``sess_tc_deep`` bait
          into the result yet stays ~141x under the buffer budget, so the budget
          alone cannot ring on the exact regression (gate deletion) it exists to
          catch. Mirrors ``TestGhostAsstSweepBounded``'s exact-set fence for the
          structurally-identical ``GHOST_ASST_SQL``.
        - **Buffer budget (perf)** — buffer work stays linear, a backstop for an
          N+1/unbounded plan-choice regression even while the gate is intact.
        """
        session_ids = [
            *(f"sess_perf_{i:03d}" for i in range(_N_SESSIONS)),
            *(f"sess_tc_{i:03d}" for i in range(_N_TC_SESSIONS)),
            "sess_tc_deep",
            "sess_tc_open",
        ]
        async with seeded_pool.acquire() as conn:
            # Cardinality first: the scalar gate suppresses every fully-resolved
            # session, so only the genuinely-open one survives. With the gate
            # removed, the resolved/deep bait sessions leak into the result set
            # and this assertion fails — as it must.
            rows = await conn.fetch(OPEN_CANDIDATES_ASST_SQL, session_ids)
            returned = {r["session_id"] for r in rows}
            assert returned == {"sess_tc_open"}, (
                f"OPEN_CANDIDATES_ASST_SQL must return rows ONLY for sessions "
                f"with open_tool_call_count > 0; expected {{'sess_tc_open'}}, got "
                f"{returned}. Resolved/deep bait sessions leaking in means the "
                f"scalar gate was dropped."
            )

            result = await conn.fetchval(
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {OPEN_CANDIDATES_ASST_SQL}",
                session_ids,
            )
        if isinstance(result, str):
            result = json.loads(result)
        hits = _total_buffer_hits(result[0]["Plan"])
        assert hits < 50_000, (
            f"empty_no_inflight OPEN_CANDIDATES_ASST_SQL uses {hits} buffer hits "
            f"— unbounded/N+1 regression?"
        )

    async def test_list_sessions_status_derivation_budget(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        """The read-time ``status`` derivation (``_SESSION_STATUS_EXPR``) runs
        per returned row on ``list_sessions``/``get_session``. Keyset + LIMIT
        bound it to the page, so it must stay cheap — guard against an accidental
        O(events^2) plan (e.g. a cross-join in the correlated subqueries).

        Crucially this also exercises the *expensive* branch: the ``sess_tc_*``
        sessions are fully resolved + fully reacted, so the active expression
        can't short-circuit and runs its CROSS JOIN LATERAL over ``tool_calls``
        + NOT EXISTS anti-join over results to completion on every row. A
        quadratic re-scan there would blow this budget — so this assertion is
        the N+1 lock for ``_SESSION_ACTIVE_EXPR``'s tool_call branch."""
        list_sql = (
            f"SELECT sessions.*, ({_SESSION_STATUS_EXPR}) AS status, "
            "(SELECT e.created_at FROM events e WHERE e.session_id = sessions.id "
            "ORDER BY e.seq DESC LIMIT 1) AS last_event_at "
            "FROM sessions WHERE sessions.archived_at IS NULL "
            "AND sessions.account_id = $1 ORDER BY sessions.id DESC LIMIT 50"
        )
        async with seeded_pool.acquire() as conn:
            result = await conn.fetchval(
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {list_sql}", "acc_test_stub"
            )
        if isinstance(result, str):
            result = json.loads(result)
        hits = _total_buffer_hits(result[0]["Plan"])
        assert hits < 50_000, (
            f"list_sessions status derivation uses {hits} buffer hits — correlated-subquery blowup?"
        )


# ─── ghost-scan bound (#840) ─────────────────────────────────────────────────


@needs_docker
class TestGhostAsstSweepBounded:
    """#840: GHOST_ASST_SQL is bounded by sessions.open_tool_call_count > 0.

    This locks the bound in BOTH directions on a single fixture that mixes a
    genuinely-open session with deep/fully-resolved ones:

    - **Pass-through (positive)** — ``sess_tc_open`` has one unresolved tool_call
      (``open_tool_call_count = 1``), so the cross-session scan MUST return it.
      Catches an OVER-restrictive bound (e.g. ``AND 1=0`` or ``> 99999``) that
      would silently disable ghost detection while still passing a zero-row test.
    - **Flatness (negative/suppression)** — every other session is fully
      resolved (count = 0), including the DEEP 500-turn ``sess_tc_deep`` history.
      They contribute zero rows no matter how deep the resolved log grows.
      Catches an UNDER-restrictive bound (the line removed) that leaks resolved
      sessions and reintroduces the N+1 scan over ``events``.

    The exact-set assertion ``returned == {"sess_tc_open"}`` is the core fence;
    the two directional assertions above it document intent. The all-resolved
    zero-row test this replaced was vacuous: an over-restrictive mutation passed
    it AND the structural N+1 test, silently disabling ghost detection (#840)."""

    async def test_ghost_asst_scoped_to_open_call_sessions(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        async with seeded_pool.acquire() as conn:
            rows = await conn.fetch(GHOST_ASST_SQL.format(scope_clause=""))
        returned = {r["session_id"] for r in rows}

        # Pass-through: the one session with an unresolved call IS detected.
        # An over-restrictive bound (AND 1=0, > 99999) empties this and fails.
        assert "sess_tc_open" in returned, (
            f"GHOST_ASST_SQL dropped sess_tc_open (open_tool_call_count = 1) — "
            f"an over-restrictive bound disables ghost detection (#840). Got {returned}."
        )
        # Flatness: deep + resolved sessions never leak in, regardless of depth.
        assert "sess_tc_deep" not in returned, (
            f"GHOST_ASST_SQL leaked the DEEP fully-resolved sess_tc_deep — bound "
            f"is under-restrictive and rescans resolved history (#840). Got {returned}."
        )
        leaked_resolved = {
            sid for sid in returned if sid.startswith(("sess_tc_", "sess_perf_"))
        } - {"sess_tc_open"}
        assert not leaked_resolved, (
            f"GHOST_ASST_SQL leaked fully-resolved sessions {leaked_resolved} — "
            f"bound must suppress open_tool_call_count = 0 sessions (#840)."
        )
        # Exact scope: rows ONLY for open-call sessions, regardless of how deep
        # the resolved history grows. This single assertion is the core fence.
        assert returned == {"sess_tc_open"}, (
            f"GHOST_ASST_SQL must return rows ONLY for sessions with "
            f"open_tool_call_count > 0; expected {{'sess_tc_open'}}, got {returned} (#840)."
        )


@needs_docker
class TestBatchFilterBounded:
    """#1729: the batch filter's assistant/result fetches are BOUNDED to the
    batches referenced by the session's *unreacted* tool_call_ids — not the
    session's entire lifetime.

    The pre-#1729 filter ran two unbounded lifetime scans (every ``role='tool'``
    row + every assistant ``tool_calls`` payload, 126 MB observed) and decoded
    them on the worker event loop on every full sweep. The deep, fully-resolved
    ``sess_tc_deep`` (500 turns x 4 tool_calls) is the fixture that would blow up
    such a scan; here the bounded queries must return only the handful of rows
    tied to a referenced batch id, no matter how deep the resolved history is.

    Structural (row-set) assertions, not wall-clock — deterministic on any CI.
    """

    async def test_referenced_asst_scoped_to_containment_probe(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        # Probe for a single tool_call id belonging to ONE assistant turn deep in
        # sess_tc_deep. The containment-bounded query must return exactly that
        # one owning batch row, not the 500 assistant turns in the session.
        probe_tcid = "tc_sess_tc_deep_250_0"
        probes = [json.dumps([{"id": probe_tcid}])]
        async with seeded_pool.acquire() as conn:
            rows = await conn.fetch(REFERENCED_ASST_BATCH_SQL, "sess_tc_deep", probes)

        assert len(rows) == 1, (
            f"REFERENCED_ASST_BATCH_SQL must return ONLY the batch owning the "
            f"probed tcid, got {len(rows)} rows — an unbounded scan leaked the "
            f"deep resolved history (#1729)."
        )
        ids = {tcid for tcid in rows[0]["tool_call_ids"]}
        assert probe_tcid in ids
        # Payload-stripped: the row carries the id array, not the full ``data``.
        assert "data" not in dict(rows[0])

    async def test_batch_result_scoped_to_batch_ids(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        # Results are fetched ONLY for the specific batch ids, not every
        # ``role='tool'`` row the deep session ever produced (2000 of them).
        batch_ids = [f"tc_sess_tc_deep_250_{k}" for k in range(_N_TC_PER_ASST)]
        async with seeded_pool.acquire() as conn:
            rows = await conn.fetch(BATCH_RESULT_ROWS_SQL, "sess_tc_deep", batch_ids)

        returned = {r["tool_call_id"] for r in rows}
        assert returned == set(batch_ids), (
            f"BATCH_RESULT_ROWS_SQL must return results ONLY for the requested "
            f"batch ids; expected {set(batch_ids)}, got {returned} (#1729)."
        )


def _total_buffer_hits(plan_node: dict[str, Any]) -> int:
    total = int(plan_node.get("Shared Hit Blocks", 0) or 0)
    for child in plan_node.get("Plans", []):
        total += _total_buffer_hits(child)
    return total


# ─── fast-path admission gate (#1659) ────────────────────────────────────────


def _collect_nodes(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the plan tree into a list of nodes (pre-order)."""
    out = [plan_node]
    for child in plan_node.get("Plans", []):
        out.extend(_collect_nodes(child))
    return out


def _plan_node_shape(plan_node: dict[str, Any]) -> list[tuple[str, str, str]]:
    """A structural fingerprint of the plan: the ordered ``(Node Type, Relation
    Name, Index Name)`` triples across the tree. Deliberately excludes any
    cost/row *estimates* — the invariant is that the SHAPE is identical across
    N, not that the numbers match."""
    return [
        (
            str(n.get("Node Type", "")),
            str(n.get("Relation Name", "")),
            str(n.get("Index Name", "")),
        )
        for n in _collect_nodes(plan_node)
    ]


async def _seed_fast_path_session(pool: asyncpg.Pool[Any], session_id: str, n_events: int) -> None:
    """Seed one session at ``n_events`` user/assistant message events, then
    reconcile the migration-0066 scalar columns (the fast-path reads only these,
    never ``events``). Used by the plan-shape guard at N=10 and N=10 000 to prove
    the ``session_has_pending_work`` plan does not grow with event history."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_perf', 'perf', 'openrouter/x', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_perf', 'env_perf', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
            "VALUES ($1, 'agt_perf', 'env_perf', '/tmp/ws_' || $1, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            session_id,
        )
        rows: list[tuple[Any, ...]] = []
        for i in range(n_events):
            role = "user" if i % 2 == 0 else "assistant"
            data: dict[str, Any] = {"role": role, "content": f"m{i}"}
            if role == "assistant":
                data["reacting_to"] = i
            rows.append(
                (f"ev_{session_id}_{i}", session_id, i + 1, "message", json.dumps(data), role)
            )
        await conn.executemany(
            "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING",
            rows,
        )
        # Reconcile the maintained scalar columns (mirrors migration 0066).
        await conn.execute(
            """
            UPDATE sessions s
               SET last_event_seq = COALESCE(
                       (SELECT MAX(e.seq) FROM events e WHERE e.session_id = s.id), 0),
                   last_user_seq = COALESCE(
                       (SELECT MAX(e.seq) FROM events e WHERE e.session_id = s.id
                          AND e.kind = 'message' AND e.role = 'user'), 0),
                   last_stimulus_seq = COALESCE(
                       (SELECT MAX(e.seq) FROM events e WHERE e.session_id = s.id
                          AND e.kind = 'message' AND e.role <> 'assistant'), 0),
                   last_reacted_seq = COALESCE(
                       (SELECT MAX(COALESCE((e.data->>'reacting_to')::bigint, e.seq))
                          FROM events e WHERE e.session_id = s.id
                          AND e.kind = 'message' AND e.role = 'assistant'), 0)
             WHERE s.id = $1
            """,
            session_id,
        )
        await conn.execute("ANALYZE events")
        await conn.execute("ANALYZE sessions")


@needs_docker
class TestFastPathPlanShapeAsymptotic:
    """Guard 1 (issue #1659): the per-turn fast-path ``session_has_pending_work``
    is asymptotically O(1) in event count.

    Seed the SAME session at N=10 and N=10 000 events, ``EXPLAIN (FORMAT JSON)``
    the ``FAST_PATH_PENDING_WORK_SQL`` at both, and assert:

    - there is **no Seq Scan on ``events``** (nor any ``events`` scan at all) —
      the load-bearing #1659 regression guard;
    - the ``sessions`` row is reached (the fast path is a single-row PK-scoped
      lookup) — WITHOUT pinning a specific access method, since at tiny N the
      planner legitimately picks a Seq Scan over ``sessions_pkey`` and that
      choice varies with table size (not a stable signal);
    - the plan **node shape is identical across N** — so the plan does not grow
      with event history.

    This is a plan-SHAPE assertion (present-vs-absent + shape identity), never a
    wall-clock threshold — the 3.1↔4.7↔5.2s spread in the profile proves a
    wall-clock gate would flake.
    """

    async def test_pk_indexed_no_events_scan_and_shape_stable_across_n(
        self, aios_env: dict[str, str]
    ) -> None:
        from aios.db.pool import create_pool

        pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=2)
        try:
            shapes: dict[int, list[tuple[str, str, str]]] = {}
            for n in (10, 10_000):
                sid = f"sess_fastpath_n{n}"
                await _seed_fast_path_session(pool, sid, n)
                plan = await _explain(pool, FAST_PATH_PENDING_WORK_SQL, sid)
                nodes = _collect_nodes(plan)

                # No scan of ``events`` at any node — the fast path reads only the
                # maintained scalar columns on ``sessions`` (+ the tiny cancel-marker
                # table), never the event log.
                events_scans = [n2 for n2 in nodes if n2.get("Relation Name") == "events"]
                assert not events_scans, (
                    f"session_has_pending_work scans ``events`` at N={n} "
                    f"({[s.get('Node Type') for s in events_scans]}) — the fast path "
                    f"must be O(1) in event count, reading only sessions scalar columns."
                )
                # No Seq Scan on events specifically (the explicit #1659 assertion).
                seq_scans_events = [
                    n2
                    for n2 in nodes
                    if n2.get("Node Type") == "Seq Scan" and n2.get("Relation Name") == "events"
                ]
                assert not seq_scans_events, (
                    f"session_has_pending_work has a Seq Scan on ``events`` at N={n}."
                )
                # The ``sessions`` row is reached (the fast path is a single-row
                # PK-scoped lookup, ``WHERE id = $1``). We deliberately do NOT
                # assert a specific access method here: at tiny N the planner
                # legitimately prefers a Seq Scan over ``sessions_pkey`` because a
                # seq scan is cheaper for a handful of rows, and that choice varies
                # with table size — it is not a stable, load-bearing signal. The
                # durable #1659 invariant is "no ``events`` scan + O(1) in event
                # count" (asserted above and by the shape-identity check below).
                sessions_nodes = [n2 for n2 in nodes if n2.get("Relation Name") == "sessions"]
                assert sessions_nodes, (
                    f"session_has_pending_work does not touch ``sessions`` at N={n}; "
                    f"nodes: {[(x.get('Node Type'), x.get('Relation Name')) for x in nodes]}"
                )
                shapes[n] = _plan_node_shape(plan)

            assert shapes[10] == shapes[10_000], (
                "session_has_pending_work plan SHAPE changed between N=10 and "
                f"N=10 000 — it must not grow with event history.\n"
                f"N=10:    {shapes[10]}\nN=10000: {shapes[10_000]}"
            )
        finally:
            await pool.close()


class TestFastPathSpanCompositionInvariant:
    """Guard 2 (issue #1659): the per-turn entry guard calls the CHEAP fast-path
    ``session_has_pending_work`` (single PK lookup), NOT the heavy multi-CTE
    ``find_sessions_needing_inference`` — the full sweep is only reached on
    fall-through.

    This makes "the entry guard silently regains the heavy sweep" a RED test
    (caught here), not a latency mystery found months later. It is a pure
    call-graph assertion on ``_run_session_step_body`` — no DB, no docker.
    """

    async def test_entry_guard_uses_fast_path_then_falls_through(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        from aios.harness.loop import run_session_step

        fast_path = AsyncMock(return_value=False)
        full_sweep = AsyncMock(return_value=set())
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch("aios.harness.loop.session_has_pending_work", fast_path),
            patch("aios.harness.loop.find_sessions_needing_inference", full_sweep),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
        ):
            await run_session_step("sess_x", cause="message")

        # The cheap fast path IS on the hot path.
        fast_path.assert_awaited_once()
        # And when it early-outs (False), the heavy multi-CTE sweep is NOT run —
        # the entry guard has not silently regained the full sweep.
        full_sweep.assert_not_awaited()

    async def test_full_sweep_reached_only_on_fall_through(self) -> None:
        """The complementary half: when the fast path says "maybe work" (True),
        the entry guard DOES fall through to ``find_sessions_needing_inference``
        (so a wrong fast-path predicate can never miss a wake)."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        from aios.harness.loop import run_session_step

        fast_path = AsyncMock(return_value=True)
        full_sweep = AsyncMock(return_value=set())
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch("aios.harness.loop.session_has_pending_work", fast_path),
            patch("aios.harness.loop.find_sessions_needing_inference", full_sweep),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
        ):
            await run_session_step("sess_x", cause="message")

        fast_path.assert_awaited_once()
        full_sweep.assert_awaited_once()
