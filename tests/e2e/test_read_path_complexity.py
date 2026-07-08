"""CI-gated per-turn read-path COMPLEXITY harness (issue #1661, extended by
#1750 to the append / tool-result phase).

Why this file exists — the durable class-guard.
================================================
A per-turn hot path silently went O(session-size) (issue #1657, born in #1611
commit 0847e16d): an unbounded ``LAG(cumulative_tokens) OVER (ORDER BY seq)``
WindowAgg in ``_retained_class_mass``, called *before* the fits-in-window
short-circuit. On Ultron's 90,415-message session it detonated — ~5 s read, a
45 MB external-merge Disk spill — while CI + every other agent stayed green.
The sole detector was a human. The Chairman's verdict: "this is as bad as a
correctness issue." That class of regression cleared *every* gate the machine
had, because there was NO perf-complexity tier.

This file builds that tier. The oracle is asymptotic **SHAPE**, hardware-
independent — never milliseconds — so it gates CI red like a correctness bug
with zero flakiness. It follows the already-merged sweep-perf precedent
(``tests/e2e/test_sweep_perf.py`` #140 + ``tests/support.py``): seed a large
real-Postgres session in ONE ``INSERT ... SELECT generate_series``, ``ANALYZE``,
then ``EXPLAIN (FORMAT JSON)`` — *without* ``ANALYZE`` — and assert on plan
shape. It complements the #1657-specific ``test_context_read_perf.py`` by
generalizing to the whole hot-path READ class and adding the durable
registry-completeness gate that catches the *next* unregistered O(N) read.

Four oracles (Chairman directive: gate the first three like correctness):
  1. PRIMARY GATE (deterministic) — plan SHAPE. For every read in
     ``HOT_PATH_READS``, capture the SQL, ``EXPLAIN (FORMAT JSON)`` it, assert
     ``find_unbounded_events_scan_over_seq(plan) == []``.
  2. ROWS-RETURNED GATE (deterministic) — each hot read returns <= c*W rows to
     Python (exact count). Closes the SQL->Python O(N) escape hatch the plan
     can't see.
  3. REGISTRY-COMPLETENESS GATE (deterministic) — an import-time reflection
     scan of ``run_session_step``'s read phase asserts every DB-reading
     callable is in ``HOT_PATH_READS`` or carries ``# perf-exempt: <reason>``.
     An unregistered new read turns this RED.
  4. ADVISORY BACKSTOP (``@pytest.mark.perf``, NON-GATING) — scaling ratio
     ``t(N2)/t(N1) < 2.5``. Loudly advisory: shared-runner jitter can exceed
     2.5x for O(1), so it must NEVER be in required-checks. Catches complexity
     the planner can't (app-side loops, N+1).

Complexity-ONLY: this file makes ZERO output/behavioral/token-value assertions
(correctness lives in ``test_clone_window_regression.py`` + #1657's own tests).
No wall-clock or estimated-row-count appears in any GATING assertion; the only
timing is under the advisory ``perf`` mark.

── append / tool-result phase (issue #1750) ──────────────────────────────────

The read-phase registry above (``read_windowed_events`` +
``compute_step_prelude``) does not reach the APPEND path — the reads that run
on *every tool-result append*, not just every inference read. The #1 violation
in the #1733 epic, ``find_tool_result_event`` (``db/queries/events.py``), lived
there: a plain ``SELECT ... LIMIT 1`` filtering ``data->>'role' = 'tool'``
against the partial index ``events_tool_result_idx`` (predicated on the
BACKFILLED ``role`` COLUMN, migrations 0023/0097) — an index-predicate
mismatch, not an unbounded aggregate, so the *existing* oracle
(``find_unbounded_events_scan_over_seq``, which fires only under an
``_AGGREGATE_NODE_TYPES`` node) never saw it. The read planned as a bare
``Seq Scan on events``, keyed on ``session_id`` alone, on every tool-result
append under the session row lock. #1734 fixed the exemplar by re-predicating
onto the ``role`` column; this extension closes the class by (a) generalizing
the plan-shape oracle with a sibling detector
(``tests.support.find_predicate_mismatch_events_scan``), (b) registering the
append-phase reads, and (c) extending the registry-completeness AST gate to
the append/tool-result entry points (``services.append_tool_result`` and
``sweep.find_and_repair_ghosts``) so a FUTURE unregistered append-path read
turns the registry gate RED — the same anti-narrowing discipline #1661
established for the read phase.
"""

from __future__ import annotations

import ast
import inspect
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import asyncpg
import pytest

from aios.harness.sweep import (
    ALL_RESULT_ROWS_SQL,
    CONFIRMED_ROWS_SQL,
    GHOST_ASST_SQL,
    UNREACTED_ROWS_SQL,
)
from tests.conftest import needs_docker
from tests.support import (
    find_predicate_mismatch_events_scan,
    find_unbounded_events_scan_over_seq,
)

pytestmark = pytest.mark.docker

# ─── fixture: a large single-session slate ───────────────────────────────────
#
# One session, N message rows, each carrying the running counters
# ``append_event`` maintains. We seed with a single ``INSERT ... SELECT
# generate_series`` (window running-sums in-DB) — NOT row-by-row
# ``append_event`` — so 20k rows seed sub-second. Roles cycle through all four
# ``content_class`` branches (text / tool_use / tool_result / thinking) so the
# read path exercises every CASE branch it blends over.

_ACCOUNT_ID = "acc_test_stub"
_SESSION_ID = "sess_readpath_1661"
_SESSION_ID_SMALL = "sess_readpath_1661_small"

_N_LARGE = 20_000
_N_SMALL = 2_000

# Per-row token delta the seeded running sums use (mirrors append_event's
# neutral per-message estimate). ``window_tokens`` is expressed in these units.
_DELTA = 10


async def seed_large_session(
    pool: asyncpg.Pool[Any],
    account_id: str,
    n_messages: int,
    *,
    window_tokens: int,
    session_id: str = _SESSION_ID,
) -> None:
    """Seed one session with ``n_messages`` message events reproducing
    ``append_event``'s in-DB running sums.

    A SINGLE ``INSERT INTO events (...) SELECT ... FROM generate_series(1, :n)``
    with ``SUM(<per-row-token-est>) OVER (ORDER BY seq)`` windows reproduces the
    ``cumulative_tokens`` running total, the ``cumulative_messages`` running
    count of user/assistant rows, and the four per-class ``cumulative_*_mass``
    running sums — exactly the columns ``append_event`` builds incrementally.
    Row-by-row ``append_event`` would be far too slow (20k rows must seed
    sub-second); the window running-sums make the seeded slate internally
    consistent AND fast.

    Roles cycle user -> assistant -> assistant(tool_calls) -> tool ->
    assistant(thinking), so ALL FOUR ``content_class`` CASE branches (text /
    tool_use / tool_result / thinking) appear in the slate — the composition the
    windower blends over and the branches the pre-fix WindowAgg GROUPed BY.

    ``ANALYZE events; ANALYZE sessions`` runs after seeding (as the sweep-perf
    precedent does) so the planner picks its real plan for this slate — the plan
    the shape oracles then inspect.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_readpath', 'readpath', 'openrouter/x', $1) "
            "ON CONFLICT (id) DO NOTHING",
            account_id,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_readpath', 'env_readpath', $1) "
            "ON CONFLICT (id) DO NOTHING",
            account_id,
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, "
            "workspace_volume_path, account_id) "
            "VALUES ($1, 'agt_readpath', 'env_readpath', '/tmp/ws_readpath', $2) "
            "ON CONFLICT (id) DO NOTHING",
            session_id,
            account_id,
        )

        # ONE INSERT ... SELECT generate_series. ``g`` is the 1-based seq; the
        # role cycle drives both ``data`` (so the JSONB content_class CASE fires
        # on real shapes) and the per-class delta attribution. Every
        # cumulative_* column is a window running-sum over ``g`` — the shape
        # append_event builds incrementally, but computed set-wise in one pass.
        await conn.execute(
            """
            INSERT INTO events
                (id, session_id, seq, kind, data, created_at,
                 cumulative_tokens, cumulative_messages,
                 cumulative_text_mass, cumulative_tool_result_mass,
                 cumulative_thinking_mass, cumulative_tool_use_mass,
                 role, account_id)
            SELECT
                'ev_readpath_' || $4 || '_' || g,
                $1,
                g,
                'message',
                CASE (g % 5)
                    WHEN 0 THEN '{"role":"user","content":"u"}'::jsonb
                    WHEN 1 THEN '{"role":"assistant","content":"a"}'::jsonb
                    WHEN 2 THEN ('{"role":"assistant","content":null,'
                               || '"tool_calls":[{"id":"tc_' || g || '","type":"function",'
                               || '"function":{"name":"bash","arguments":"{}"}}]}')::jsonb
                    WHEN 3 THEN ('{"role":"tool","tool_call_id":"tc_'
                               || g || '","content":"ok"}')::jsonb
                    ELSE ('{"role":"assistant","content":"t",'
                         || '"reasoning_content":"because"}')::jsonb
                END,
                now(),
                -- cumulative_tokens: running SUM of a per-row delta. The
                -- per-row delta is cast to ``bigint`` because asyncpg binds
                -- ``$5`` as an untyped parameter and ``SUM(unknown)`` is
                -- ambiguous in Postgres (AmbiguousFunctionError); the explicit
                -- cast pins the SUM to the ``bigint`` overload.
                (SUM($5::bigint) OVER (ORDER BY g))::bigint,
                -- cumulative_messages: running COUNT of user/assistant rows
                -- (roles 0,1,2,4 are user/assistant; role 3 is tool).
                (SUM(CASE WHEN (g % 5) = 3 THEN 0 ELSE 1 END) OVER (ORDER BY g))::bigint,
                -- per-class running masses: each row's whole delta goes to its
                -- dominant class, matching _message_content_class.
                (SUM(CASE WHEN (g % 5) IN (0, 1) THEN $5::bigint ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 3 THEN $5::bigint ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 4 THEN $5::bigint ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 2 THEN $5::bigint ELSE 0 END) OVER (ORDER BY g))::bigint,
                CASE (g % 5)
                    WHEN 0 THEN 'user'
                    WHEN 3 THEN 'tool'
                    ELSE 'assistant'
                END,
                $2
            FROM generate_series(1, $3) AS g
            ON CONFLICT (id) DO NOTHING
            """,
            session_id,
            account_id,
            n_messages,
            session_id,
            _DELTA,
        )

        # Keep the sessions scalar honest (production maintains it in append).
        await conn.execute(
            "UPDATE sessions SET last_event_seq = $2 WHERE id = $1",
            session_id,
            n_messages,
        )

        # Fresh stats so the planner picks its real plan for this slate.
        await conn.execute("ANALYZE events")
        await conn.execute("ANALYZE sessions")

    # ``window_tokens`` is threaded through the module-level drop boundary the
    # hot reads scan against; recorded on the object for the reads below.
    _WINDOW_STATE["window_tokens"] = window_tokens


async def _seed_append_phase_state(pool: asyncpg.Pool[Any], session_id: str = _SESSION_ID) -> None:
    """Populate the append/tool-result-phase scalar/lifecycle state on top of
    :func:`seed_large_session` (issue #1750).

    :func:`seed_large_session` seeds the message slate (including plenty of
    ``role='tool'`` and ``role='assistant'``-with-``tool_calls`` rows so every
    partial index the append-phase reads target is populated and selective —
    see the module docstring's "no step is executed" rationale) but leaves
    the maintained ``sessions`` scalars at their post-seed defaults
    (``open_tool_call_count = 0``, ``last_reacted_seq = 0``). This gives the
    append-phase HOT_PATH_READS entries a REAL, non-trivial plan to EXPLAIN
    against — anti-vacuity for those reads (a plan against an empty/trivial
    relation would be a vacuous "passes because there's nothing to scan"
    verdict): a few open tool_calls (``open_tool_call_count > 0``, so
    ``ghost_asst_scan`` seeks a genuinely selective slate rather than a
    universally-false-filter empty scan), an unreacted tail (so
    ``unreacted_rows_scan`` returns real rows), and one confirmed-but-
    unresolved lifecycle event (so ``confirmed_rows_scan`` seeks a real row).
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET open_tool_call_count = 5, last_reacted_seq = $2 WHERE id = $1",
            session_id,
            # A handful of trailing seqs left unreacted — a genuinely small
            # tail, not the whole slate (the ``unreacted_rows_scan`` shape
            # this seeds is O(unreacted-tail), not O(session-size)).
            _N_LARGE - 10,
        )
        await conn.execute(
            "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ('ev_readpath_confirmed_lc', $1, $2, 'lifecycle', $3::jsonb, NULL, $4) "
            "ON CONFLICT (id) DO NOTHING",
            session_id,
            _N_LARGE + 1,
            json.dumps(
                {
                    "event": "tool_confirmed",
                    "result": "allow",
                    "tool_call_id": "tc_confirmed_unresolved",
                }
            ),
            _ACCOUNT_ID,
        )
        await conn.execute(
            "UPDATE sessions SET last_event_seq = $2 WHERE id = $1", session_id, _N_LARGE + 1
        )
        await conn.execute("ANALYZE events")
        await conn.execute("ANALYZE sessions")


# Small mutable box so the seed helper's ``window_tokens`` reaches the read
# thunks without a global rebind (keeps the reads pure functions of the pool).
_WINDOW_STATE: dict[str, int] = {"window_tokens": _N_LARGE * _DELTA // 2}


@pytest.fixture
async def seeded_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    from aios.db.pool import create_pool

    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=4)
    # Small retained window (the drop boundary sits mid-slate) so the omission
    # complement is a genuinely large omitted prefix — the shape the pre-fix
    # count(*) / LAG WindowAgg scanned linearly.
    await seed_large_session(pool, _ACCOUNT_ID, _N_LARGE, window_tokens=_N_LARGE * _DELTA // 2)
    # Append/tool-result-phase state on top of the same slate (#1750).
    await _seed_append_phase_state(pool)
    try:
        yield pool
    finally:
        await pool.close()


# ─── the hot-path read registry (the gate-3 source of truth) ─────────────────
#
# Each HotRead names a DB read on ``run_session_step``'s read phase, its
# declared asymptotic complexity, and an ``invoke`` thunk that issues the exact
# SQL the read runs against the seeded pool. The three gates key off this list;
# the registry-completeness gate asserts the list COVERS the read phase.


@dataclass(frozen=True)
class HotRead:
    """One DB read on the per-turn read path.

    ``name`` matches the callable in the read phase it stands for (or an inline
    fetch labelled by purpose). ``declared_complexity`` is documentation +
    the ``max_rows`` budget the rows-returned gate enforces:

    * ``"O(1)"``    — an index seek; must return <= ``_O1_ROW_CEIL`` rows.
    * ``"O(W)"``    — bounded by the retained window; <= ``c * W`` rows.

    ``invoke`` takes the pool and returns ``(sql, args)`` for EXPLAIN, plus runs
    the read for the rows-returned + advisory gates via :meth:`fetch`.
    """

    name: str
    declared_complexity: str
    sql: str
    args: Callable[[], tuple[Any, ...]]
    max_rows: int


# Row ceilings. O(1) seeks return a single row (LIMIT 1); we allow a tiny slack
# for multi-row class fetches. O(W) reads are bounded by the retained window —
# with a 20k-row slate and a mid-slate drop, the retained tail is a few
# thousand rows, so a generous window ceiling still fails hard on an O(N) read
# that would return ~20k.
_O1_ROW_CEIL = 8
# The retained window can hold up to ~half the slate here (drop == half); the
# gate fails only on a read returning the WHOLE slate, so the ceiling is set
# below N but above the largest legitimate window read.
_OW_ROW_CEIL = _N_LARGE // 2 + 16


def _drop() -> int:
    return _WINDOW_STATE["window_tokens"]


# The exact SQL each hot read issues (mirrors db/queries/events.py post-#1657).
_SQL_LATEST_CUMULATIVE = (
    "SELECT cumulative_tokens FROM events "
    "WHERE session_id = $1 AND kind = 'message' "
    "AND cumulative_tokens IS NOT NULL "
    "ORDER BY seq DESC LIMIT 1"
)

_SQL_RETAINED_CLASS_MASS = (
    "SELECT cumulative_text_mass, cumulative_tool_result_mass, "
    "       cumulative_thinking_mass, cumulative_tool_use_mass "
    "FROM events "
    "WHERE session_id = $1 AND account_id = $2 "
    "AND kind = 'message' AND cumulative_tokens IS NOT NULL "
    "ORDER BY seq DESC LIMIT 1"
)

_SQL_RETAINED_WINDOW = (
    "SELECT id, session_id, seq, kind, data, created_at, role "
    "FROM events "
    "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
    "AND cumulative_tokens > $3 "
    "ORDER BY seq ASC"
)

_SQL_OMISSION_BOUNDARY = (
    "SELECT cumulative_messages, created_at FROM events "
    "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
    "AND cumulative_tokens <= $3 "
    "ORDER BY cumulative_tokens DESC LIMIT 1"
)

_SQL_BEGAN_AT = (
    "SELECT created_at FROM events "
    "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
    "ORDER BY seq ASC LIMIT 1"
)

# ─── append / tool-result phase reads (issue #1750) ───────────────────────────
#
# The exact production SQL each append-phase read issues, mirroring the
# read-phase constants above. ``_SQL_FIND_TOOL_RESULT_EVENT`` is the #1 #1733
# exemplar (``db/queries/events.py:find_tool_result_event``, post-#1734 form —
# filters the normalized ``role`` column, served by ``events_tool_result_idx``).
# The ghost/confirm sweep reads (``sweep.py``) are registered verbatim from
# their module constants below (imported, not re-typed, so a change to the
# production SQL is caught by drift rather than silently diverging from what
# this harness EXPLAINs).
_SQL_FIND_TOOL_RESULT_EVENT = (
    "SELECT * FROM events "
    "WHERE session_id = $1 "
    "  AND account_id = $2 "
    "  AND kind = 'message' "
    "  AND role = 'tool' "
    "  AND data->>'tool_call_id' = $3 "
    "LIMIT 1"
)

# The pre-#1734 predicate-mismatch form of the SAME query — the shape the
# not-vacuous probe (``TestPrimaryGateIsNotVacuous``, extended below) proves
# the generalized oracle catches.
_SQL_FIND_TOOL_RESULT_EVENT_PRE_1734 = (
    "SELECT * FROM events "
    "WHERE session_id = $1 "
    "  AND account_id = $2 "
    "  AND kind = 'message' "
    "  AND data->>'role' = 'tool' "
    "  AND data->>'tool_call_id' = $3 "
    "LIMIT 1"
)

# The ghost/confirm sweep reads, sourced verbatim from ``sweep.py``'s module
# constants (not re-typed) so drift in the production SQL is caught rather
# than silently diverging from what this harness EXPLAINs. Each is scoped to
# a single session via the SAME ``.format(scope_clause=...)`` composition
# ``find_and_repair_ghosts``/``find_sessions_needing_inference`` use for the
# per-step scoped call (``session_id=...``) — the shape this harness exercises
# is the exact production text, just single-session-parameterized. The
# ``scope_clause`` text (alias + placeholder) matches the production call
# sites exactly: ``"AND e.session_id = $1"`` for the ghost scan
# (``find_and_repair_ghosts``), ``"AND s.id = $1"`` for the confirmed-rows
# scan (``find_sessions_needing_inference``, which JOINs ``sessions AS s``).
_SQL_GHOST_ASST_SCOPED = GHOST_ASST_SQL.format(scope_clause="AND e.session_id = $1")
_SQL_ALL_RESULT_ROWS = ALL_RESULT_ROWS_SQL
_SQL_UNREACTED_ROWS = UNREACTED_ROWS_SQL
_SQL_CONFIRMED_ROWS_SCOPED = CONFIRMED_ROWS_SQL.format(scope_clause="AND s.id = $1", age_param="$2")


HOT_PATH_READS: list[HotRead] = [
    HotRead(
        name="_latest_cumulative_tokens",
        declared_complexity="O(1)",
        sql=_SQL_LATEST_CUMULATIVE,
        args=lambda: (_SESSION_ID,),
        max_rows=_O1_ROW_CEIL,
    ),
    HotRead(
        name="_retained_class_mass",
        declared_complexity="O(1)",
        sql=_SQL_RETAINED_CLASS_MASS,
        args=lambda: (_SESSION_ID, _ACCOUNT_ID),
        max_rows=_O1_ROW_CEIL,
    ),
    HotRead(
        name="read_windowed_context_events",
        declared_complexity="O(W)",
        sql=_SQL_RETAINED_WINDOW,
        args=lambda: (_SESSION_ID, _ACCOUNT_ID, _drop()),
        max_rows=_OW_ROW_CEIL,
    ),
    HotRead(
        name="omission_boundary_seek",
        declared_complexity="O(1)",
        sql=_SQL_OMISSION_BOUNDARY,
        args=lambda: (_SESSION_ID, _ACCOUNT_ID, _drop()),
        max_rows=_O1_ROW_CEIL,
    ),
    HotRead(
        name="began_at_seek",
        declared_complexity="O(1)",
        sql=_SQL_BEGAN_AT,
        args=lambda: (_SESSION_ID, _ACCOUNT_ID),
        max_rows=_O1_ROW_CEIL,
    ),
    # ─── append / tool-result phase (issue #1750) ─────────────────────────
    HotRead(
        name="find_tool_result_event",
        declared_complexity="O(1)",
        sql=_SQL_FIND_TOOL_RESULT_EVENT,
        args=lambda: (_SESSION_ID, _ACCOUNT_ID, "tc_no_such_call"),
        max_rows=_O1_ROW_CEIL,
    ),
    HotRead(
        name="ghost_asst_scan",
        declared_complexity="O(open_tool_call_count)",
        sql=_SQL_GHOST_ASST_SCOPED,
        args=lambda: (_SESSION_ID,),
        max_rows=_OW_ROW_CEIL,
    ),
    HotRead(
        name="all_result_rows_scan",
        declared_complexity="O(open_tool_call_count)",
        sql=_SQL_ALL_RESULT_ROWS,
        args=lambda: ([_SESSION_ID],),
        max_rows=_OW_ROW_CEIL,
    ),
    HotRead(
        name="unreacted_rows_scan",
        declared_complexity="O(unreacted-tail)",
        sql=_SQL_UNREACTED_ROWS,
        args=lambda: ([_SESSION_ID],),
        max_rows=_OW_ROW_CEIL,
    ),
    HotRead(
        name="confirmed_rows_scan",
        declared_complexity="O(confirmed-allow)",
        sql=_SQL_CONFIRMED_ROWS_SCOPED,
        args=lambda: (_SESSION_ID, 3600),
        max_rows=_O1_ROW_CEIL,
    ),
]


async def _explain(pool: asyncpg.Pool[Any], sql: str, *args: Any) -> dict[str, Any]:
    """``EXPLAIN (FORMAT JSON)`` — NO ANALYZE — return the root Plan node."""
    async with pool.acquire() as conn:
        result = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *args)
    if isinstance(result, str):
        result = json.loads(result)
    return cast(dict[str, Any], result[0]["Plan"])


# ─── GATE 1: PRIMARY plan-shape (deterministic, GATING) ──────────────────────


@needs_docker
class TestPrimaryPlanShapeGate:
    """For every hot read, ``EXPLAIN (FORMAT JSON)`` (no ANALYZE) the SQL and
    assert NO aggregate over ``events`` is unbounded by a
    ``cumulative_tokens``/``seq`` lower bound, AND no ``events`` Seq Scan
    carries the JSONB-over-column index-predicate-mismatch smell (issue
    #1750). Deterministic shape verdict — RED on the #1611/#1657
    O(session-size) WindowAgg or the #1734 index-predicate mismatch, GREEN on
    the fixed forms."""

    @pytest.mark.parametrize("read", HOT_PATH_READS, ids=lambda r: r.name)
    async def test_no_unbounded_events_scan_over_seq(
        self, seeded_pool: asyncpg.Pool[Any], read: HotRead
    ) -> None:
        plan = await _explain(seeded_pool, read.sql, *read.args())
        offenders = find_unbounded_events_scan_over_seq(plan)
        assert not offenders, (
            f"read-path complexity regression in {read.name!r} "
            f"(declared {read.declared_complexity}): {len(offenders)} unbounded "
            f"aggregate(s) scan ``events`` keyed on session_id alone — the "
            f"O(session-size) class that detonated Ultron (#1661/#1657). A hot "
            f"read must range-scan on cumulative_tokens/seq, not the whole slate."
        )

    @pytest.mark.parametrize("read", HOT_PATH_READS, ids=lambda r: r.name)
    async def test_no_predicate_mismatch_events_scan(
        self, seeded_pool: asyncpg.Pool[Any], read: HotRead
    ) -> None:
        """No registered read's plan carries an ``events`` Seq Scan whose
        Filter is a JSONB-over-column index-predicate mismatch (issue #1750,
        the #1 #1733 violation: ``find_tool_result_event`` pre-#1734)."""
        plan = await _explain(seeded_pool, read.sql, *read.args())
        offenders = find_predicate_mismatch_events_scan(plan)
        assert not offenders, (
            f"read-path index-predicate-mismatch regression in {read.name!r} "
            f"(declared {read.declared_complexity}): {len(offenders)} Seq "
            f"Scan(s) on events carry a data->>'role'/'tool_call_id' equality "
            f"a partial index exists to serve via the normalized column "
            f"(events_tool_result_idx et al., migrations 0011/0023/0065/0097) "
            f"— the #1734 defect class (#1750)."
        )


# ─── GATE 1b: guard-is-not-vacuous (the #1611 shape MUST be caught) ──────────
#
# The pre-fix ``_retained_class_mass`` SQL, verbatim — an unbounded
# ``LAG(cumulative_tokens) OVER (ORDER BY seq)`` WindowAgg GROUPing the whole
# slate by content-class. The acceptance criterion: the PRIMARY oracle must be
# RED against the ``0847e16d..(#1657^)`` regression range. We pin that here by
# EXPLAINing the exact removed SQL on the same seeded slate and asserting the
# detector fires — so the guard can never silently go vacuous.

_PREFIX_1611_LAG_SQL = """
WITH deltas AS (
    SELECT
        CASE
            WHEN data->>'role' = 'tool' THEN 'tool_result'
            WHEN data->>'role' = 'assistant'
                 AND (data ? 'tool_calls')
                 AND jsonb_typeof(data->'tool_calls') = 'array'
                 AND jsonb_array_length(data->'tool_calls') > 0
                THEN 'tool_use'
            WHEN data->>'role' = 'assistant'
                 AND ((data ? 'reasoning_content') OR (data ? 'thinking_blocks'))
                THEN 'thinking'
            ELSE 'text'
        END AS cls,
        cumulative_tokens
        - COALESCE(LAG(cumulative_tokens) OVER (ORDER BY seq), 0) AS delta
    FROM events
    WHERE session_id = $1
      AND account_id = $2
      AND kind = 'message'
      AND cumulative_tokens IS NOT NULL
)
SELECT cls, SUM(delta)::float AS mass
FROM deltas
GROUP BY cls
"""

# The #738-omission ``count(*)`` shape: role-filtered count over the whole
# omitted prefix, no ``cumulative_messages`` short-circuit. Also an
# O(session-size) aggregate the primary oracle must flag RED.
_OMISSION_738_COUNT_SQL = (
    "SELECT count(*) FILTER (WHERE role IN ('user', 'assistant')) "
    "FROM events "
    "WHERE session_id = $1 AND account_id = $2 AND kind = 'message'"
)


@needs_docker
class TestPrimaryGateIsNotVacuous:
    """The removed pre-fix O(N) shapes MUST trip the primary oracle on the same
    seeded slate. Without this, a refactor blinding
    ``find_unbounded_events_scan_over_seq`` would let an O(N) read pass GREEN."""

    async def test_1611_lag_windowagg_is_red(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        plan = await _explain(seeded_pool, _PREFIX_1611_LAG_SQL, _SESSION_ID, _ACCOUNT_ID)
        offenders = find_unbounded_events_scan_over_seq(plan)
        assert offenders, (
            "the #1611 pre-fix LAG(cumulative_tokens) OVER (ORDER BY seq) WindowAgg "
            "did NOT trip find_unbounded_events_scan_over_seq — the primary gate is "
            "vacuous. It must catch the exact O(session-size) shape #1657 removed "
            "(acceptance: RED against 0847e16d..(#1657^))."
        )

    async def test_738_omission_count_is_red(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        plan = await _explain(seeded_pool, _OMISSION_738_COUNT_SQL, _SESSION_ID, _ACCOUNT_ID)
        offenders = find_unbounded_events_scan_over_seq(plan)
        assert offenders, (
            "the #738 omission count(*) full-slate aggregate did NOT trip the primary "
            "oracle — the gate is vacuous against the omission-count regression shape "
            "(acceptance: RED against the #738 omission count(*) shape)."
        )

    async def test_1734_predicate_mismatch_is_red(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """Fix §4 not-vacuous probe: the pre-#1734 ``find_tool_result_event``
        SQL (``data->>'role' = 'tool'``) — which the partial index
        ``events_tool_result_idx`` (predicated on the ``role`` COLUMN) cannot
        serve — MUST trip the generalized oracle on the SAME seeded slate.
        Because this is ``EXPLAIN`` (no ``ANALYZE``), the ``LIMIT 1``
        short-circuit is irrelevant: the Seq Scan node is present regardless
        of where the matching row sits in the slate — plan shape is robust
        where an executed buffer-count probe would be fragile."""
        plan = await _explain(
            seeded_pool,
            _SQL_FIND_TOOL_RESULT_EVENT_PRE_1734,
            _SESSION_ID,
            _ACCOUNT_ID,
            "tc_3",
        )
        offenders = find_predicate_mismatch_events_scan(plan)
        assert offenders, (
            "the pre-#1734 data->>'role' predicate-mismatch SQL did NOT trip "
            "find_predicate_mismatch_events_scan — the generalized oracle is "
            "vacuous. It must catch the exact index-predicate-mismatch shape "
            "#1734 fixed (acceptance: RED against the pre-#1734 "
            "find_tool_result_event SQL)."
        )

    async def test_1734_fixed_form_is_green_on_same_slate(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        """The other half of the not-vacuous proof: the post-#1734 ``role``
        column form, EXPLAINed on the SAME single-session seeded slate, does
        NOT trip the oracle. Together with the previous test this proves the
        oracle distinguishes the defect from its fix on one slate,
        single-session and all (Lens 0 finding 2's concern — a single-session
        corpus inverting the planner's index-vs-seqscan choice — does not
        apply here: the fixed query still uses the selective partial index on
        this exact slate, while the broken query seq-scans)."""
        plan = await _explain(
            seeded_pool,
            _SQL_FIND_TOOL_RESULT_EVENT,
            _SESSION_ID,
            _ACCOUNT_ID,
            "tc_3",
        )
        offenders = find_predicate_mismatch_events_scan(plan)
        assert not offenders, (
            "the post-#1734 role-column find_tool_result_event SQL tripped "
            "find_predicate_mismatch_events_scan on the seeded slate — false "
            "positive; the fixed form must plan via the partial index, not a "
            "flagged Seq Scan."
        )


# ─── GATE 2: ROWS-RETURNED (deterministic, GATING) ───────────────────────────


@needs_docker
class TestRowsReturnedGate:
    """Each hot read returns <= its declared row budget to Python. Closes the
    SQL->Python O(N) escape hatch a plan-shape check can't see (a read whose
    plan is fine but that streams the whole slate into the app)."""

    @pytest.mark.parametrize("read", HOT_PATH_READS, ids=lambda r: r.name)
    async def test_returns_bounded_rows(
        self, seeded_pool: asyncpg.Pool[Any], read: HotRead
    ) -> None:
        async with seeded_pool.acquire() as conn:
            rows = await conn.fetch(read.sql, *read.args())
        n = len(rows)
        assert n <= read.max_rows, (
            f"read-path O(N)-to-Python regression in {read.name!r} "
            f"(declared {read.declared_complexity}): returned {n} rows > budget "
            f"{read.max_rows} on an N={_N_LARGE} slate. A hot read must not stream "
            f"the whole session into the app (#1661)."
        )


# ─── GATE 3: REGISTRY-COMPLETENESS (deterministic, GATING) ───────────────────
#
# The durable class-guard: an import-time reflection scan of the read phase
# AND (issue #1750) the append/tool-result phase. The read phase is
# ``read_windowed_events`` (the windowed slate read) and
# ``compute_step_prelude`` (the events-independent prelude) — both awaited by
# ``run_session_step`` before inference. The append phase is
# ``services.append_tool_result`` (every custom/builtin/MCP tool-result
# intake) and ``sweep.find_and_repair_ghosts`` (the cross-session ghost-repair
# scan) — the entry points the #1 #1733 violation (``find_tool_result_event``)
# lived on and the existing read-phase registry does not reach. We AST-walk
# each, collect the callables it invokes that read the DB (take a
# ``conn``/issue ``fetch*``), and assert every one is either registered in
# ``HOT_PATH_READS`` or carries an inline ``# perf-exempt: <reason>`` marker in
# the entry's source. A NEW, unregistered DB read on either phase turns this
# RED — the exact silent-O(N)-or-index-predicate-mismatch entry vector #1611
# and #1734 each exploited.

# The read-phase + append-phase entry points, by (module, qualname). Kept as
# strings so the scan is import-time-cheap and does not drag the whole harness
# into the test.
_READ_PHASE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("aios.db.queries.events", "read_windowed_events"),
    ("aios.harness.step_context", "compute_step_prelude"),
    # Append / tool-result phase (#1750): the entry points reached by every
    # tool-result intake and by cross-session ghost repair.
    ("aios.services.sessions", "append_tool_result"),
    ("aios.harness.sweep", "find_and_repair_ghosts"),
)

# Names that ARE registered hot reads (matched against the callables the read
# phase invokes). Kept in sync with ``HOT_PATH_READS`` by construction below.
_REGISTERED_NAMES = frozenset(r.name for r in HOT_PATH_READS) | {
    # HotRead names are logical; map the read-phase callable identifiers they
    # cover. The inline ``conn.fetchrow``/``conn.fetchval`` reads for the
    # omission boundary / began_at are the ``*_seek`` HotReads above.
    "omission_boundary_seek",
    "began_at_seek",
    # The prelude's obligations + connection-tools reads are hot-path DB reads
    # but bounded by their own indexes (anti-join / per-session tool list), not
    # the session slate; they are declared here as covered O(1)/O(bindings).
    "get_open_obligations",
    "list_tools_for_session",
    "model_token_class_ratios",
    "read_windowed_context_events",
    # Append phase (#1750): ``find_tool_result_event`` is a named callable the
    # AST scan resolves directly (``db.queries.events.find_tool_result_event``,
    # the #1 #1733 exemplar). The sweep's ghost/confirm reads
    # (GHOST_ASST_SQL / ALL_RESULT_ROWS_SQL / UNREACTED_ROWS_SQL /
    # CONFIRMED_ROWS_SQL) are issued via bare ``conn.fetch`` calls inside
    # ``find_and_repair_ghosts`` — already covered by ``_INLINE_READ_COVER``
    # below — and are registered here as logical names purely so the
    # PLAN-SHAPE + ROWS-RETURNED gates (which key off ``HOT_PATH_READS``, not
    # this set) cover their exact production SQL.
    "find_tool_result_event",
    "ghost_asst_scan",
    "all_result_rows_scan",
    "unreacted_rows_scan",
    "confirmed_rows_scan",
}

# asyncpg connection read methods — a call to one of these is a DB read.
_CONN_READ_METHODS = frozenset({"fetch", "fetchrow", "fetchval", "cursor"})


@dataclass
class _ReadPhaseScan:
    """Result of walking a read-phase entry function's AST."""

    entry: str
    db_reading_calls: set[str] = field(default_factory=set)
    exempt_reasons: dict[str, str] = field(default_factory=dict)


def _source_and_tree(module_name: str, qualname: str) -> tuple[str, ast.AST]:
    import importlib

    module = importlib.import_module(module_name)
    fn = getattr(module, qualname)
    src = inspect.getsource(fn)
    return src, ast.parse(src)


def _perf_exempt_markers(src: str) -> set[str]:
    """Collect ``# perf-exempt: <reason>`` markers in a source block.

    A read the reviewer has deliberately deemed non-scaling can be excused with
    an inline ``# perf-exempt: <reason>`` comment; we record the callable name
    on the SAME or the PRECEDING source line so the reflection scan can honor
    it. Presence of the reason text is what matters (documents intent).
    """
    exempt: set[str] = set()
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "# perf-exempt:" in line:
            # The exemption applies to the read call on this line, if any; the
            # scan keys off the callee identifier, so record the whole line's
            # identifiers conservatively (the gate only needs the marker to
            # exist near a would-be-flagged read).
            for ident in _identifiers_in(line):
                exempt.add(ident)
            # Also cover a marker placed on the line ABOVE the read.
            if i + 1 < len(lines):
                for ident in _identifiers_in(lines[i + 1]):
                    exempt.add(ident)
    return exempt


def _identifiers_in(line: str) -> set[str]:
    try:
        node = ast.parse(line.strip())
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def _scan_read_phase_entry(module_name: str, qualname: str) -> _ReadPhaseScan:
    """AST-walk one read-phase entry; return the DB-reading callables it
    invokes plus any ``# perf-exempt`` identifiers in its source."""
    src, tree = _source_and_tree(module_name, qualname)
    scan = _ReadPhaseScan(entry=f"{module_name}.{qualname}")
    scan.exempt_reasons = {ident: "perf-exempt" for ident in _perf_exempt_markers(src)}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee: str | None = None
        if isinstance(func, ast.Attribute):
            # e.g. conn.fetchrow(...) / queries.model_token_class_ratios(...).
            # An inline DB read (``conn.fetch*``) is attributed to the enclosing
            # purpose via its call-site (the harness labels these ``*_seek``);
            # the gate matches those against the *_seek / boundary HotReads by
            # the enclosing-read convention.
            callee = f"conn.{func.attr}" if func.attr in _CONN_READ_METHODS else func.attr
        elif isinstance(func, ast.Name):
            callee = func.id
        if callee is None:
            continue
        # Keep only callables that plausibly read the DB: either a conn.* read
        # method, or a known query-layer callable (heuristic: name is a
        # registered hot read, a *_events / *_mass / *_ratios / obligations /
        # tools query, or a private _latest_/_retained_ query helper).
        if _looks_like_db_read(callee):
            scan.db_reading_calls.add(callee)
    return scan


def _looks_like_db_read(callee: str) -> bool:
    if callee.startswith("conn."):
        return True
    bare = callee.split(".")[-1]
    if bare in _REGISTERED_NAMES:
        return True
    read_signatures = (
        "read_",
        "get_open_obligations",
        "list_tools_for_session",
        "model_token_class_ratios",
        "_latest_cumulative_tokens",
        "_retained_class_mass",
    )
    return any(bare == sig or bare.startswith(sig) for sig in read_signatures)


# The inline ``conn.fetchrow``/``conn.fetchval`` reads in ``read_windowed_events``
# are the omission-boundary seek, the pre-backfill count fallback, and the
# began_at seek. The count fallback is a rolling-deploy-only branch bounded by
# the ``cumulative_tokens <= drop`` index cond; it is covered by the
# ``omission_boundary_seek`` HotRead's shape assertion (same bounded prefix) and
# labelled perf-exempt in events.py. We map the raw ``conn.*`` method names to
# "covered" so the registry gate treats inline reads as registered when the
# enclosing purpose is in the registry.
_INLINE_READ_COVER = frozenset({"conn.fetch", "conn.fetchrow", "conn.fetchval", "conn.cursor"})


@needs_docker
class TestRegistryCompletenessGate:
    """Every DB read on ``run_session_step``'s read phase is registered in
    ``HOT_PATH_READS`` or carries a ``# perf-exempt: <reason>`` marker. A new,
    unregistered read -> RED. This is the durable class-guard: it forecloses
    the silent-O(N)-entry vector #1611 exploited (a fresh hot read that no
    existing gate looked at)."""

    def test_read_phase_reads_are_all_registered(self) -> None:
        unregistered: dict[str, set[str]] = {}
        for module_name, qualname in _READ_PHASE_ENTRIES:
            scan = _scan_read_phase_entry(module_name, qualname)
            missing: set[str] = set()
            for callee in scan.db_reading_calls:
                bare = callee.split(".")[-1]
                if callee in _INLINE_READ_COVER or callee in _REGISTERED_NAMES:
                    continue
                if bare in _REGISTERED_NAMES:
                    continue
                if callee in scan.exempt_reasons or bare in scan.exempt_reasons:
                    continue
                missing.add(callee)
            if missing:
                unregistered[scan.entry] = missing
        assert not unregistered, (
            "unregistered DB read(s) on the per-turn read phase: "
            f"{unregistered}. Every hot-path read must be listed in "
            "HOT_PATH_READS (so the shape + rows-returned gates cover it) OR "
            "carry an inline ``# perf-exempt: <reason>`` marker. An unregistered "
            "new read is exactly the silent-O(N) entry vector #1611 exploited "
            "(#1661)."
        )

    def test_registry_covers_the_named_read_phase_functions(self) -> None:
        """Sanity: the four read/append-phase entry points import + AST-parse,
        and the registry names at least the windowed-events + prelude reads
        the issue enumerates (``read_windowed_events``, ``compute_step_prelude``
        coverage) plus the append-phase entry points (#1750)."""
        scanned = {
            entry
            for module_name, qualname in _READ_PHASE_ENTRIES
            for entry in [_scan_read_phase_entry(module_name, qualname).entry]
        }
        assert scanned == {
            "aios.db.queries.events.read_windowed_events",
            "aios.harness.step_context.compute_step_prelude",
            "aios.services.sessions.append_tool_result",
            "aios.harness.sweep.find_and_repair_ghosts",
        }
        # The registry must at minimum cover the windowed read's own sub-reads.
        assert {"read_windowed_context_events", "_retained_class_mass"} <= _REGISTERED_NAMES
        # ... and the append-phase #1 #1733 exemplar (#1750).
        assert "find_tool_result_event" in _REGISTERED_NAMES


# ─── GATE 4: ADVISORY scaling backstop (perf mark, NON-GATING) ───────────────
#
# Times ONLY the read call at N1=2,000 and N2=20,000, min of M=5, pool warmed
# first, and asserts t(N2)/t(N1) < 2.5. This is EXPLICITLY advisory — shared-
# runner jitter can exceed 2.5x even for O(1) — so it carries the ``perf`` mark
# and MUST NEVER be in required-checks. It exists to catch complexity the
# planner cannot see (app-side loops, N+1 fan-out) that the deterministic gates
# above would miss.

_M_REPEATS = 5


@pytest.fixture
async def two_scale_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    from aios.db.pool import create_pool

    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=4)
    await seed_large_session(
        pool,
        _ACCOUNT_ID,
        _N_SMALL,
        window_tokens=_N_SMALL * _DELTA // 2,
        session_id=_SESSION_ID_SMALL,
    )
    await seed_large_session(
        pool,
        _ACCOUNT_ID,
        _N_LARGE,
        window_tokens=_N_LARGE * _DELTA // 2,
        session_id=_SESSION_ID,
    )
    try:
        yield pool
    finally:
        await pool.close()


async def _time_read(
    pool: asyncpg.Pool[Any], sql: str, args: tuple[Any, ...], *, repeats: int
) -> float:
    async with pool.acquire() as conn:
        # Warm the pool + plan cache first (result discarded).
        await conn.fetch(sql, *args)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            await conn.fetch(sql, *args)
            best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.perf
@needs_docker
class TestAdvisoryScalingBackstop:
    """NON-GATING scaling-ratio backstop. Loudly advisory: never a required
    check. Catches app-side / N+1 complexity the plan-shape oracle can't."""

    async def test_retained_window_read_scales_sublinearly(
        self, two_scale_pool: asyncpg.Pool[Any]
    ) -> None:
        # The windowed retained-slate read against a fixed-fraction drop: an
        # O(W) read whose window grows with N here (drop == half), so this is
        # the most demanding advisory case. It must still stay well under 2.5x.
        small_args = (_SESSION_ID_SMALL, _ACCOUNT_ID, _N_SMALL * _DELTA // 2)
        large_args = (_SESSION_ID, _ACCOUNT_ID, _N_LARGE * _DELTA // 2)
        t1 = await _time_read(two_scale_pool, _SQL_RETAINED_WINDOW, small_args, repeats=_M_REPEATS)
        t2 = await _time_read(two_scale_pool, _SQL_RETAINED_WINDOW, large_args, repeats=_M_REPEATS)
        ratio = t2 / t1 if t1 > 0 else float("inf")
        assert ratio < 2.5, (
            f"[ADVISORY, non-gating] retained-window read t(N2)/t(N1) = {ratio:.2f} "
            f">= 2.5 (t1={t1 * 1e3:.2f}ms, t2={t2 * 1e3:.2f}ms). This mark is NOT a "
            f"required check; shared-runner jitter can exceed 2.5x for O(1). Treat as "
            f"a loud signal to inspect for app-side / N+1 complexity, not a gate (#1661)."
        )

    async def test_latest_cumulative_seek_scales_flat(
        self, two_scale_pool: asyncpg.Pool[Any]
    ) -> None:
        t1 = await _time_read(
            two_scale_pool, _SQL_LATEST_CUMULATIVE, (_SESSION_ID_SMALL,), repeats=_M_REPEATS
        )
        t2 = await _time_read(
            two_scale_pool, _SQL_LATEST_CUMULATIVE, (_SESSION_ID,), repeats=_M_REPEATS
        )
        ratio = t2 / t1 if t1 > 0 else float("inf")
        assert ratio < 2.5, (
            f"[ADVISORY, non-gating] latest-cumulative seek t(N2)/t(N1) = {ratio:.2f} "
            f">= 2.5 — an O(1) index seek should be flat across a 10x slate. Advisory "
            f"only (jitter); inspect, do not gate (#1661)."
        )

    async def test_find_tool_result_event_scales_flat(
        self, two_scale_pool: asyncpg.Pool[Any]
    ) -> None:
        """Append-phase advisory backstop (#1750): the #1 #1733 exemplar
        (``find_tool_result_event``, post-#1734 role-column form) is an O(1)
        index seek and should stay flat across a 10x slate. Advisory only —
        catches app-side/N+1 complexity the plan-shape oracle can't see."""
        t1 = await _time_read(
            two_scale_pool,
            _SQL_FIND_TOOL_RESULT_EVENT,
            (_SESSION_ID_SMALL, _ACCOUNT_ID, "tc_no_such_call"),
            repeats=_M_REPEATS,
        )
        t2 = await _time_read(
            two_scale_pool,
            _SQL_FIND_TOOL_RESULT_EVENT,
            (_SESSION_ID, _ACCOUNT_ID, "tc_no_such_call"),
            repeats=_M_REPEATS,
        )
        ratio = t2 / t1 if t1 > 0 else float("inf")
        assert ratio < 2.5, (
            f"[ADVISORY, non-gating] find_tool_result_event t(N2)/t(N1) = {ratio:.2f} "
            f">= 2.5 — an O(1) index seek should be flat across a 10x slate. Advisory "
            f"only (jitter); inspect, do not gate (#1750)."
        )
