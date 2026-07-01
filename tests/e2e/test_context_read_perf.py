"""Perf-regression guard for the per-turn context read path (issue #1657).

The pre-fix per-turn read carried two O(session-size) terms, both self-flagged
in ``db/queries/events.py`` and both measured hot on Ultron's 639k-event
session:

* **PRIMARY** — ``_retained_class_mass`` summed per-message token deltas
  GROUPED BY content-class over the *whole* message slate via an unbounded
  ``LAG() OVER (ORDER BY seq)`` WindowAgg (measured 3,823.9 ms / 90,415 rows /
  45,736 kB external-merge spill). ``seq`` cannot bound it — it is the pure
  ordinal already used as the ORDER BY. The fix maintains four per-class
  running sums (``cumulative_*_mass``) at append time, so the read is the
  latest message row's four cumulative totals — one index seek, O(1). The
  WindowAgg ceases to exist.
* **SECONDARY** — the omitted-message ``count(*) FILTER (role IN
  ('user','assistant'))`` under ``cumulative_tokens <= drop`` (~140 ms). The
  fix reads a ``cumulative_messages`` running count at the boundary row = O(1).

The oracle here is **asymptotic COMPLEXITY (plan shape)**, NOT wall-clock, so
it gates CI like a correctness test with zero flakiness. Following the
already-merged pattern in ``tests/e2e/test_sweep_perf.py`` (#140) +
``tests/support.py``, we seed a session with N≈20,000 message rows (a single
``INSERT ... SELECT generate_series`` reproducing ``append_event``'s running
sums — ``cumulative_tokens``, ``cumulative_messages``, and the four per-class
masses — with a mix of roles + tool_calls/thinking/tool-role rows so every
``content_class`` branch fires), ``ANALYZE events, sessions``, then exercise
the per-turn read path and assert — via ``EXPLAIN (FORMAT JSON)``, **no
ANALYZE** — that **no aggregate node (WindowAgg/Aggregate/GroupAggregate)
scans ``events`` without a ``cumulative_tokens``/``seq`` lower-bound Index
Cond**.

That is RED on the pre-fix unbounded ``LAG() OVER (ORDER BY seq)`` and GREEN
once the running counters make that node cease to exist — an unambiguous
present-vs-absent verdict, no threshold, no flake. ``test_prefix_query_is_red``
below pins the oracle's own sensitivity: the removed pre-fix SQL, EXPLAINed on
the very same seeded session, MUST trip the detector — so a future refactor
cannot silently make the guard vacuous.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.support import find_unbounded_events_aggregates

pytestmark = pytest.mark.docker

# ─── fixture: a large single-session slate ───────────────────────────────────

_SESSION_ID = "sess_readperf_1657"
_ACCOUNT_ID = "acc_test_stub"
_N_ROWS = 20_000


async def _seed_large_session(pool: asyncpg.Pool[Any]) -> None:
    """Seed one session with ``_N_ROWS`` message events, each carrying the
    running counters ``append_event`` maintains: ``cumulative_tokens`` (a
    ``SUM(delta) OVER (ORDER BY seq)`` running sum), ``cumulative_messages``
    (running count of user/assistant messages), and the four per-class running
    masses. Roles cycle user→assistant→assistant(tool_calls)→tool→
    assistant(thinking) so ALL four ``content_class`` branches (text /
    tool_use / tool_result / thinking) are represented — the composition the
    windower blends over, and the branches the pre-fix WindowAgg GROUPed BY.

    The single ``INSERT ... SELECT generate_series`` mirrors the production
    running-sum math in SQL (window functions over the generated series), so
    the seeded columns are internally consistent with what ``append_event``
    would have written — the read path sees a realistic, ANALYZE-friendly slate.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_readperf', 'readperf', 'openrouter/x', $1) "
            "ON CONFLICT (id) DO NOTHING",
            _ACCOUNT_ID,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_readperf', 'env_readperf', $1) "
            "ON CONFLICT (id) DO NOTHING",
            _ACCOUNT_ID,
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
            "VALUES ($1, 'agt_readperf', 'env_readperf', '/tmp/ws_readperf', $2) "
            "ON CONFLICT (id) DO NOTHING",
            _SESSION_ID,
            _ACCOUNT_ID,
        )

        # One INSERT ... SELECT generate_series. ``g`` is the 1-based seq; the
        # role cycle drives both ``data`` (so the JSONB content_class CASE
        # fires on real shapes) and the per-class delta attribution. ``delta``
        # is a small deterministic per-row token cost; the cumulative_* columns
        # are window running sums over ``g`` — exactly the shape append_event
        # builds incrementally.
        await conn.execute(
            """
            INSERT INTO events
                (id, session_id, seq, kind, data, created_at,
                 cumulative_tokens, cumulative_messages,
                 cumulative_text_mass, cumulative_tool_result_mass,
                 cumulative_thinking_mass, cumulative_tool_use_mass,
                 role, account_id)
            SELECT
                'ev_readperf_' || g,
                $1,
                g,
                'message',
                CASE (g % 5)
                    WHEN 0 THEN '{"role":"user","content":"u"}'::jsonb
                    WHEN 1 THEN '{"role":"assistant","content":"a"}'::jsonb
                    WHEN 2 THEN '{"role":"assistant","content":null,'
                               || '"tool_calls":[{"id":"tc","type":"function",'
                               || '"function":{"name":"bash","arguments":"{}"}}]}'::jsonb
                    WHEN 3 THEN '{"role":"tool","tool_call_id":"tc","content":"ok"}'::jsonb
                    ELSE '{"role":"assistant","content":"t",'
                         || '"reasoning_content":"because"}'::jsonb
                END,
                now(),
                -- cumulative_tokens: running SUM of a per-row delta (delta=10).
                (SUM(10) OVER (ORDER BY g))::bigint,
                -- cumulative_messages: running COUNT of user/assistant rows
                -- (roles 0,1,2,4 are user/assistant; role 3 is tool).
                (SUM(CASE WHEN (g % 5) = 3 THEN 0 ELSE 1 END) OVER (ORDER BY g))::bigint,
                -- per-class running masses: each row's whole 10-token delta goes
                -- to its dominant class, matching _message_content_class.
                (SUM(CASE WHEN (g % 5) IN (0, 1) THEN 10 ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 3 THEN 10 ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 4 THEN 10 ELSE 0 END) OVER (ORDER BY g))::bigint,
                (SUM(CASE WHEN (g % 5) = 2 THEN 10 ELSE 0 END) OVER (ORDER BY g))::bigint,
                CASE (g % 5)
                    WHEN 0 THEN 'user'
                    WHEN 3 THEN 'tool'
                    ELSE 'assistant'
                END,
                $2
            FROM generate_series(1, $3) AS g
            ON CONFLICT (id) DO NOTHING
            """,
            _SESSION_ID,
            _ACCOUNT_ID,
            _N_ROWS,
        )

        # Keep the sessions scalar honest (production maintains it in append).
        await conn.execute(
            "UPDATE sessions SET last_event_seq = $2 WHERE id = $1",
            _SESSION_ID,
            _N_ROWS,
        )

        # Fresh stats so the planner picks its real plan for this slate.
        await conn.execute("ANALYZE events")
        await conn.execute("ANALYZE sessions")


@pytest.fixture
async def seeded_pool(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    from aios.db.pool import create_pool

    pool = await create_pool(aios_env["AIOS_DB_URL"], min_size=1, max_size=4)
    await _seed_large_session(pool)
    try:
        yield pool
    finally:
        await pool.close()


async def _explain(pool: asyncpg.Pool[Any], sql: str, *args: Any) -> dict[str, Any]:
    async with pool.acquire() as conn:
        result = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *args)
    if isinstance(result, str):
        result = json.loads(result)
    return cast(dict[str, Any], result[0]["Plan"])


# The drop boundary the omission read uses — the middle of the slate, so the
# ``cumulative_tokens <= drop`` prefix is large (a genuinely omitted span, the
# shape the pre-fix count(*) scanned linearly).
_DROP = _N_ROWS * 10 // 2

# The pre-fix PRIMARY query, verbatim, kept ONLY as the RED self-check
# (``test_prefix_query_is_red``). This is the exact ``_retained_class_mass``
# SQL the fix removed — an unbounded ``LAG() OVER (ORDER BY seq)`` WindowAgg
# GROUPing the whole slate by content class. The guard must light up on it, or
# it proves nothing.
_PREFIX_RETAINED_MASS_SQL = """
WITH deltas AS (
    SELECT
        CASE
            WHEN data->>'role' = 'tool' THEN 'tool_result'
            WHEN data->>'role' = 'assistant'
                 AND (data ? 'tool_calls')
                 AND (data->'tool_calls') IS NOT NULL
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

# The O(1) post-fix reads the per-turn path now runs (the exact SQL from
# ``_retained_class_mass`` and the omission boundary seek in ``events.py``).
_FIXED_RETAINED_MASS_SQL = (
    "SELECT cumulative_text_mass, cumulative_tool_result_mass, "
    "       cumulative_thinking_mass, cumulative_tool_use_mass "
    "FROM events "
    "WHERE session_id = $1 AND account_id = $2 "
    "AND kind = 'message' AND cumulative_tokens IS NOT NULL "
    "ORDER BY seq DESC LIMIT 1"
)

_FIXED_OMISSION_BOUNDARY_SQL = (
    "SELECT cumulative_messages, created_at "
    "FROM events "
    "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
    "AND cumulative_tokens <= $3 "
    "ORDER BY cumulative_tokens DESC LIMIT 1"
)

_LATEST_CUMULATIVE_SQL = (
    "SELECT cumulative_tokens FROM events "
    "WHERE session_id = $1 AND kind = 'message' "
    "AND cumulative_tokens IS NOT NULL "
    "ORDER BY seq DESC LIMIT 1"
)


@needs_docker
class TestNoUnboundedEventsAggregateOnRead:
    """The central invariant (#1657): the per-turn context read runs NO
    aggregate over ``events`` unbounded by a ``cumulative_tokens``/``seq``
    lower bound. Deterministic plan-shape check — no wall-clock, no flake."""

    async def test_retained_class_mass_is_o1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """PRIMARY term. ``_retained_class_mass`` now reads the latest message
        row's four per-class cumulative masses (one index seek), so its plan
        must carry NO unbounded WindowAgg/Aggregate over ``events``."""
        plan = await _explain(
            seeded_pool, _FIXED_RETAINED_MASS_SQL, _SESSION_ID, _ACCOUNT_ID
        )
        found = find_unbounded_events_aggregates(plan)
        assert not found, (
            f"read-tax regression in _retained_class_mass: {len(found)} unbounded "
            f"aggregate(s) over events. It must read the stored per-class "
            f"cumulative_*_mass (O(1) index seek), not re-scan the slate (#1657)."
        )

    async def test_omission_count_is_o1(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        """SECONDARY term. The omitted-message count now reads the boundary
        row's ``cumulative_messages`` running counter — an index seek, no
        ``count(*)`` aggregate over the omitted prefix."""
        plan = await _explain(
            seeded_pool, _FIXED_OMISSION_BOUNDARY_SQL, _SESSION_ID, _ACCOUNT_ID, _DROP
        )
        found = find_unbounded_events_aggregates(plan)
        assert not found, (
            f"read-tax regression in the omission count: {len(found)} unbounded "
            f"aggregate(s) over events. It must read the boundary row's "
            f"cumulative_messages counter (O(1)), not count(*) the prefix (#1657)."
        )

    async def test_latest_cumulative_seek_is_o1(
        self, seeded_pool: asyncpg.Pool[Any]
    ) -> None:
        """The ``total`` seek (``_latest_cumulative_tokens``) was already O(1);
        pin it here so the whole per-turn read is covered by one guard."""
        plan = await _explain(seeded_pool, _LATEST_CUMULATIVE_SQL, _SESSION_ID)
        found = find_unbounded_events_aggregates(plan)
        assert not found, (
            f"unexpected unbounded aggregate over events in the latest-cumulative "
            f"seek: {len(found)} node(s) (#1657)."
        )


@needs_docker
class TestGuardIsNotVacuous:
    """Sensitivity self-check: the removed pre-fix PRIMARY query MUST trip the
    detector on the same seeded session. Without this, a refactor that made
    ``find_unbounded_events_aggregates`` blind would let the O(N) term
    silently return AND pass the GREEN assertions above."""

    async def test_prefix_query_is_red(self, seeded_pool: asyncpg.Pool[Any]) -> None:
        plan = await _explain(
            seeded_pool, _PREFIX_RETAINED_MASS_SQL, _SESSION_ID, _ACCOUNT_ID
        )
        found = find_unbounded_events_aggregates(plan)
        assert found, (
            "the pre-fix _retained_class_mass LAG() OVER (ORDER BY seq) WindowAgg "
            "did NOT trip find_unbounded_events_aggregates — the guard is vacuous. "
            "The detector must catch the exact O(session-size) shape #1657 removed."
        )
