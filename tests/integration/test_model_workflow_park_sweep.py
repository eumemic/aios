"""Live park → sweep → harvest for the ``workflow:`` model binding (#1634).

The root coverage gap the seat review flagged: ``test_model_workflow.py`` stubs
``find_latest_model_workflow_park`` / ``find_model_workflow_harvest`` with
pre-populated dicts, so it never exercises the LIVE transition where the
multi-dispatch / multi-billing defect lives. The defect: a parked step writes a
``span`` event, which does NOT advance ``last_stimulus_seq`` / ``last_reacted_seq``,
so the unreacted-stimulus inequality that caused the park STILL holds afterwards →
``find_sessions_needing_inference`` keeps returning the session → the periodic sweep
re-wakes it every tick while the inner run deliberates → each re-wake re-entered the
park branch and launched a FRESH awaited inner run (N parallel paid runs per turn).

These tests drive the REAL park/sweep/harvest path against a testcontainer Postgres:

* ``test_sweep_ticks_do_not_relaunch_inner_run`` — park-open, then ≥2 sweep ticks
  while the inner run is unresolved → assert exactly ONE awaited inner run exists
  (no re-dispatch), the session is genuinely a sweep candidate the whole time (so
  the ticks are real), then resolve the run + harvest → the harvest folds in once,
  the outer turn is NOT re-charged (``no_recharge``), and the run-level ``call_llm``
  cost meter carries exactly one unit of inner spend (one run charged, not N).
* ``test_deliberation_spanning_multiple_wakes_does_not_trip_cap`` — a deliberation
  spanning ≥2 wakes never trips the 960s harness-step cap, because each park-then-end
  is its own bounded ``run_session_step`` call rather than one step awaiting inline.

The fire-and-forget harvest task (``_launch_harvest_task``, which would poll the run
via ``await_task`` against a live db_url) is patched to a no-op; the harvest is driven
manually by resolving the inner run + calling ``write_harvest_event`` — exactly the
path the park task would take, minus the LISTEN/NOTIFY plumbing.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.config import HARNESS_STEP_TIMEOUT_S
from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import run_session_step
from aios.harness.model_workflow import write_harvest_event
from aios.harness.sweep import find_sessions_needing_inference
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.workflows import run_tools
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration


class _EmptyToolProvider:
    """A ``ToolProvider`` for sessions with no connections — the step's prelude
    merges in no connector tools (this binding session has none)."""

    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        return []

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, Any]:
        return {}


# The inner workflow returns a valid assistant-turn shape. Its inference cost is
# charged the way a real ``call_llm`` leaf charges — one unit on the run's own
# run-level ``call_llm`` meter — but driven by the test at resolution (below), so
# the script stays trivial. SUM of that meter across ALL inner runs the turn
# launched is the multi-billing gauge: one run → one unit; N runs (the bug) → N.
_INNER_COST_MICROUSD = 1234
_INNER_SCRIPT = (
    "async def main(input):\n"
    "    return {'content': 'inner answer', 'tool_calls': [], 'finish_reason': 'stop'}\n"
)

_ACCOUNT = "acc_mwf"
_ENV = "env_mwf"


@pytest.fixture
async def mwf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + an inflight registry + a seeded root tenant.

    ``defer_*wake`` enqueues are patched out (the steps are driven directly), and
    ``_launch_harvest_task`` is a no-op so the park does not spawn the background
    ``await_task`` poller — the harvest is driven manually in-test.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool = runtime.pool
    prev_reg = runtime.inflight_tool_registry
    prev_tp = runtime.tool_provider
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    runtime.tool_provider = _EmptyToolProvider()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'mwf-root')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'mwf-env', '{}'::jsonb, $2)",
                _ENV,
                _ACCOUNT,
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
            # The post-step wakes ``run_session_step`` itself enqueues (nudge / retry /
            # archive-reclaim / autoerror) — imported into ``loop``'s own namespace, so
            # patch them there. Keeps the step driven directly with no real procrastinate
            # enqueue against the test DB.
            mock.patch("aios.harness.loop.defer_wake", new=AsyncMock()),
            mock.patch("aios.harness.loop.defer_run_wake", new=AsyncMock()),
            # The park's fire-and-forget harvest poller — patch it out so the park is
            # a pure DB write (``_park_and_signal`` never runs; the harvest is driven
            # manually below). This also avoids a real ``await_task`` LISTEN against
            # the test DB.
            mock.patch("aios.harness.model_workflow._launch_harvest_task", new=mock.Mock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev_pool
        runtime.inflight_tool_registry = prev_reg
        runtime.tool_provider = prev_tp
        await pool.close()


async def _make_bound_session(pool: asyncpg.Pool[Any]) -> tuple[str, str]:
    """Create a workflow, an agent bound to ``workflow:<id>``, and a session.

    Returns ``(session_id, workflow_id)``.
    """
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id=_ACCOUNT, name="inner-model", script=_INNER_SCRIPT
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="mwf-agent",
        model=f"workflow:{wf.id}",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.get_environment(pool, _ENV, account_id=_ACCOUNT)
    session = await sessions_service.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="mwf",
        metadata={},
        account_id=_ACCOUNT,
    )
    # A user message is a stimulus → the session is active (needs inference): the
    # precondition the park path consumes.
    await sessions_service.append_user_message(pool, session.id, "answer this", account_id=_ACCOUNT)
    return session.id, wf.id


async def _inner_run_ids(pool: asyncpg.Pool[Any], session_id: str) -> list[str]:
    """All awaited inner runs this session launched (the multi-dispatch gauge)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id FROM wf_runs WHERE launcher_session_id = $1 AND account_id = $2 "
            "ORDER BY created_at",
            session_id,
            _ACCOUNT,
        )
    return [r["id"] for r in rows]


async def _account_spent(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return await db_queries.get_account_spent_microusd(conn, _ACCOUNT)


async def _park_events(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_park'",
            session_id,
        )


async def _assistant_messages(pool: asyncpg.Pool[Any], session_id: str) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'assistant' ORDER BY seq",
            session_id,
        )
    return [r["data"] for r in rows]


async def _total_inner_call_llm_cost(pool: asyncpg.Pool[Any], session_id: str) -> int:
    """Sum the ``call_llm`` cost meter across EVERY inner run the turn launched.

    The multi-billing gauge: each launched inner run is a paid inference site that
    charges its own meter once. One run → one unit; N runs (the bug) → N units.
    """
    total = 0
    async with pool.acquire() as conn:
        for run_id in await _inner_run_ids(pool, session_id):
            total += await wf_queries.get_run_call_llm_cost_microusd(
                conn, run_id, account_id=_ACCOUNT
            )
    return total


async def _resolve_inner_run_and_harvest(
    pool: asyncpg.Pool[Any], session_id: str, run_id: str
) -> None:
    """Run the inner workflow to completion (charging its ``call_llm`` meter once,
    as a real inference leaf would), then write the harvest event.

    The harvest write is exactly what the (patched-out) park task does on resolution
    — read the run's terminal output and append the ``model_workflow_harvest`` span —
    minus the ``await_task`` LISTEN/NOTIFY poll.
    """
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        assert run.status == "completed", f"inner run not completed: {run.status}"
        # Charge the run's ``call_llm`` meter once — the spend a real inference leaf
        # inside the run would book at its own site (the harvest re-charges nothing).
        await wf_queries.add_run_call_llm_cost_microusd(
            conn, run_id, _INNER_COST_MICROUSD, account_id=_ACCOUNT
        )
    await write_harvest_event(
        pool,
        session_id,
        run_id=run_id,
        outcome="ok",
        output=run.output,
        error=None,
        account_id=_ACCOUNT,
    )


async def test_sweep_ticks_do_not_relaunch_inner_run(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """≥2 sweep ticks during deliberation launch NO second run; harvest bills once."""
    pool = mwf_runtime
    session_id, _ = await _make_bound_session(pool)

    # ── PARK (step N): the first wake opens exactly one awaited inner run. ──
    await run_session_step(session_id)
    assert await _park_events(pool, session_id) == 1
    runs_after_park = await _inner_run_ids(pool, session_id)
    assert len(runs_after_park) == 1, "park must open exactly one awaited inner run"
    inner_run_id = runs_after_park[0]

    # The park wrote a ``span`` (not a stimulus), so the unreacted-stimulus inequality
    # that caused the park STILL holds → the session is genuinely a sweep candidate.
    # This is the precondition that makes the periodic sweep re-wake it every tick;
    # if it were false, the test couldn't exercise the defect at all.
    needing = await find_sessions_needing_inference(pool, runtime.require_inflight_tool_registry())
    assert session_id in needing, (
        "parked session must still be a sweep candidate (the defect's root)"
    )

    # ── ≥2 SWEEP TICKS while the inner run is unresolved. ──
    # Pre-fix: each tick re-enters the park branch and launches a fresh awaited run.
    # Post-fix: each tick sees PARK_PENDING and ends the step without re-parking.
    for _ in range(3):
        await run_session_step(session_id, cause="sweep")

    assert await _inner_run_ids(pool, session_id) == [inner_run_id], (
        "exactly one inner awaited run must exist regardless of how many sweep ticks "
        "elapsed during deliberation"
    )
    # No second park, and still no assistant turn (the run hasn't resolved).
    assert await _park_events(pool, session_id) == 1
    assert await _assistant_messages(pool, session_id) == []

    # ── RESOLVE the inner run + harvest. ──
    await _resolve_inner_run_and_harvest(pool, session_id, inner_run_id)
    spent_before_harvest = await _account_spent(pool)
    await run_session_step(session_id, cause="model_workflow_harvest")

    # The harvest folded into exactly one assistant turn carrying the inner answer.
    assistants = await _assistant_messages(pool, session_id)
    assert len(assistants) == 1
    assert assistants[0]["content"] == "inner answer"

    # The harvest does NOT re-charge the outer turn (``no_recharge``) — the only
    # inference spend for the turn is the inner run's, charged ONCE at its own
    # ``call_llm`` site.
    assert await _account_spent(pool) == spent_before_harvest, "harvest must not re-charge"
    # The run-level cost meter, summed across EVERY inner run the turn launched, is
    # exactly one unit. Pre-fix the sweep ticks would have launched extra runs, each
    # a second paid inference site → this sum would be N units. (== one unit only
    # because exactly one inner run exists.)
    assert await _total_inner_call_llm_cost(pool, session_id) == _INNER_COST_MICROUSD, (
        "run-level call_llm cost must be charged exactly once across the whole turn"
    )
    assert await _inner_run_ids(pool, session_id) == [inner_run_id]


async def test_deliberation_spanning_multiple_wakes_does_not_trip_cap(
    mwf_runtime: asyncpg.Pool[Any],
) -> None:
    """A deliberation spanning ≥2 wakes never trips the 960s harness-step cap.

    "Stay responsive while parked" is by construction: each park-then-end is its own
    bounded ``run_session_step`` call, NOT one step awaiting the inner run inline. So
    no single step's wall-clock approaches ``HARNESS_STEP_TIMEOUT_S`` no matter how
    long the inner workflow deliberates across wakes.
    """
    pool = mwf_runtime
    session_id, _ = await _make_bound_session(pool)

    # Park, then several re-wakes spanning the deliberation — each must be far under cap.
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    for _ in range(3):
        t0 = time.monotonic()
        await run_session_step(session_id, cause="sweep")
        elapsed = time.monotonic() - t0
        assert elapsed < HARNESS_STEP_TIMEOUT_S, (
            f"a single parked-step wake took {elapsed:.1f}s, approaching the "
            f"{HARNESS_STEP_TIMEOUT_S:.0f}s cap — the deliberation is being awaited "
            "inline instead of spanning wakes"
        )

    # No step was latched errored by a step_timeout across the multi-wake span.
    async with pool.acquire() as conn:
        timeouts = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'step_timeout'",
            session_id,
        )
    assert timeouts == 0

    # Still exactly one inner run across the whole multi-wake deliberation.
    assert await _inner_run_ids(pool, session_id) == [inner_run_id]
