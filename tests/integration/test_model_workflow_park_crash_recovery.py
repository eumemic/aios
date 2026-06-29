"""Crash-recovery for the model-dispatch park (#1635).

The session-side park introduced by the ``workflow:`` model binding (#1634) needs
its own crash-recovery. The existing recovery is **tool-call-bound**:
``find_parked_servicer`` keys on a ``tool_call_id`` and the ghost-scan re-parks only
servicers backed by an open tool_call in a *persisted assistant message*. A
model-dispatch park has **neither** — the assistant message is the bound workflow's
output, produced only after the park resolves — so a worker crash while parked
**strands** the session: the inner run completes, but the in-process harvest task
died with the worker, so nothing writes the harvest event back and the turn never
completes.

These tests drive the REAL park → crash → sweep-repark → harvest path against a
testcontainer Postgres:

* ``test_crash_while_parked_is_recovered_by_sweep`` — park, simulate a crash (the
  harvest task never ran), the inner run resolves, then the crash-recovery sweep
  re-parks → the harvest lands and the turn completes with **no loss and no
  double-charge**.
* ``test_repark_is_idempotent_no_double_harvest`` — re-parking a park that has
  already been harvested writes NO second harvest (the dedup guard on ``run_id``),
  and folding still produces exactly one assistant turn.
* ``test_steady_state_park_is_not_double_parked`` — a park whose live harvest task is
  in-flight in THIS worker is NOT re-parked by the sweep (the in-flight key gate).
* ``test_consumed_park_is_not_reparked`` — a fully-folded (consumed) park is not a
  crash-recovery candidate.

As in ``test_model_workflow_park_sweep.py`` the fire-and-forget harvest poller is
patched to a no-op; the harvest is driven manually by resolving the inner run +
calling ``write_harvest_event`` — exactly the path the (lost) park task would take,
minus the LISTEN/NOTIFY plumbing. The crash-recovery re-park is exercised directly
via ``repark_stranded_model_dispatch``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import model_workflow as mwf
from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import run_session_step
from aios.harness.model_workflow import write_harvest_event
from aios.harness.sweep import repark_stranded_model_dispatch
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.workflows import run_tools
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration


class _EmptyToolProvider:
    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        return []

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, Any]:
        return {}


_INNER_COST_MICROUSD = 4321
_INNER_SCRIPT = (
    "async def main(input):\n"
    "    return {'content': 'recovered answer', 'tool_calls': [], 'finish_reason': 'stop'}\n"
)

_ACCOUNT = "acc_mwfcr"
_ENV = "env_mwfcr"


@pytest.fixture
async def mwf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + an inflight registry + a seeded root tenant.

    ``defer_*wake`` enqueues are patched out (the steps are driven directly), and
    ``_launch_harvest_task`` is a no-op so the park does not spawn the background
    poller — simulating the worker crash (the in-process harvest task never runs).
    The crash-recovery re-park is driven explicitly in-test.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev_pool = runtime.pool
    prev_reg = runtime.inflight_tool_registry
    prev_tp = runtime.tool_provider
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    runtime.tool_provider = _EmptyToolProvider()
    mwf.reset_inflight_harvests()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'mwfcr-root')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'mwfcr-env', '{}'::jsonb, $2)",
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
            mock.patch("aios.harness.loop.defer_wake", new=AsyncMock()),
            mock.patch("aios.harness.loop.defer_run_wake", new=AsyncMock()),
            # Simulate the worker crash: the park's fire-and-forget harvest poller never
            # runs, so no harvest is ever written by the park task itself. The harvest is
            # driven manually below, and crash-recovery is exercised via the sweep.
            mock.patch("aios.harness.model_workflow._launch_harvest_task", new=mock.Mock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        mwf.reset_inflight_harvests()
        runtime.pool = prev_pool
        runtime.inflight_tool_registry = prev_reg
        runtime.tool_provider = prev_tp
        await pool.close()


async def _make_bound_session(pool: asyncpg.Pool[Any]) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id=_ACCOUNT, name="inner-model", script=_INNER_SCRIPT
        )
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="mwfcr-agent",
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
        title="mwfcr",
        metadata={},
        account_id=_ACCOUNT,
    )
    await sessions_service.append_user_message(pool, session.id, "answer this", account_id=_ACCOUNT)
    return session.id


async def _inner_run_ids(pool: asyncpg.Pool[Any], session_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id FROM wf_runs WHERE launcher_session_id = $1 AND account_id = $2 "
            "ORDER BY created_at",
            session_id,
            _ACCOUNT,
        )
    return [r["id"] for r in rows]


async def _run_caller(pool: asyncpg.Pool[Any], run_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT caller FROM wf_runs WHERE id = $1 AND account_id = $2",
            run_id,
            _ACCOUNT,
        )


async def _account_spent(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return await db_queries.get_account_spent_microusd(conn, _ACCOUNT)


async def _harvest_events(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_harvest'",
            session_id,
        )


async def _harvest_end_events(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'span' "
            "AND data->>'event' = 'model_workflow_harvest_end'",
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
    total = 0
    async with pool.acquire() as conn:
        for run_id in await _inner_run_ids(pool, session_id):
            total += await wf_queries.get_run_call_llm_cost_microusd(
                conn, run_id, account_id=_ACCOUNT
            )
    return total


async def _resolve_inner_run(pool: asyncpg.Pool[Any], run_id: str) -> Any:
    """Run the inner workflow to completion + charge its ``call_llm`` meter once.

    This is the inner run reaching its terminal state — which is exactly what
    happens during the worker crash: the run finishes, but nobody on the SESSION
    side wrote the harvest back (the park task died). Returns the run's output.
    """
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        assert run.status == "completed", f"inner run not completed: {run.status}"
        await wf_queries.add_run_call_llm_cost_microusd(
            conn, run_id, _INNER_COST_MICROUSD, account_id=_ACCOUNT
        )
    return run.output


async def test_crash_while_parked_is_recovered_by_sweep(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """Kill the worker while parked → on restart the sweep re-parks, harvests, completes."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)

    # ── PARK (step N): one awaited inner run; the harvest task is a no-op (crash). ──
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    # The park's caller edge carries the tool_call_id-less model-dispatch discriminant —
    # the durable marker crash-recovery keys on (no assistant message, no tool_call_id).
    caller = await _run_caller(pool, inner_run_id)
    assert caller["kind"] == "session" and caller["id"] == session_id
    assert caller["purpose"] == "model_dispatch"

    # The worker crashed: NO harvest was written (the in-process task never ran), and
    # the session is stranded — the existing ghost scan can't see this park.
    assert await _harvest_events(pool, session_id) == 0
    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert (session_id, inner_run_id, _ACCOUNT) in stranded, (
        "a crashed-while-parked session must be a crash-recovery candidate"
    )

    # The inner run resolves (it completed before / after the crash — its terminal state
    # is durable regardless). Nothing has written that state back to the session yet.
    run_output = await _resolve_inner_run(pool, inner_run_id)

    spent_after_run = await _account_spent(pool)

    # ── RESTART: the crash-recovery sweep re-derives the stranded park and re-parks. ──
    # Patch the (still-no-op) harvest task to a real harvest write — this stands in for
    # the re-launched ``_park_and_signal`` reading the run's terminal state and writing
    # the harvest, minus the LISTEN/NOTIFY poll.
    async def _fake_relaunched_harvest() -> None:
        await write_harvest_event(
            pool,
            session_id,
            run_id=inner_run_id,
            outcome="ok",
            output=run_output,
            error=None,
            account_id=_ACCOUNT,
        )

    with mock.patch.object(
        mwf, "_launch_harvest_task", side_effect=lambda *a, **k: None
    ) as launch_spy:
        reparked = await repark_stranded_model_dispatch(pool)
    assert reparked == 1, "the sweep must re-park exactly the one stranded session"
    launch_spy.assert_called_once()
    # The re-park launched a harvest task for the right (session, run) — drive it.
    _, kwargs = launch_spy.call_args
    assert kwargs["run_id"] == inner_run_id
    await _fake_relaunched_harvest()

    # The harvest event now exists; the next session step folds it into the turn.
    assert await _harvest_events(pool, session_id) == 1
    await run_session_step(session_id, cause="model_workflow_harvest")

    # The turn completed: exactly one assistant message carrying the inner answer.
    assistants = await _assistant_messages(pool, session_id)
    assert len(assistants) == 1
    assert assistants[0]["content"] == "recovered answer"
    assert await _harvest_end_events(pool, session_id) == 1, "the fold writes one consumed marker"

    # No double-charge: the harvest does not re-charge the outer turn, and the only
    # inference spend is the inner run's, charged ONCE (no second run was launched).
    assert await _account_spent(pool) == spent_after_run, "harvest must not re-charge"
    assert await _total_inner_call_llm_cost(pool, session_id) == _INNER_COST_MICROUSD
    assert await _inner_run_ids(pool, session_id) == [inner_run_id], (
        "crash-recovery must NOT launch a second inner run"
    )

    # The session is fully recovered — no longer a crash-recovery candidate.
    async with pool.acquire() as conn:
        stranded_after = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded_after]


async def test_repark_is_idempotent_no_double_harvest(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """Re-parking a park whose harvest already landed writes NO second harvest."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    run_output = await _resolve_inner_run(pool, inner_run_id)
    # The harvest already landed (e.g. a racing task won, or a prior re-park).
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    assert await _harvest_events(pool, session_id) == 1

    # A park WITH a harvest is no longer stranded — the sweep does not re-park it.
    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded], (
        "a harvested park is not a crash-recovery candidate"
    )
    reparked = await repark_stranded_model_dispatch(pool)
    assert reparked == 0

    # Even a FORCED re-park (the dedup guard is what protects us) writes no second harvest.
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    assert await _harvest_events(pool, session_id) == 1, "harvest write is idempotent on run_id"

    # The fold still produces exactly one assistant turn.
    await run_session_step(session_id, cause="model_workflow_harvest")
    assert len(await _assistant_messages(pool, session_id)) == 1


async def test_steady_state_park_is_not_double_parked(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """A park whose live harvest task is in-flight in THIS worker is not re-parked."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    # Simulate the live (not-crashed) harvest task: its in-flight key is registered.
    mwf._INFLIGHT_HARVESTS.add((session_id, inner_run_id))
    try:
        # The park IS still unharvested (the live task hasn't written it yet) — so it is a
        # query candidate — but the in-flight gate means the sweep does NOT re-park it.
        async with pool.acquire() as conn:
            stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
        assert (session_id, inner_run_id, _ACCOUNT) in stranded
        reparked = await repark_stranded_model_dispatch(pool)
        assert reparked == 0, "a park with a live in-flight harvest task must not be double-parked"
    finally:
        mwf._INFLIGHT_HARVESTS.discard((session_id, inner_run_id))


async def test_consumed_park_is_not_reparked(mwf_runtime: asyncpg.Pool[Any]) -> None:
    """A fully-folded (consumed) park is not a crash-recovery candidate."""
    pool = mwf_runtime
    session_id = await _make_bound_session(pool)
    await run_session_step(session_id)
    [inner_run_id] = await _inner_run_ids(pool, session_id)

    run_output = await _resolve_inner_run(pool, inner_run_id)
    await write_harvest_event(
        pool,
        session_id,
        run_id=inner_run_id,
        outcome="ok",
        output=run_output,
        error=None,
        account_id=_ACCOUNT,
    )
    # Fold the harvest — writes the consumed marker (model_workflow_harvest_end).
    await run_session_step(session_id, cause="model_workflow_harvest")
    assert await _harvest_end_events(pool, session_id) == 1

    async with pool.acquire() as conn:
        stranded = await db_queries.find_unharvested_model_dispatch_parks(conn)
    assert session_id not in [s[0] for s in stranded], "a consumed park is never re-parked"
    assert await repark_stranded_model_dispatch(pool) == 0
