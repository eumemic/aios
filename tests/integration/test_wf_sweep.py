"""B1.9 — worker wiring: the wake_workflow task + the needs-step run sweep (#780).

The filter's contract is recall: every lost-wake mode must leave SQL-visible state
one clause matches, while a parked run with nothing new is NOT woken (each blanket
wake costs a full memo reship + script replay). Each test pins one clause from
``list_run_ids_needing_step``'s docstring.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.models.workflows import WfRunSignalKind
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from aios.workflows.sweep import wake_runs_needing_step

pytestmark = pytest.mark.integration

AGENT_DEADLINE = 3600.0
TOOL_STALE = 60.0


@pytest.fixture
async def sweep_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_sw', NULL, TRUE, 'sweep-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_sw', 'sweep-env', '{}'::jsonb, 'acc_sw')"
            )
        yield pool
    finally:
        await pool.close()


def test_wake_workflow_task_registered_on_workflows_queue() -> None:
    import aios.harness.tasks  # noqa: F401  — importing registers the @app.task
    from aios.harness.procrastinate_app import app

    assert app.tasks["harness.wake_workflow"].queue == "workflows"


async def _make_run(pool: asyncpg.Pool[Any], *, status: str = "suspended") -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_sw",
            name=f"w-{secrets.token_hex(4)}",  # (account_id, name) is unique
            script="async def main(input):\n    return 1",
        )
        run = await wf_queries.insert_wf_run(
            conn,
            account_id="acc_sw",
            workflow_id=wf.id,
            environment_id="env_sw",
            script="x",
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            depth=10,  # #1124: root-budget seed for a directly-inserted run
            script_sha="x",
        )
        if status != "pending":
            await wf_queries.set_run_status(conn, run.id, status, account_id="acc_sw")
    return run.id


async def _call_started(
    pool: asyncpg.Pool[Any], run_id: str, call_key: str, capability: str, *, age_seconds: float = 0
) -> None:
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn,
            account_id="acc_sw",
            run_id=run_id,
            type="call_started",
            call_key=call_key,
            payload={"capability": capability},
        )
        if age_seconds:
            await conn.execute(
                "UPDATE wf_run_events SET created_at = now() - make_interval(secs => $1) "
                "WHERE run_id = $2 AND call_key = $3 AND type = 'call_started'",
                age_seconds,
                run_id,
                call_key,
            )


async def _call_result(pool: asyncpg.Pool[Any], run_id: str, call_key: str) -> None:
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn,
            account_id="acc_sw",
            run_id=run_id,
            type="call_result",
            call_key=call_key,
            payload={"result": "v", "is_error": False},
        )


async def _signal(
    pool: asyncpg.Pool[Any], run_id: str, call_key: str, kind: WfRunSignalKind
) -> None:
    async with pool.acquire() as conn:
        await wf_queries.insert_run_signal(conn, run_id=run_id, call_key=call_key, kind=kind)


async def _needing(pool: asyncpg.Pool[Any]) -> set[str]:
    async with pool.acquire() as conn:
        ids = await wf_queries.list_run_ids_needing_step(
            conn, agent_deadline_seconds=AGENT_DEADLINE, tool_stale_seconds=TOOL_STALE
        )
    return set(ids)


async def test_status_clauses(sweep_pool: asyncpg.Pool[Any]) -> None:
    """pending (seeded, never stepped — the locked MUST) and running (the per-step
    lease: any mid-step crash) are ALWAYS swept; a bare parked run and terminals
    never are."""
    pool = sweep_pool
    runs = {s: await _make_run(pool, status=s) for s in ("pending", "running", "suspended")}
    for s in ("completed", "errored", "cancelled"):
        runs[s] = await _make_run(pool, status=s)
    assert await _needing(pool) == {runs["pending"], runs["running"]}


@pytest.mark.parametrize("kind", ["gate_resume", "child_done", "tool_result", "cancel"])
async def test_unharvested_signal_wakes(sweep_pool: asyncpg.Pool[Any], kind: str) -> None:
    """Every signal kind is a wake source until a matching call_result exists —
    a lost defer_run_wake from any resume/completion path self-heals in one tick."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    await _signal(pool, run_id, f"sha:{kind}#0", kind)  # type: ignore[arg-type]
    assert run_id in await _needing(pool)
    # Harvested (call_result journaled) → the signal row is inert; back to quiet.
    await _call_result(pool, run_id, f"sha:{kind}#0")
    assert run_id not in await _needing(pool)


async def test_inflight_agent_wakes_only_past_deadline(sweep_pool: asyncpg.Pool[Any]) -> None:
    """A run waiting on a live agent child is QUIET (the child's completion wakes
    it); past the wall-clock deadline the sweep must wake it so the step's harvest
    force-resolves the timeout — this clause DRIVES the H3 backstop."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    await _call_started(pool, run_id, "sha:a#0", "agent")
    assert run_id not in await _needing(pool)
    await _call_started(pool, run_id, "sha:a#1", "agent", age_seconds=AGENT_DEADLINE + 60)
    assert run_id in await _needing(pool)


async def test_inflight_tool_wakes_past_redispatch_horizon(
    sweep_pool: asyncpg.Pool[Any],
) -> None:
    """A tool task that crashed without a signal leaves only its call_started; the
    stale-tool clause re-wakes the run so the harvest re-dispatches (idempotent)."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    await _call_started(pool, run_id, "sha:t#0", "tool")
    assert run_id not in await _needing(pool)
    await _call_started(pool, run_id, "sha:t#1", "tool", age_seconds=TOOL_STALE * 2)
    assert run_id in await _needing(pool)
    # Resolved → quiet again.
    await _call_result(pool, run_id, "sha:t#1")
    assert run_id not in await _needing(pool)


async def test_inflight_bash_tool_wakes_past_sandbox_horizon(
    sweep_pool: asyncpg.Pool[Any],
) -> None:
    """bash rides the ``tool`` capability (#988, Option 1), so the SAME ``tool``
    stale-clause is its crash backstop — but widened to the sandbox horizon
    (``_sandbox_redispatch_horizon(bash_default_timeout_seconds)``), which must exceed
    the longest a live bash exec can occupy (the bash ceiling + provisioning slack) so
    the sweep never re-drives a still-running command.

    An inflight bash call (capability='tool') aged BELOW the widened horizon is left
    alone; aged ABOVE it, the stale-tool clause re-wakes the run for re-dispatch."""
    from aios.config import get_settings
    from aios.workflows.sweep import _sandbox_redispatch_horizon

    horizon = _sandbox_redispatch_horizon(get_settings().bash_default_timeout_seconds)

    async def _needing_at_horizon() -> set[str]:
        async with sweep_pool.acquire() as conn:
            ids = await wf_queries.list_run_ids_needing_step(
                conn, agent_deadline_seconds=AGENT_DEADLINE, tool_stale_seconds=horizon
            )
        return set(ids)

    pool = sweep_pool
    run_id = await _make_run(pool)
    # A bash call within the horizon (the default 60s TOOL_STALE would wrongly wake it,
    # but the widened sandbox horizon — ≥300s — leaves a slow-but-alive exec alone).
    await _call_started(pool, run_id, "sha:b#0", "tool", age_seconds=horizon - 30)
    assert run_id not in await _needing_at_horizon()
    # Past the widened horizon: the crashed-exec backstop fires.
    await _call_started(pool, run_id, "sha:b#1", "tool", age_seconds=horizon + 60)
    assert run_id in await _needing_at_horizon()
    # Resolved → quiet again.
    await _call_result(pool, run_id, "sha:b#1")
    # The still-young "sha:b#0" remains under the horizon, so the run is quiet.
    assert run_id not in await _needing_at_horizon()


async def test_signal_clause_correlates_by_call_key(sweep_pool: asyncpg.Pool[Any]) -> None:
    """A harvested call must not quiet an UNRELATED unharvested signal — the
    anti-join correlates on call_key, and dropping that correlation would zombie
    any partially-harvested run (a recall loss, the forbidden fail direction)."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    await _call_started(pool, run_id, "sha:a#0", "gate")
    await _call_result(pool, run_id, "sha:a#0")  # call A harvested...
    await _signal(pool, run_id, "sha:b#0", "gate_resume")  # ...signal B is not
    assert run_id in await _needing(pool)


async def test_signal_clause_correlates_by_run(sweep_pool: asyncpg.Pool[Any]) -> None:
    """call_keys are pure content hashes — IDENTICAL across two runs of the same
    workflow+input — so one run's harvested call must never quiet another run's
    unharvested signal under the same key."""
    pool = sweep_pool
    run1 = await _make_run(pool)
    run2 = await _make_run(pool)
    await _call_result(pool, run1, "sha:k#0")
    await _signal(pool, run2, "sha:k#0", "gate_resume")
    needing = await _needing(pool)
    assert run2 in needing
    assert run1 not in needing


async def test_inflight_gate_is_never_stale(sweep_pool: asyncpg.Pool[Any]) -> None:
    """A gate is resume-driven only: a run parked on one for a day is exactly the
    run the filter exists to leave alone."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    await _call_started(pool, run_id, "sha:g#0", "gate", age_seconds=86_400)
    assert run_id not in await _needing(pool)


async def test_deferred_frontier_does_not_trip_sweep(sweep_pool: asyncpg.Pool[Any]) -> None:
    """A ``frontier_deferred`` marker (a wave-deferred agent frontier, #784) with NO
    ``call_started`` is a WAIT, not work: the child that frees its slot wakes the run,
    so the sweep must leave a deferred-only parked run alone (the stale-inflight clause
    only matches ``call_started``)."""
    pool = sweep_pool
    run_id = await _make_run(pool)
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn,
            account_id="acc_sw",
            run_id=run_id,
            type="frontier_deferred",
            call_key="sha:fd#0",
            payload={"capability": "agent"},
        )
    assert run_id not in await _needing(pool)


async def test_archived_run_is_never_swept(sweep_pool: asyncpg.Pool[Any]) -> None:
    pool = sweep_pool
    run_id = await _make_run(pool, status="pending")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE wf_runs SET archived_at = now() WHERE id = $1", run_id)
    assert run_id not in await _needing(pool)


async def test_wake_runs_needing_step_defers_for_matches(sweep_pool: asyncpg.Pool[Any]) -> None:
    """The sweep entrypoint (settings-bound) defers exactly the filter's matches."""
    pool = sweep_pool
    pending = await _make_run(pool, status="pending")
    parked = await _make_run(pool)  # suspended, nothing new — must stay quiet
    with mock.patch("aios.workflows.sweep.defer_run_wake", new=AsyncMock()) as deferred:
        swept = await wake_runs_needing_step(pool)
    woken = {call.args[0] for call in deferred.call_args_list}
    assert swept == 1 and woken == {pending}
    assert parked not in woken
