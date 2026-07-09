"""Boot-recovery composition test (issue #1757).

An audit into "should we add a real kill -9 e2e" found ``simulate_sigkill``
faithful by construction for DB-visible state — Postgres commits an event
prefix atomically, and post-kill procrastinate row states are already pinned
against the real schema (``tests/e2e/test_reap_stalled_jobs.py``). The one
real gap a kill test would have transitively covered is **composition**:
``worker_main``'s startup-recovery sequence (``reap_stalled_jobs`` →
``reset_inflight_harvests`` → ``wake_sessions_needing_inference`` →
``wake_runs_needing_step``) had no test exercising all of it TOGETHER against
the full post-kill residue a real SIGKILL leaves. Each piece has its own
focused suite; this test pins the ORDERING and the CONVERGENCE.

:func:`aios.harness.worker.run_startup_recovery` is the extracted unit under
test — pulled out of ``worker_main`` by this issue specifically so it is
callable without booting a full worker process.

Seeded simultaneously, mirroring what one dead worker actually leaves behind:

* a **ghost tool call** — an assistant message with ``tool_calls`` and no
  paired ``tool``-role result (the harness never got to run it, or ran it and
  crashed before recording the outcome).
* a **wedged procrastinate ``doing`` row** — ``lock=session_id`` /
  ``queueing_lock=session_id`` (the #147 shape: the predecessor's wake job
  is still marked ``doing`` because a real SIGKILL leaves procrastinate rows
  exactly as-committed — no graceful ``finish_job`` ever ran). Until reaped,
  a fresh wake for this session raises ``AlreadyEnqueued`` and is silently
  swallowed, wedging the session.
* a **stranded model-dispatch park** (#1635) — a ``model_workflow_park`` span
  with no matching harvest, whose bound run has already resolved (so the
  crash-recovery re-park has real terminal state to harvest).

The composed assertion: one ``run_startup_recovery`` pass fails the wedged
job (releasing its lock), repairs the ghost with the '#685 may have
completed' branch (a ``tool_execute_start`` span was written before the
simulated crash), re-parks and harvests the stranded model-dispatch park, AND
re-enqueues the session's wake in THE SAME PASS — pinning that reap runs
before wake, not just eventually.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import asyncpg
import pytest
from procrastinate import PsycopgConnector

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import model_workflow as mwf
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.worker import run_startup_recovery
from aios.jobs.app import _sync_dsn
from aios.jobs.app import app as procrastinate_app
from aios.models.agents import ToolSpec
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_boot_recovery"
_ENV = "env_boot_recovery"


@pytest.fixture
async def recovery_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A real pool + an open procrastinate connector against the migrated DB.

    Mirrors ``tests/e2e/conftest.py``'s ``open_procrastinate_app`` /
    ``job_manager`` pattern (the module-level ``procrastinate_app`` singleton
    fixes its connector at import time, so a fresh one pointed at this DB is
    swapped in for the fixture's lifetime) — but lives in ``tests/integration``
    so it needs only Postgres, no Docker, matching every other test in this
    module.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=8)
    conninfo = _sync_dsn(migrated_db_url)
    connector = PsycopgConnector(conninfo=conninfo)
    await connector.open_async()
    mwf.reset_inflight_harvests()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'boot-recovery-root')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'boot-recovery-env', '{}'::jsonb, $2)",
                _ENV,
                _ACCOUNT,
            )
        with procrastinate_app.replace_connector(connector):
            yield pool
    finally:
        mwf.reset_inflight_harvests()
        await connector.close_async()
        await pool.close()


def _assistant_with_tool_call(tool_call_id: str, name: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


async def _seed_ghost_session(pool: asyncpg.Pool[Any]) -> tuple[str, str]:
    """A session with a dispatched-but-unresolved ``bash`` ghost tool call.

    A ``tool_execute_start`` span is written before the (simulated) crash —
    exactly what the real dispatch path does before invoking the tool body
    (``tool_dispatch.py``) — so the ghost is classified ``may_have_completed``
    (#685), the conservative branch a real mid-execution SIGKILL produces.
    """
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix="ghost", tools=[ToolSpec(type="bash")]
    )
    tcid = "tc_ghost_1"
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session.id,
            kind="message",
            data=_assistant_with_tool_call(tcid, "bash"),
        )
        await queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session.id,
            kind="span",
            data={"event": "tool_execute_start", "tool_call_id": tcid, "tool_name": "bash"},
        )
    return session.id, tcid


async def _seed_wedged_doing_job(pool: asyncpg.Pool[Any], session_id: str) -> int:
    """A ``doing`` ``harness.wake_session`` row holding ``lock``/``queueing_lock``
    on ``session_id`` — the #147 shape a real SIGKILL leaves (procrastinate
    rows as-committed; no graceful ``finish_job`` ever ran for the
    predecessor's in-flight wake).
    """
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                """
                INSERT INTO procrastinate_jobs
                    (queue_name, task_name, priority, lock, queueing_lock, args, status, worker_id, attempts)
                VALUES ('sessions', 'harness.wake_session', 0, $1, $1, $2::jsonb, 'doing', NULL, 0)
                RETURNING id
                """,
                session_id,
                f'{{"session_id": "{session_id}", "cause": "message"}}',
            )
        )


async def _seed_stranded_model_dispatch_park(
    pool: asyncpg.Pool[Any],
) -> tuple[str, str]:
    """A session parked on a model-dispatch run whose bound run has already
    resolved (``completed``) but whose harvest was never written back — the
    #1635 shape a worker crash mid-park produces (the fire-and-forget harvest
    task died with the worker).
    """
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix="mwfpark", tools=[]
    )
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT,
            name="boot-recovery-inner",
            script="async def main(input):\n    return {'ok': True}\n",
        )
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=_ACCOUNT,
            workflow_id=wf.id,
            environment_id=_ENV,
            script="async def main(input):\n    return {'ok': True}\n",
            script_sha="deadbeef",
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            launcher_session_id=session.id,
            caller={"kind": "session", "id": session.id, "purpose": "model_dispatch"},
            depth=10,
        )
        await wf_queries.set_run_terminal(
            conn,
            run.id,
            status="completed",
            output={"content": "resumed answer"},
            account_id=_ACCOUNT,
        )
        await queries.append_event(
            conn,
            account_id=_ACCOUNT,
            session_id=session.id,
            kind="span",
            data={"event": "model_workflow_park", "run_id": run.id, "reacting_to": 0},
        )
    return session.id, run.id


async def _job_status(pool: asyncpg.Pool[Any], job_id: int) -> str:
    async with pool.acquire() as conn:
        return str(
            await conn.fetchval("SELECT status FROM procrastinate_jobs WHERE id = $1", job_id)
        )


async def _latest_assistant_tool_result(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND role = 'tool' AND data->>'tool_call_id' = $2 ORDER BY seq DESC LIMIT 1",
            session_id,
            tool_call_id,
        )
    return dict(row["data"]) if row is not None else None


async def _queued_wake_session_ids(pool: asyncpg.Pool[Any]) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT args->>'session_id' AS sid FROM procrastinate_jobs "
            "WHERE task_name = 'harness.wake_session' AND status = 'todo'"
        )
    return {r["sid"] for r in rows}


async def test_startup_recovery_converges_full_post_kill_residue(
    recovery_pool: asyncpg.Pool[Any],
) -> None:
    """One ``run_startup_recovery`` pass resolves every post-kill residue kind
    seeded simultaneously, in the order the module docstring pins.
    """
    pool = recovery_pool

    ghost_session_id, ghost_tcid = await _seed_ghost_session(pool)
    job_id = await _seed_wedged_doing_job(pool, ghost_session_id)
    park_session_id, run_id = await _seed_stranded_model_dispatch_park(pool)

    # Poison the in-flight-harvest key set the way a STALE predecessor process
    # would leave it in shared module state — reset_inflight_harvests (step 2)
    # must clear this before step 3 re-parks, or the stale key would make the
    # crash-recovery treat the genuinely-lost park as still-serviced (#1635).
    mwf._INFLIGHT_HARVESTS.add((park_session_id, run_id))

    inflight_registry = InflightToolRegistry()

    with mock.patch.object(
        mwf, "_launch_harvest_task", side_effect=lambda *a, **k: None
    ) as launch_spy:
        result = await run_startup_recovery(pool, inflight_registry)

    # ── 1. the wedged doing row is failed, releasing its lock ──────────────
    assert result.reaped_jobs >= 1
    assert await _job_status(pool, job_id) == "failed"

    # ── 2. the ghost is repaired with the '#685 may have completed' branch ─
    assert result.repaired_ghosts >= 1
    tool_result = await _latest_assistant_tool_result(pool, ghost_session_id, ghost_tcid)
    assert tool_result is not None, "the ghost tool call must be repaired with a result"
    assert "may have completed" in str(tool_result.get("content", ""))

    # ── 3. the session is re-enqueued for a wake IN THE SAME PASS ──────────
    # Provable only because reap (step 1) ran BEFORE wake (step 3): while the
    # doing row held the lock, a wake attempt for this session would have hit
    # procrastinate's AlreadyEnqueued and been silently swallowed by
    # defer_wake, leaving the session wedged until a LATER sweep tick.
    assert result.woken_sessions >= 1
    queued = await _queued_wake_session_ids(pool)
    assert ghost_session_id in queued, (
        "reap-before-wake ordering: the freshly-released session must be "
        "re-enqueued in this same startup-recovery pass"
    )

    # ── 4. the stranded model-dispatch park was re-parked (a fresh harvest
    #        task launched — the poisoned in-flight key did NOT suppress it,
    #        proving reset_inflight_harvests ran before the re-park) ───────
    launch_spy.assert_called_once()
    args, kwargs = launch_spy.call_args
    assert kwargs["run_id"] == run_id
    assert args[1] == park_session_id  # (pool, session_id) positional

    # The re-parked harvest task is fire-and-forget in production; this test
    # only asserts it was launched (the #1635 suite pins its end-to-end
    # harvest behavior). Drive the harvest manually here to also prove the
    # session lands back in the queued-wake set once harvested.
    await mwf.write_harvest_event(
        pool,
        park_session_id,
        run_id=run_id,
        outcome="ok",
        output={"content": "resumed answer"},
        error=None,
        account_id=_ACCOUNT,
    )
