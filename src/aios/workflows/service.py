"""Service-layer entry points for the workflows runtime (Block 1, internal).

No HTTP/CLI surface yet — these are the functions tests and (later) the API layer
call to create and resume runs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.models.workflows import WfRun
from aios.services.wake import defer_run_wake


async def create_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    workflow_id: str,
    environment_id: str,
    input: Any = None,
) -> WfRun:
    """Create a run that snapshots the workflow's current script, then wake it.

    The run carries its own immutable ``script`` (+ ``script_sha``), so every
    wake execs exactly that source regardless of later edits to the workflow.
    ``environment_id`` binds the run (and the sessions its ``agent()`` children
    spawn into) to an environment — chosen at run-creation time, like a session's.
    """
    async with pool.acquire() as conn:
        workflow = await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)
        script_sha = hashlib.sha256(workflow.script.encode("utf-8")).hexdigest()
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=workflow_id,
            environment_id=environment_id,
            script=workflow.script,
            script_sha=script_sha,
            input=input,
        )
    await defer_run_wake(run.id)
    return run


async def resume_gate(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    call_key: str,
    result: Any,
) -> None:
    """Deliver an external resume for a suspended gate.

    The Block-1 internal stand-in for the (deferred) HTTP resume endpoint: record
    a durable ``wf_run_signals`` row and defer a wake. The run's next step harvests
    the signal into the journal — keeping ``wf_run_events`` single-writer.
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        if run is None:
            raise NotFoundError(f"workflow run {run_id} not found", detail={"id": run_id})
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)
