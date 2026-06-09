"""Service-layer entry points for the workflows runtime (Block 1, internal).

No HTTP/CLI surface yet — these are the functions tests and (later) the API layer
call to create and resume runs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg

from aios.db.queries import get_environment, get_session_vault_ids
from aios.db.queries import workflows as wf_queries
from aios.errors import AiosError, ForbiddenError, NotFoundError
from aios.models.workflows import WfRun
from aios.services.wake import defer_run_wake

# Vertical recursion bound: how deep a chain of runs (a run whose agent() child
# launches a sub-run, and so on) may nest. Mirrors WAKE_SESSION_MAX_DEPTH — bounds
# the strange loop the agent-acting builtins enable. Only the agent path threads a
# parent_run_id, so the operator/HTTP path (parent_run_id=None) is never capped.
WORKFLOW_RUN_MAX_DEPTH = 10


class WorkflowRunDepthExceededError(AiosError):
    """A ``create_run`` would nest deeper than ``WORKFLOW_RUN_MAX_DEPTH``."""

    error_type = "workflow_run_depth_exceeded"
    status_code = 409


async def create_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    workflow_id: str,
    environment_id: str,
    input: Any = None,
    vault_ids: list[str] | None = None,
    launcher_session_id: str | None = None,
    parent_run_id: str | None = None,
) -> WfRun:
    """Create a run that snapshots the workflow's current script, then wake it.

    The run carries its own immutable ``script`` (+ ``script_sha``), so every
    wake execs exactly that source regardless of later edits to the workflow.
    ``environment_id`` binds the run (and the sessions its ``agent()`` children
    spawn into) to an environment — chosen at run-creation time, like a session's.
    It is validated as account-owned (``[security]``: a bare FK would accept another
    tenant's env id and leak its image/env-vars/networking into this run).

    ``vault_ids`` binds credentials to the run (resolved at tool-call time, like a
    session's). **Launch-time attenuation:** when ``launcher_session_id`` is set (an
    agent launching the run), the requested vaults must be a subset of the launcher's
    own — authority never exceeds the invoker, so a breach raises
    :class:`ForbiddenError`. With no launcher (the HTTP/operator path) the requested
    vaults bind as-is, account-scoped. Insert + bind are one transaction, so a breach
    or a bad vault leaves no run row; the wake fires only after commit.

    ``parent_run_id`` records run lineage (an agent inside a run launching a sub-run)
    and feeds the **vertical depth cap**: nesting past ``WORKFLOW_RUN_MAX_DEPTH`` raises
    :class:`WorkflowRunDepthExceededError`. The operator/HTTP path passes none, so it is
    a root run (never capped).
    """
    requested = list(vault_ids or [])
    async with pool.acquire() as conn, conn.transaction():
        await get_environment(conn, environment_id, account_id=account_id)  # 404s foreign/absent
        workflow = await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)
        script_sha = hashlib.sha256(workflow.script.encode("utf-8")).hexdigest()
        if parent_run_id is not None:
            # ``parent_run_id`` is trusted same-account: the only caller that sets it is
            # the builtin, threading the launcher session's own ``parent_run_id`` (set by
            # the run-spawn machinery to a same-account run). The account-scoped ancestor
            # walk relies on that — a foreign parent would resolve to depth 0 and read as
            # a root. If a future path ever lets ``parent_run_id`` be caller-supplied, it
            # must be account-validated here (like ``environment_id`` above).
            parent_depth = await wf_queries.run_ancestor_depth(
                conn, parent_run_id, account_id=account_id
            )
            if parent_depth + 1 > WORKFLOW_RUN_MAX_DEPTH:
                raise WorkflowRunDepthExceededError(
                    f"run nesting would exceed depth {WORKFLOW_RUN_MAX_DEPTH}",
                    detail={"max_depth": WORKFLOW_RUN_MAX_DEPTH, "parent_run_id": parent_run_id},
                )
        if launcher_session_id is not None:
            held = set(
                await get_session_vault_ids(conn, launcher_session_id, account_id=account_id)
            )
            ungranted = [v for v in requested if v not in held]
            if ungranted:
                raise ForbiddenError(
                    "run requested vaults the launching agent does not hold",
                    detail={"ungranted_vault_ids": ungranted},
                )
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=workflow_id,
            environment_id=environment_id,
            parent_run_id=parent_run_id,
            script=workflow.script,
            script_sha=script_sha,
            input=input,
            # Snapshot the declared surface at launch (like script), so a later
            # update_workflow never shifts this run's tool-authority.
            tools=workflow.tools,
            mcp_servers=workflow.mcp_servers,
            http_servers=workflow.http_servers,
        )
        if requested:
            await wf_queries.set_run_vaults(conn, run.id, requested, account_id=account_id)
    await defer_run_wake(run.id)
    return run


async def resume_gate(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    call_key: str,
    result: Any,
) -> None:
    """Deliver an external resume for a suspended gate, keyed by ``call_key``.

    The internal/trusted variant: it keys off ``call_key`` directly and does NOT
    account-scope, so it must never be the HTTP path. The HTTP face is
    :func:`aios.services.workflows.resume_gate_by_nonce`, which scopes the run and
    resolves the public ``gate_nonce`` to a ``call_key`` before calling through here.
    Both record a durable ``wf_run_signals`` row and defer a wake; the run's next
    step harvests the signal into the journal — keeping ``wf_run_events`` single-writer.
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        if run is None:
            raise NotFoundError(f"workflow run {run_id} not found", detail={"id": run_id})
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)
