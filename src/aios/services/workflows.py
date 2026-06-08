"""Service layer for the workflows HTTP/CLI surface (Block 3).

The router imports this as ``from aios.services import workflows as service``,
matching every other resource router. It is the account-scoped, pool-taking face
over the query layer (``db.queries.workflows``) and the runtime entry points.

``create_run`` / ``resume_gate`` (the runtime entry points) live in
``aios.workflows.service`` — re-exported here so callers have one import, and kept
there because the integration tests patch ``aios.workflows.service.defer_run_wake``.
The HTTP path adds account-scoping (the runtime variants are internal/trusted) and
the **by-nonce** gate resume.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.models.workflows import WfRun, WfRunEvent, Workflow
from aios.services.wake import defer_run_wake
from aios.workflows.service import create_run, resume_gate

__all__ = [
    "create_run",
    "create_workflow",
    "get_run",
    "get_workflow",
    "list_run_events",
    "list_runs",
    "list_workflows",
    "resume_gate",
    "resume_gate_by_nonce",
]


# ─── workflow definitions ────────────────────────────────────────────────────


async def create_workflow(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    name: str,
    script: str,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Workflow:
    async with pool.acquire() as conn:
        return await wf_queries.insert_workflow(
            conn,
            account_id=account_id,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
        )


async def get_workflow(pool: asyncpg.Pool[Any], workflow_id: str, *, account_id: str) -> Workflow:
    async with pool.acquire() as conn:
        return await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)


async def list_workflows(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Workflow]:
    async with pool.acquire() as conn:
        return await wf_queries.list_workflows(
            conn, account_id=account_id, limit=limit, after=after, name=name
        )


# ─── runs ────────────────────────────────────────────────────────────────────


async def get_run(pool: asyncpg.Pool[Any], run_id: str, *, account_id: str) -> WfRun:
    async with pool.acquire() as conn:
        return await wf_queries.get_wf_run(conn, run_id, account_id=account_id)


async def list_runs(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    workflow_id: str | None = None,
    status: str | None = None,
) -> list[WfRun]:
    async with pool.acquire() as conn:
        return await wf_queries.list_wf_runs(
            conn,
            account_id=account_id,
            limit=limit,
            after=after,
            workflow_id=workflow_id,
            status=status,
        )


async def list_run_events(
    pool: asyncpg.Pool[Any],
    run_id: str,
    *,
    account_id: str,
    after_seq: int = 0,
    limit: int = 200,
) -> list[WfRunEvent]:
    async with pool.acquire() as conn:
        return await wf_queries.list_run_events_scoped(
            conn, run_id, account_id=account_id, after_seq=after_seq, limit=limit
        )


async def resume_gate_by_nonce(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    account_id: str,
    gate_nonce: str,
    result: Any,
) -> None:
    """Deliver an external resume to the gate identified by its capability ``nonce``.

    The HTTP-facing gate resume: account-scope the run first (the internal
    :func:`aios.workflows.service.resume_gate` keys off ``call_key`` and does NOT
    scope, so it must never be the HTTP path), then resolve ``gate_nonce`` →
    ``call_key`` by scanning the run's **open** gate ``call_started`` events for the
    one whose payload carries that nonce. Only an OPEN gate (no ``call_result`` yet)
    matches — a nonce for an already-resolved gate (or any gate on a terminal run)
    raises ``NotFoundError`` rather than writing an orphaned signal nothing harvests.
    ``insert_run_signal`` is idempotent, so a concurrent double-resume of a
    still-open gate is a no-op.
    """
    async with pool.acquire() as conn:
        await wf_queries.get_wf_run(conn, run_id, account_id=account_id)  # 404s cross-tenant
        events = await wf_queries.list_run_events(conn, run_id)
        resolved = {e.call_key for e in events if e.type == "call_result"}
        call_key = None
        for event in events:
            if (
                event.type == "call_started"
                and event.payload.get("capability") == "gate"
                and event.payload.get("gate_nonce") == gate_nonce
                and event.call_key is not None
                and event.call_key not in resolved
            ):
                call_key = event.call_key
                break
        if call_key is None:
            raise NotFoundError(
                "no open gate matches that nonce on this run", detail={"run_id": run_id}
            )
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)
