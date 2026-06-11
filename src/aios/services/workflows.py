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

from aios.db.listen import open_listen_for_run_events
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec
from aios.models.attenuation import Surface, surface_diff, surface_of
from aios.models.workflows import (
    TERMINAL_RUN_STATUSES,
    WfRun,
    WfRunEvent,
    WfRunWaitResponse,
    Workflow,
)
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services import sessions as sessions_service
from aios.services.await_completion import await_completion
from aios.services.wake import defer_run_wake
from aios.workflows.service import create_run, resume_gate

__all__ = [
    "await_run",
    "cancel_run",
    "create_run",
    "create_workflow",
    "get_run",
    "get_workflow",
    "list_run_events",
    "list_runs",
    "list_workflows",
    "resume_gate",
    "resume_gate_by_nonce",
    "update_workflow",
]


# ─── workflow definitions ────────────────────────────────────────────────────


async def _enforce_surface_attenuation(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    actor_session_id: str,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerSpec],
) -> Surface:
    """Raise ``ForbiddenError`` unless the declared surface is admissible against the acting
    agent's; return the effective (clamped) surface for storage.

    Serves both create (the creator's declared surface) and update (the editor's merged
    final surface). The predicate is per-dimension:

    * tools / mcp_servers — byte-equality of ``clamp(declared, actor)`` against
      ``canonicalize(declared)``: equal iff the meet did not narrow anything, which is
      exactly "``declared`` ≤ ``actor``" on membership *and* per-tool permission/transport
      and MCP server identity (joint name+url).
    * http_servers — identity survival: every declared ``(name, base_url)`` must appear in
      the clamp's http servers. The agent's routes/fields are inherited launcher-frozen
      into storage (the returned ``effective``), so a workflow need only name the server,
      not reproduce its routes byte-perfect.

    http servers remain parent-wins-frozen at run-time (run-launch clamps to
    launcher-verbatim) — only the AUTHORING gate relaxes to identity, which grants no new
    run-time authority. The HTTP/operator path passes no actor and skips this entirely.
    """
    session = await sessions_service.get_session_basic(
        pool, actor_session_id, account_id=account_id
    )
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    declared = Surface(tools, mcp_servers, http_servers)
    expected = attenuation_service.normalize(declared)
    effective = attenuation_service.clamp(declared, surface_of(agent))
    surviving_http = {(s.name, s.base_url) for s in effective.http_servers}
    exceeds = (
        effective.tools != expected.tools
        or effective.mcp_servers != expected.mcp_servers
        or any((s.name, s.base_url) not in surviving_http for s in expected.http_servers)
    )
    if exceeds:
        raise ForbiddenError(
            "workflow surface exceeds the acting agent's permissions",
            detail={"exceeds": surface_diff(expected, effective)},
        )
    return effective


async def create_workflow(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    name: str,
    script: str,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    creator_session_id: str | None = None,
) -> Workflow:
    """Create a workflow definition.

    **Create-time attenuation:** when ``creator_session_id`` is set (an agent authoring
    the workflow), the declared surface (``tools``/``mcp_servers``/``http_servers``) must
    be a subset of the creating agent's own — an agent cannot grant a workflow a tool or
    server it does not itself have; a breach raises :class:`ForbiddenError`. http servers
    are admitted by identity (name + base_url) and their routes inherited launcher-frozen
    into storage. With no creator (the HTTP/operator path) any surface may be declared
    verbatim, account-scoped.
    """
    effective: Surface | None = None
    if creator_session_id is not None:
        effective = await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=creator_session_id,
            tools=tools or [],
            mcp_servers=mcp_servers or [],
            http_servers=http_servers or [],
        )
    # Agent-authored: store the agent's launcher-frozen http routes (inherited by identity).
    # Operator path: store the declared http servers verbatim.
    http_to_store = effective.http_servers if effective is not None else http_servers
    async with pool.acquire() as conn:
        return await wf_queries.insert_workflow(
            conn,
            account_id=account_id,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_to_store,
        )


async def update_workflow(
    pool: asyncpg.Pool[Any],
    workflow_id: str,
    *,
    account_id: str,
    expected_version: int,
    name: str | None = None,
    script: str | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    actor_session_id: str | None = None,
) -> Workflow:
    """Update a workflow in place (optimistic concurrency on ``expected_version``).

    **Update-time attenuation:** when ``actor_session_id`` is set (an agent editing the
    workflow), the **merged final surface** must be a subset of the acting agent's own —
    checked even when the update doesn't touch the surface fields, because an agent that
    rewrites the *script* of a workflow with a broader surface would wield that surface
    through it. An agent may only update a workflow whose resulting surface it could have
    declared itself. With no actor (the HTTP/operator path) anything may be updated.

    In-flight runs are unaffected: a run snapshots ``script`` + the declared surface at
    launch; only runs created after the update see the new definition.

    http servers are admitted by identity and their routes inherited launcher-frozen into
    storage (mirroring create); the operator path stores declared http verbatim.
    """
    effective: Surface | None = None
    if actor_session_id is not None:
        current = await get_workflow(pool, workflow_id, account_id=account_id)
        if current.version != expected_version:
            # Pin the attenuation read to the caller's token. Without this, an actor
            # could send a FUTURE token: attenuation passes against today's surface,
            # a concurrent update broadens it and bumps to exactly that token, and the
            # optimistic UPDATE then matches — landing the actor's script on a surface
            # that was never checked. With the pin, the UPDATE's ``WHERE version``
            # guarantees nothing changed between this read and the write.
            raise ConflictError(
                f"version mismatch: expected {expected_version}, current is {current.version}",
                detail={
                    "expected": expected_version,
                    "current": current.version,
                    "id": workflow_id,
                },
            )
        effective = await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=actor_session_id,
            tools=tools if tools is not None else current.tools,
            mcp_servers=mcp_servers if mcp_servers is not None else current.mcp_servers,
            http_servers=http_servers if http_servers is not None else current.http_servers,
        )
    # Agent-actor touching http: store the inherited launcher-frozen routes. Operator path,
    # or an edit that didn't touch http (``http_servers is None`` → query preserves current):
    # pass the original argument through unchanged.
    http_to_store = (
        effective.http_servers
        if (effective is not None and http_servers is not None)
        else http_servers
    )
    async with pool.acquire() as conn:
        return await wf_queries.update_workflow(
            conn,
            workflow_id,
            account_id=account_id,
            expected_version=expected_version,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_to_store,
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
    parent_run_id: str | None = None,
) -> list[WfRun]:
    async with pool.acquire() as conn:
        return await wf_queries.list_wf_runs(
            conn,
            account_id=account_id,
            limit=limit,
            after=after,
            workflow_id=workflow_id,
            status=status,
            parent_run_id=parent_run_id,
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


async def await_run(
    pool: asyncpg.Pool[Any],
    db_url: str,
    run_id: str,
    *,
    account_id: str,
    timeout_seconds: float,
) -> WfRunWaitResponse:
    """Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The run backing of the ``await``-a-completion primitive. Account-scopes the run FIRST (so a
    cross-tenant/missing ``run_id`` 404s before we open any connection — mirroring ``/stream``,
    and so an unauthorized caller can neither open a LISTEN on a foreign run's channel nor churn
    connections), then subscribes to the run's ``wf_run_events`` channel BEFORE the predicate's
    first status read (LISTEN-before-read: a completion landing between subscribe and read has
    already queued its notify, so it can't be missed), drives :func:`await_completion` with the
    terminal predicate, and returns the completion record — or, on timeout, the run's current
    (non-terminal) status so the caller re-polls.
    """
    await get_run(pool, run_id, account_id=account_id)  # 404s cross-tenant before we subscribe

    async def _read_run() -> WfRun:
        async with pool.acquire() as conn:
            return await wf_queries.get_wf_run(conn, run_id, account_id=account_id)

    subscription = await open_listen_for_run_events(db_url, run_id)
    try:
        run = await await_completion(
            subscription.queue,
            read_state=_read_run,
            is_done=lambda r: r.status in TERMINAL_RUN_STATUSES,
            timeout_seconds=timeout_seconds,
        )
    finally:
        subscription.terminate()

    done = run.status in TERMINAL_RUN_STATUSES
    is_error = run.status == "errored"
    error: dict[str, Any] | None = None
    if is_error:
        # ``error.kind`` lives only in the run_completed payload, not on the run row.
        async with pool.acquire() as conn:
            error = await wf_queries.resolve_run_error(conn, run_id)
    return WfRunWaitResponse(
        run_status=run.status, done=done, output=run.output, is_error=is_error, error=error
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
        call_key = await wf_queries.find_open_gate_call_key(conn, run_id, gate_nonce=gate_nonce)
        if call_key is None:
            raise NotFoundError(
                "no open gate matches that nonce on this run", detail={"run_id": run_id}
            )
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)


async def cancel_run(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    account_id: str,
    reason: Any = None,
    canceller_session_id: str | None = None,
) -> WfRun:
    """Request cancellation of a run, returning it (still in its pre-cancel status).

    Signal-driven, mirroring :func:`resume_gate_by_nonce`: record a ``cancel``
    side-marker and wake the run; the next ``run_workflow_step`` harvests it under
    the lock and finalizes ``cancelled`` (so the journal keeps a single writer and
    any live ``/stream`` closes on the terminal ``run_completed``). The flip lands
    on the wake, so the returned run still shows its current status — as gate-resume
    returns a still-``suspended`` run.

    **Cancel-time attenuation:** when ``canceller_session_id`` is set (an agent
    cancelling via the builtin), the run must have been launched by that very session
    — you may cancel what you launched, nothing else; a breach raises
    :class:`ForbiddenError`. With no canceller (the HTTP/operator path) any
    account-scoped run may be cancelled.

    Idempotent: an already-terminal run is returned unchanged, with no signal or
    wake. A cross-tenant / missing run 404s (via :func:`get_wf_run`).
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_wf_run(conn, run_id, account_id=account_id)  # 404s cross-tenant
        if canceller_session_id is not None and run.launcher_session_id != canceller_session_id:
            raise ForbiddenError(
                "run was not launched by this session; only the operator can cancel it",
                detail={"run_id": run_id},
            )
        if run.status in TERMINAL_RUN_STATUSES:
            return run  # already terminal — nothing to cancel
        await wf_queries.insert_run_signal(
            conn,
            run_id=run_id,
            call_key=wf_queries.CANCEL_SIGNAL_CALL_KEY,
            kind="cancel",
            result=reason,
        )
    await defer_run_wake(run_id)
    return run
