"""Workflow + run HTTP endpoints (Block 3 surface).

Two routers in one module: ``/v1/workflows`` (the definition resource, mirroring
``/v1/agents``) and the top-level ``/v1/runs`` (the execution instances, mirroring
``/v1/sessions`` — a run carries ``workflow_id`` the way a session carries
``agent_id``). Both are account-scoped via ``AccountIdDep``; the SSE
``/v1/runs/{id}/stream`` endpoint is added in the streaming slice.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from sse_starlette import EventSourceResponse

from aios.api.deps import AccountIdDep, DbUrlDep, PoolDep
from aios.api.sse import make_sse_response, preflight_subscription, wf_run_event_stream
from aios.db.listen import open_listen_for_run_events
from aios.logging import get_logger
from aios.models.common import ListResponse
from aios.models.pagination import (
    MAX_EVENT_PAGE_LIMIT,
    EventPageLimit,
    PageLimit,
    cursor_as_int,
    page_cursor,
    resolve_page_limit,
)
from aios.models.trace import TraceResponse
from aios.models.workflows import (
    GateResume,
    WfRun,
    WfRunCreate,
    WfRunEvent,
    WfRunWaitResponse,
    Workflow,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowVersion,
)
from aios.services import trace as trace_service
from aios.services import workflows as service

log = get_logger("aios.api.routers.workflows")

router = APIRouter(prefix="/v1/workflows", tags=["workflows"])
runs_router = APIRouter(prefix="/v1/runs", tags=["runs"])


# ─── /v1/workflows (definitions) ─────────────────────────────────────────────


@router.post("", operation_id="create_workflow", status_code=status.HTTP_201_CREATED)
async def create_workflow(
    body: WorkflowCreate, pool: PoolDep, account_id: AccountIdDep
) -> Workflow:
    """Create a workflow definition (version 1).

    The HTTP path is unattenuated operator authority — no ``creator_session_id``, so the
    declared tool surface is not subset-checked against any agent (an agent authoring a
    workflow goes through the create-time-attenuated builtin, a later slice)."""
    return await service.create_workflow(
        pool,
        account_id=account_id,
        name=body.name,
        script=body.script,
        input_schema=body.input_schema,
        output_schema=body.output_schema,
        description=body.description,
        tools=body.tools,
        mcp_servers=body.mcp_servers,
        http_servers=body.http_servers,
    )


@router.put("/{workflow_id}", operation_id="update_workflow")
async def update_workflow(
    workflow_id: str, body: WorkflowUpdate, pool: PoolDep, account_id: AccountIdDep
) -> Workflow:
    """Update a workflow in place, bumping ``version``.

    ``body.version`` must match the current version (optimistic concurrency — 409 on a
    stale token; re-fetch and retry). Omitted fields are preserved; an identical update
    is a no-op. In-flight runs are unaffected (a run snapshots script + surface at
    launch). The HTTP path is unattenuated operator authority — no ``actor_session_id``."""
    return await service.update_workflow(
        pool,
        workflow_id,
        account_id=account_id,
        expected_version=body.version,
        name=body.name,
        script=body.script,
        input_schema=body.input_schema,
        output_schema=body.output_schema,
        description=body.description,
        tools=body.tools,
        mcp_servers=body.mcp_servers,
        http_servers=body.http_servers,
    )


@router.get("", operation_id="list_workflows")
async def list_workflows(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    name: str | None = None,
    limit: PageLimit = None,
) -> ListResponse[Workflow]:
    """List the account's workflows, newest first. First page: optional ``name`` +
    ``limit``; subsequent pages: ``?cursor=<next_cursor>``."""
    st = page_cursor(cursor, {"name": name, "limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = resolve_page_limit(st, limit)
    if st is not None:
        name = st.filters.get("name")
    items = await service.list_workflows(
        pool, account_id=account_id, limit=page_limit + 1, after=after, name=name
    )
    return ListResponse[Workflow].paginate(
        items, page_limit, cursor=lambda x: x.id, filters={"name": name}
    )


@router.get("/{workflow_id}", operation_id="get_workflow")
async def get_workflow(workflow_id: str, pool: PoolDep, account_id: AccountIdDep) -> Workflow:
    """Fetch one workflow definition by id."""
    return await service.get_workflow(pool, workflow_id, account_id=account_id)


@router.post(
    "/{workflow_id}/archive",
    operation_id="archive_workflow",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive_workflow(workflow_id: str, pool: PoolDep, account_id: AccountIdDep) -> Workflow:
    return await service.archive_workflow(pool, workflow_id, account_id=account_id)


@router.post("/{workflow_id}/unarchive", operation_id="unarchive_workflow")
async def unarchive_workflow(workflow_id: str, pool: PoolDep, account_id: AccountIdDep) -> Workflow:
    return await service.unarchive_workflow(pool, workflow_id, account_id=account_id)


@router.get("/{workflow_id}/versions", operation_id="list_workflow_versions")
async def list_workflow_versions(
    workflow_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    limit: PageLimit = None,
) -> ListResponse[WorkflowVersion]:
    """List a workflow's immutable definition history, newest first.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Each
    version is a complete snapshot of the workflow's definition at the time it
    was written (``name`` is versioned — a rename mints a new version). An
    archived workflow's versions remain readable (post-mortem audit).
    """
    st = page_cursor(cursor, {"limit": limit})
    after = cursor_as_int(st.cursor) if st is not None else None
    page_limit = resolve_page_limit(st, limit)
    items = await service.list_workflow_versions(
        pool, workflow_id, limit=page_limit + 1, after=after, account_id=account_id
    )
    return ListResponse[WorkflowVersion].paginate(items, page_limit, cursor=lambda x: x.version)


@router.get("/{workflow_id}/versions/{version}", operation_id="get_workflow_version")
async def get_workflow_version(
    workflow_id: str, version: int, pool: PoolDep, account_id: AccountIdDep
) -> WorkflowVersion:
    """Fetch one historical version's definition snapshot.

    The snapshot reflects the workflow's definition at the time the version was
    written and is unaffected by subsequent updates or archival.
    """
    return await service.get_workflow_version(pool, workflow_id, version, account_id=account_id)


# ─── /v1/runs (execution instances) ──────────────────────────────────────────


@runs_router.post("", operation_id="create_run", status_code=status.HTTP_201_CREATED)
async def create_run(body: WfRunCreate, pool: PoolDep, account_id: AccountIdDep) -> WfRun:
    """Launch a run of a workflow. Snapshots the workflow's current script, binds
    the run to ``environment_id`` (its ``agent()`` children spawn there) and to
    ``vault_ids`` (credentials it resolves at tool-call time), and wakes it. A missing
    workflow or environment 404s. The HTTP path is unattenuated operator authority — no
    ``launcher_session_id``, so the requested vaults are bound as-is (account-scoped)."""
    return await service.create_run(
        pool,
        account_id=account_id,
        workflow_id=body.workflow_id,
        environment_id=body.environment_id,
        input=body.input,
        vault_ids=body.vault_ids,
        budget_usd=body.budget_usd,
        default_child_model=body.default_child_model,
    )


@runs_router.get("", operation_id="list_runs")
async def list_runs(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    workflow_id: str | None = None,
    status: str | None = None,
    parent_run_id: str | None = None,
    limit: PageLimit = None,
) -> ListResponse[WfRun]:
    """List the account's runs, newest first. First page: optional ``workflow_id`` /
    ``status`` / ``parent_run_id`` filters + ``limit``; subsequent pages:
    ``?cursor=<next_cursor>``. ``parent_run_id`` scopes to a run's child runs."""
    st = page_cursor(
        cursor,
        {
            "workflow_id": workflow_id,
            "status": status,
            "parent_run_id": parent_run_id,
            "limit": limit,
        },
    )
    after = str(st.cursor) if st is not None else None
    page_limit = resolve_page_limit(st, limit)
    if st is not None:
        workflow_id = st.filters.get("workflow_id")
        status = st.filters.get("status")
        parent_run_id = st.filters.get("parent_run_id")
    items = await service.list_runs(
        pool,
        account_id=account_id,
        limit=page_limit + 1,
        after=after,
        workflow_id=workflow_id,
        status=status,
        parent_run_id=parent_run_id,
    )
    return ListResponse[WfRun].paginate(
        items,
        page_limit,
        cursor=lambda x: x.id,
        filters={"workflow_id": workflow_id, "status": status, "parent_run_id": parent_run_id},
    )


@runs_router.get("/{run_id}", operation_id="get_run")
async def get_run(run_id: str, pool: PoolDep, account_id: AccountIdDep) -> WfRun:
    """Fetch one run by id (status, output, last_event_seq, …)."""
    return await service.get_run(pool, run_id, account_id=account_id)


@runs_router.get("/{run_id}/wait", operation_id="await_run")
async def await_run(
    run_id: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
    timeout_seconds: Annotated[int, Query(alias="timeout", ge=0, le=60)] = 30,
) -> WfRunWaitResponse:
    """Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The ``await``-a-completion primitive (runs backing): one JSON round-trip returning the
    completion record — ``done`` + ``output``, or ``is_error`` + ``error``. A run still running
    after ``timeout`` seconds returns ``done=false`` with its current status; call again to keep
    blocking. Unlike the SSE ``/stream`` this is a plain request/response, so it works as an MCP
    tool — an agent can await a sub-run and join. A cross-tenant run 404s.

    Watch the right field (#1140): poll ``done`` (bool) or ``run_status`` — the
    response has NO ``state`` field, so a watcher keying on ``.state`` reads
    ``None`` forever even after ``output`` is populated.
    """
    return await service.await_run(
        pool, db_url, run_id, account_id=account_id, timeout_seconds=timeout_seconds
    )


@runs_router.get("/{run_id}/events", operation_id="list_run_events")
async def list_run_events(
    run_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    limit: EventPageLimit = None,
) -> ListResponse[WfRunEvent]:
    """A run's journal by sequence (oldest first). First page: optional ``limit``;
    subsequent pages: ``?cursor=<next_cursor>``.

    Transient-empty (#1140): an empty ``items`` list is NOT a "run reset" — it
    only means no journal rows past this ``seq`` yet. Page by ``seq`` and treat
    an empty page as "nothing new yet."

    Schema (#1140): each item is a *run* event ``{type, payload, seq}`` — a
    DIFFERENT shape from a child-*session* event (``{kind, data}`` on
    ``/v1/sessions/{id}/events``). See ``docs/reference/run-observability.md``.
    """
    # Scope check: 404 a cross-tenant run id before reading its journal.
    await service.get_run(pool, run_id, account_id=account_id)
    st = page_cursor(cursor, {"limit": limit})
    page_limit = resolve_page_limit(st, limit, default=200, maximum=MAX_EVENT_PAGE_LIMIT)
    after_seq = cursor_as_int(st.cursor) if st is not None else 0
    items = await service.list_run_events(
        pool, run_id, account_id=account_id, after_seq=after_seq, limit=page_limit + 1
    )
    # Forward (ascending seq) read — label the cursor accordingly (paginate defaults
    # to "backward", which is right only for the id-DESC list endpoints).
    return ListResponse[WfRunEvent].paginate(
        items, page_limit, cursor=lambda x: x.seq, direction="forward"
    )


@runs_router.get("/{run_id}/trace", operation_id="get_run_trace")
async def get_run_trace(
    run_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    verbose: bool = False,
) -> TraceResponse:
    """One-call linear trace of a run + all nested sessions and sub-runs (#1149).

    A read-projection over the invocation-edge tree: walks the parent→child edge
    tree from this run, normalizes each node's outcome to
    ``terminal_state ∈ {ok,errored,cancelled,suspended,running}`` (+ raw
    ``error_kind``), and interleaves the nodes' journals into a flat
    **DFS-pre-order** list (a CLI renders ``depth`` as indentation). A
    deliberately-failed nested session's death cause is visible at a glance — the
    node carries ``terminal_state`` + ``error_kind`` (e.g. ``no_return``) and the
    abbreviated default co-locates the proximate ``is_error`` frame.

    ``verbose=false`` (default) keeps only the load-bearing frames per node
    (request/response/turn lifecycle, gates, errors); ``verbose=true`` lifts the
    filter to the full per-node firehose. The walk runs in one ``REPEATABLE
    READ`` snapshot with a node-count ceiling; a tree past the ceiling returns a
    typed ``truncated: {at_nodes}`` marker.

    Ordering caveat: cross-subtree time-ordering is best-effort to transaction
    granularity; only the causal parent→child edge is exact. ``wake_session``
    peer-pokes are out of scope (a peer stimulus, not a spawn). A cross-tenant
    run 404s.
    """
    # Scope check: 404 a cross-tenant run id before walking its tree.
    await service.get_run(pool, run_id, account_id=account_id)
    return await trace_service.get_trace(
        pool, root_kind="run", root_id=run_id, account_id=account_id, verbose=verbose
    )


@runs_router.post("/{run_id}/resume", operation_id="resume_gate")
async def resume_gate(
    run_id: str, body: GateResume, pool: PoolDep, account_id: AccountIdDep
) -> WfRun:
    """Resume a suspended gate by its ``gate_nonce``, delivering ``result``. Returns
    the run (still ``suspended`` here — the recorded resume signal is harvested on the
    next wake, which replays past the gate). A nonce that matches no OPEN gate, or a
    cross-tenant run, 404s."""
    await service.resume_gate_by_nonce(
        pool, run_id=run_id, account_id=account_id, gate_nonce=body.gate_nonce, result=body.result
    )
    return await service.get_run(pool, run_id, account_id=account_id)


@runs_router.post("/{run_id}/cancel", operation_id="cancel_run")
async def cancel_run(run_id: str, pool: PoolDep, account_id: AccountIdDep) -> WfRun:
    """Cancel a run (pending/running/suspended). Records a cancel marker + wakes the
    run; it finalizes ``cancelled`` on its next wake (so the returned run may still
    show its pre-cancel status — like ``resume``). Idempotent on an already-terminal
    run; a cross-tenant run 404s."""
    return await service.cancel_run(pool, run_id=run_id, account_id=account_id)


@runs_router.get("/{run_id}/stream", openapi_extra={"x-codegen": {"targets": []}})
async def stream_run_events(
    run_id: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
    after_seq: int = 0,
) -> EventSourceResponse:
    """Stream a run's journal as Server-Sent Events: backfill from ``after_seq``,
    then live, ending on ``run_completed``.

    Preflights the LISTEN before constructing the response (issue #376), so a
    transient connect failure is a clean 503 rather than a half-open stream.
    """
    await service.get_run(pool, run_id, account_id=account_id)  # 404s cross-tenant
    subscription = await preflight_subscription(
        open_listen_for_run_events(db_url, run_id),
        stream_name="wf_run_events",
        log_key="sse.wf_run_events.preflight_failed",
        log_fields={"run_id": run_id},
        log=log,
    )
    return make_sse_response(
        subscription, wf_run_event_stream(subscription, pool, run_id, after_seq=after_seq)
    )
