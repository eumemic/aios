"""Session endpoints: create, list, get sessions, append messages, list/stream events.

Posting a message appends a user-message event and defers a procrastinate
``wake_session`` job that a worker will pick up. The endpoint returns 201
immediately.

Clients that want to watch the agent reply in real time should connect to
``GET /v1/sessions/{id}/stream``, which streams events over SSE backed by
Postgres ``LISTEN``/``NOTIFY``.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Query, status
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    AuthDep,
    CryptoBoxDep,
    DbUrlDep,
    PoolDep,
    ProcrastinateDep,
)
from aios.api.sse import sse_event_stream
from aios.db import queries
from aios.db.listen import SESSION_INTERRUPT_CHANNEL, listen_for_events
from aios.errors import NotFoundError, ValidationError
from aios.harness.wake import defer_wake
from aios.ids import GITHUB_REPOSITORY, split_id
from aios.models.common import ListResponse
from aios.models.events import Event, EventKind
from aios.models.github_repositories import (
    GithubRepositoryResourceEcho,
    GithubRepositoryUpdate,
)
from aios.models.sessions import (
    ContextResponse,
    Session,
    SessionCloneRequest,
    SessionCreate,
    SessionInterruptRequest,
    SessionResourceEcho,
    SessionStatus,
    SessionUpdate,
    SessionUserMessage,
    ToolConfirmationRequest,
    ToolResultRequest,
    WaitResponse,
)
from aios.services import github_repositories as github_repo_service
from aios.services import sessions as service

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", operation_id="create_session", status_code=status.HTTP_201_CREATED)
async def create(
    body: SessionCreate,
    pool: PoolDep,
    procrastinate: ProcrastinateDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> Session:
    session = await service.create_session(
        pool,
        agent_id=body.agent_id,
        environment_id=body.environment_id,
        agent_version=body.agent_version,
        title=body.title,
        metadata=body.metadata,
        vault_ids=body.vault_ids or None,
        resources=body.resources or None,
        crypto_box=crypto_box,
        workspace_path=body.workspace_path,
        env=body.env or None,
    )
    if body.initial_message is not None:
        await service.append_user_message(pool, session.id, body.initial_message)
        await defer_wake(pool, session.id, cause="initial_message")
        session = await service.get_session(pool, session.id)
    return session


@router.get("", operation_id="list_sessions")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    agent_id: str | None = None,
    status_filter: Annotated[
        SessionStatus | None,
        Query(alias="status"),
    ] = None,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Session]:
    items = await service.list_sessions(
        pool, agent_id=agent_id, status=status_filter, limit=limit, after=after
    )
    return ListResponse[Session](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{session_id}", operation_id="get_session")
async def get(session_id: str, pool: PoolDep, _auth: AuthDep) -> Session:
    return await service.get_session(pool, session_id)


@router.put("/{session_id}", operation_id="update_session")
async def update(
    session_id: str,
    body: SessionUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> Session:
    from aios.db.queries import _UNSET

    # Use model_fields_set to distinguish "not provided" from "explicitly null".
    # agent_version=null means "latest" (auto-updating); omitted means "keep current".
    return await service.update_session(
        pool,
        session_id,
        agent_id=body.agent_id,
        agent_version=body.agent_version if "agent_version" in body.model_fields_set else _UNSET,
        title=body.title if "title" in body.model_fields_set else _UNSET,
        metadata=body.metadata,
        vault_ids=body.vault_ids,
        resources=body.resources,
        crypto_box=crypto_box,
    )


@router.get("/{session_id}/resources", operation_id="list_session_resources")
async def list_resources(
    session_id: str, pool: PoolDep, _auth: AuthDep
) -> ListResponse[SessionResourceEcho]:
    """List all resources attached to ``session_id``.

    Returns the type-discriminated union of memory store and github
    repository echoes, ordered by type (memory stores first) and rank.
    Equivalent to reading the ``resources`` field on the full session
    record, but cheaper if you don't need anything else.
    """
    # Reuse get_session for ordering + echo construction; resources are
    # already on the returned record.
    session = await service.get_session(pool, session_id)
    return ListResponse[SessionResourceEcho](data=session.resources)


@router.get("/{session_id}/resources/{resource_id}", operation_id="get_session_resource")
async def get_resource(
    session_id: str, resource_id: str, pool: PoolDep, _auth: AuthDep
) -> GithubRepositoryResourceEcho:
    """Fetch a single resource attached to ``session_id`` by its id.

    v1 only supports ``github_repository`` (id prefix ``ghrepo_``) since
    memory store attachments are keyed by ``(session_id, memory_store_id)``
    and don't have a separate attachment id.
    """
    _require_github_resource_id(resource_id)
    return await github_repo_service.get_resource(pool, session_id, resource_id)


@router.post("/{session_id}/resources/{resource_id}", operation_id="update_session_resource")
async def update_resource(
    session_id: str,
    resource_id: str,
    body: GithubRepositoryUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> GithubRepositoryResourceEcho:
    """Rotate the auth token on a github_repository attachment.

    The bumped ``updated_at`` propagates through the mount snapshot, so
    the next sandbox provision recycles the container and re-clones the
    working tree with the new token.
    """
    _require_github_resource_id(resource_id)
    # Identity is replaced atomically when the caller mentions either
    # field in the payload; absent means preserve the stored values
    # (the common token-only rotation must not silently clear identity).
    fields_set = body.model_fields_set
    identity = (
        (body.git_user_name, body.git_user_email)
        if "git_user_name" in fields_set or "git_user_email" in fields_set
        else None
    )
    return await github_repo_service.rotate_token(
        pool,
        crypto_box,
        session_id=session_id,
        resource_id=resource_id,
        new_token=body.authorization_token.get_secret_value(),
        identity=identity,
    )


def _require_github_resource_id(resource_id: str) -> None:
    """Reject non-github resource ids with a useful 400 instead of a 404.

    Memory store attachments don't have a separate id; pointing at a
    ``memstore_`` here is almost always a client bug.
    """
    try:
        prefix, _ = split_id(resource_id)
    except ValueError as exc:
        raise ValidationError(
            f"malformed resource id: {resource_id!r}",
            detail={"resource_id": resource_id},
        ) from exc
    if prefix != GITHUB_REPOSITORY:
        raise ValidationError(
            "only github_repository resource ids (prefix 'ghrepo_') are "
            "supported on the per-resource sub-collection endpoints",
            detail={"resource_id": resource_id, "prefix": prefix},
        )


@router.post(
    "/{session_id}/archive",
    operation_id="archive_session",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive(session_id: str, pool: PoolDep, _auth: AuthDep) -> Session:
    return await service.archive_session(pool, session_id)


@router.post(
    "/{session_id}/clone",
    operation_id="clone_session",
    status_code=status.HTTP_201_CREATED,
)
async def clone(
    session_id: str,
    body: SessionCloneRequest,
    pool: PoolDep,
    _auth: AuthDep,
) -> Session:
    """Clone a session at its current state into a new session.

    The clone inherits everything that defines the parent's next-step context
    (events, agent binding, vaults, focal channel, status, stop_reason) but
    has its own session_id and a fresh workspace volume by default.

    Refuses if the parent isn't ``idle`` or ``terminated``.
    """
    return await service.clone_session(pool, session_id, workspace_path=body.workspace_path)


@router.delete(
    "/{session_id}",
    operation_id="delete_session",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete(session_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.delete_session(pool, session_id)


@router.post(
    "/{session_id}/messages",
    operation_id="send_message",
    status_code=status.HTTP_201_CREATED,
)
async def post_message(
    session_id: str,
    body: SessionUserMessage,
    pool: PoolDep,
    procrastinate: ProcrastinateDep,
    _auth: AuthDep,
) -> Event:
    metadata = body.metadata or None
    if metadata is not None:
        channel = metadata.get("channel")
        if isinstance(channel, str):
            async with pool.acquire() as conn:
                bound = await queries.list_session_channels(conn, session_id)
            if channel not in bound:
                raise ValidationError(
                    f"metadata.channel={channel!r} is not a bound channel "
                    f"on this session; omit metadata.channel to inject as a "
                    f"global-inbox event"
                )
    event = await service.append_user_message(pool, session_id, body.content, metadata=metadata)
    await defer_wake(pool, session_id, cause="message")
    return event


@router.post("/{session_id}/interrupt", operation_id="interrupt_session")
async def interrupt(
    session_id: str,
    body: SessionInterruptRequest,
    pool: PoolDep,
    _auth: AuthDep,
) -> Session:
    """Interrupt a running session: cancel all in-flight work and idle it."""
    await service.append_event(pool, session_id, "interrupt", {"reason": body.reason})
    await service.set_session_status(pool, session_id, "idle", stop_reason={"type": "interrupt"})
    await service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": "interrupted", "status": "idle", "stop_reason": "interrupt"},
    )
    await pool.execute("SELECT pg_notify($1, $2)", SESSION_INTERRUPT_CHANNEL, session_id)
    return await service.get_session(pool, session_id)


@router.post(
    "/{session_id}/tool-results",
    operation_id="submit_tool_result",
    status_code=status.HTTP_201_CREATED,
)
async def submit_tool_result(
    session_id: str,
    body: ToolResultRequest,
    pool: PoolDep,
    _auth: AuthDep,
) -> Event:
    """Submit a custom tool result. Appends a tool-role message and wakes the session.

    Stamps the tool's ``name`` into the event data by looking it up on the
    parent assistant's ``tool_calls`` array — same source the harness uses
    for built-in/MCP results — so the derived ``tool_name`` column stays
    populated for custom tools too (issue #133).  Returns 404 when the
    ``tool_call_id`` has no matching parent assistant tool call, since a
    result with no parent is a client bug that would leave an orphan row.
    """
    async with pool.acquire() as conn:
        name = await queries.lookup_tool_name_by_call_id(conn, session_id, body.tool_call_id)
        if name is None:
            raise NotFoundError(f"tool_call_id {body.tool_call_id!r} not found")
        data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": body.tool_call_id,
            "content": body.content,
            "name": name,
        }
        if body.is_error:
            data["is_error"] = True
        event = await queries.append_event(conn, session_id=session_id, kind="message", data=data)
    await defer_wake(pool, session_id, cause="custom_tool_result")
    return event


@router.post(
    "/{session_id}/tool-confirmations",
    operation_id="submit_tool_confirmation",
    status_code=status.HTTP_201_CREATED,
)
async def submit_tool_confirmation(
    session_id: str,
    body: ToolConfirmationRequest,
    pool: PoolDep,
    _auth: AuthDep,
) -> Event:
    """Confirm or deny an ``always_ask`` built-in tool call.

    ``allow`` records a lifecycle event; the worker dispatches the tool on
    its next step.  ``deny`` appends a tool-role error event; the model
    sees the denial message and can adapt.
    """
    if body.result == "allow":
        event = await service.confirm_tool_allow(pool, session_id, body.tool_call_id)
    else:
        deny_msg = body.deny_message or "Tool use denied by user."
        event = await service.confirm_tool_deny(pool, session_id, body.tool_call_id, deny_msg)
    await defer_wake(pool, session_id, cause="tool_confirmation")
    return event


@router.get("/{session_id}/events", operation_id="list_session_events")
async def list_events(
    session_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    after_seq: int = 0,
    kind: EventKind | None = None,
    limit: int = 200,
) -> ListResponse[Event]:
    items = await service.read_events(pool, session_id, after_seq=after_seq, kind=kind, limit=limit)
    return ListResponse[Event](
        data=items,
        has_more=len(items) == limit,
        next_after=str(items[-1].seq) if items else None,
    )


@router.get("/{session_id}/context", operation_id="get_session_context")
async def get_context(
    session_id: str,
    pool: PoolDep,
    _auth: AuthDep,
) -> ContextResponse:
    """Return the chat-completions payload the worker would send next.

    Dry-run preview for debugging prompt construction.  Reuses the exact
    composer the worker's step function uses (:func:`compose_step_context`)
    so the endpoint's output is byte-identical to what the next model
    call would see — no divergence.  Side effects (skill provisioning,
    session-status bumps, event appends) are omitted; the endpoint is
    read-only.
    """
    from aios.harness.step_context import compose_step_context, compute_step_prelude
    from aios.harness.tokens import approx_tokens
    from aios.models.agents import Agent, AgentVersion
    from aios.services import agents as agents_service
    from aios.services.channels import list_session_channels

    session = await service.get_session(pool, session_id)

    agent: Agent | AgentVersion
    if session.agent_version is not None:
        agent = await agents_service.get_agent_version(
            pool, session.agent_id, session.agent_version
        )
    else:
        agent = await agents_service.get_agent(pool, session.agent_id)

    channels = await list_session_channels(pool, session_id)

    from aios.db import queries as _queries

    async with pool.acquire() as _conn:
        memory_echoes = await _queries.list_session_memory_store_echoes(_conn, session_id)

    prelude = await compute_step_prelude(
        pool,
        session_id,
        session=session,
        agent=agent,
        channels=channels,
        memory_store_echoes=memory_echoes,
    )
    overhead_local = (
        approx_tokens(
            [{"role": "system", "content": prelude.system_prompt}],
            tools=prelude.tools,
        )
        + prelude.tail_block_upper_bound_local
    )

    events = await service.read_windowed_events(
        pool,
        session_id,
        window_min=agent.window_min,
        window_max=agent.window_max,
        model=agent.model,
        overhead_local=overhead_local,
    )

    step_ctx = await compose_step_context(
        session=session,
        agent=agent,
        channels=channels,
        prelude=prelude,
        events=events,
    )
    return ContextResponse(
        session_id=session_id,
        model=step_ctx.model,
        messages=step_ctx.messages,
        tools=step_ctx.tools,
    )


@router.get("/{session_id}/stream", openapi_extra={"x-codegen": {"targets": []}})
async def stream_events(
    session_id: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    _auth: AuthDep,
    after_seq: int = 0,
) -> EventSourceResponse:
    """Stream session events as Server-Sent Events."""
    await service.get_session(pool, session_id)
    return EventSourceResponse(
        sse_event_stream(db_url, pool, session_id, after_seq=after_seq),
        ping=15,
    )


@router.get("/{session_id}/wait", openapi_extra={"x-codegen": {"targets": []}})
async def wait_for_events(
    session_id: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    _auth: AuthDep,
    after_seq: int = 0,
    timeout_seconds: Annotated[int, Query(alias="timeout", ge=0, le=60)] = 30,
) -> WaitResponse:
    """Long-poll for new events past ``after_seq``.

    Blocks up to ``timeout`` seconds for events to arrive; returns an empty
    list if none land in time. Alternative to SSE for clients whose HTTP
    stack can't reliably consume server-sent events (notably Node's
    ``fetch`` — see issue #40).
    """
    await service.get_session(pool, session_id)

    async with listen_for_events(db_url, session_id) as queue:
        events = await service.read_events(pool, session_id, after_seq=after_seq)
        if not events and timeout_seconds > 0:
            # The channel carries both committed-event IDs and transient
            # streaming delta payloads (shaped like {"delta": "..."}); only
            # the former advance the log, so delta pokes must not count
            # against the wait budget.
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if payload.startswith("{"):
                    continue
                events = await service.read_events(pool, session_id, after_seq=after_seq)
                if events:
                    break

    session = await service.get_session(pool, session_id)
    return WaitResponse(
        events=events,
        session_status=session.status,
        session_stop_reason=session.stop_reason,
        next_after=events[-1].seq if events else after_seq,
    )
