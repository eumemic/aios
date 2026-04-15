"""Session endpoints: create, list, get sessions, append messages, list/stream events.

Posting a message appends a user-message event and defers a procrastinate
``wake_session`` job that a worker will pick up. The endpoint returns 201
immediately.

Clients that want to watch the agent reply in real time should connect to
``GET /v1/sessions/{id}/stream``, which streams events over SSE backed by
Postgres ``LISTEN``/``NOTIFY``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, status
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    AuthDep,
    DbUrlDep,
    PoolDep,
    ProcrastinateDep,
)
from aios.api.sse import sse_event_stream
from aios.harness.wake import defer_wake
from aios.models.common import ListResponse
from aios.models.events import Event, EventKind
from aios.models.sessions import (
    Session,
    SessionCreate,
    SessionInterruptRequest,
    SessionUpdate,
    SessionUserMessage,
    ToolConfirmationRequest,
    ToolResultRequest,
)
from aios.services import sessions as service

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    body: SessionCreate,
    pool: PoolDep,
    procrastinate: ProcrastinateDep,
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
        workspace_path=body.workspace_path,
        env=body.env or None,
    )
    if body.initial_message is not None:
        await service.append_user_message(pool, session.id, body.initial_message)
        await defer_wake(session.id, cause="initial_message")
        session = await service.get_session(pool, session.id)
    return session


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    agent_id: str | None = None,
    status_filter: Annotated[
        Literal["running", "idle", "terminated"] | None,
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


@router.get("/{session_id}")
async def get(session_id: str, pool: PoolDep, _auth: AuthDep) -> Session:
    return await service.get_session(pool, session_id)


@router.put("/{session_id}")
async def update(session_id: str, body: SessionUpdate, pool: PoolDep, _auth: AuthDep) -> Session:
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
    )


@router.post("/{session_id}/archive")
async def archive(session_id: str, pool: PoolDep, _auth: AuthDep) -> Session:
    return await service.archive_session(pool, session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(session_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.delete_session(pool, session_id)


@router.post("/{session_id}/messages", status_code=status.HTTP_201_CREATED)
async def post_message(
    session_id: str,
    body: SessionUserMessage,
    pool: PoolDep,
    procrastinate: ProcrastinateDep,
    _auth: AuthDep,
) -> Event:
    event = await service.append_user_message(
        pool, session_id, body.content, metadata=body.metadata or None
    )
    await defer_wake(session_id, cause="message")
    return event


@router.post("/{session_id}/interrupt")
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
    return await service.get_session(pool, session_id)


@router.post("/{session_id}/tool-results", status_code=status.HTTP_201_CREATED)
async def submit_tool_result(
    session_id: str,
    body: ToolResultRequest,
    pool: PoolDep,
    _auth: AuthDep,
) -> Event:
    """Submit a custom tool result. Appends a tool-role message and wakes the session."""
    data: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": body.tool_call_id,
        "content": body.content,
    }
    if body.is_error:
        data["is_error"] = True
    event = await service.append_event(pool, session_id, "message", data)
    await defer_wake(session_id, cause="custom_tool_result")
    return event


@router.post("/{session_id}/tool-confirmations", status_code=status.HTTP_201_CREATED)
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
    await defer_wake(session_id, cause="tool_confirmation")
    return event


@router.get("/{session_id}/events")
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


@router.get("/{session_id}/stream")
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
