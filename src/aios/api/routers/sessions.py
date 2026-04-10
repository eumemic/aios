"""Session endpoints: create, list, get sessions, append messages, list/stream events.

Phase 2 splits the runtime into separate api and worker processes. Posting a
message no longer runs the harness loop inline — it appends the user-message
event and defers a procrastinate ``wake_session`` job that a worker will pick
up. The endpoint returns 201 immediately with the appended event in the body.

Clients that want to watch the agent reply in real time should connect to
``GET /v1/sessions/{id}/stream``, which streams events over SSE backed by
Postgres ``LISTEN``/``NOTIFY``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, status
from procrastinate import exceptions as procrastinate_exceptions
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    AuthDep,
    DbUrlDep,
    PoolDep,
    ProcrastinateDep,
)
from aios.api.sse import sse_event_stream
from aios.logging import get_logger
from aios.models.common import ListResponse
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionCreate, SessionUserMessage
from aios.services import sessions as service

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

log = get_logger("aios.api.sessions")


async def _defer_wake(procrastinate: ProcrastinateDep, session_id: str, cause: str) -> None:
    """Enqueue a wake_session job, swallowing AlreadyEnqueued.

    The api never blocks on the worker — if a wake is already queued for
    this session, our message will be picked up by it (the worker's loop
    reads ALL pending message events from the DB).
    """
    try:
        await procrastinate.configure_task("harness.wake_session").defer_async(
            session_id=session_id,
            cause=cause,
        )
    except procrastinate_exceptions.AlreadyEnqueued:
        log.info("session.wake_already_enqueued", session_id=session_id, cause=cause)


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
        title=body.title,
        metadata=body.metadata,
    )
    if body.initial_message is not None:
        await service.append_user_message(pool, session.id, body.initial_message)
        await _defer_wake(procrastinate, session.id, cause="initial_message")
        # Re-fetch so the response has the most current status.
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


@router.post("/{session_id}/messages", status_code=status.HTTP_201_CREATED)
async def post_message(
    session_id: str,
    body: SessionUserMessage,
    pool: PoolDep,
    procrastinate: ProcrastinateDep,
    _auth: AuthDep,
) -> Event:
    event = await service.append_user_message(pool, session_id, body.content)
    await _defer_wake(procrastinate, session_id, cause="message")
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
    """Stream session events as Server-Sent Events.

    Backfills events with ``seq > after_seq`` then tails live notifications
    via Postgres LISTEN/NOTIFY. The connection lives until the client
    disconnects or the session terminates.
    """
    # Validate the session exists before opening the stream so 404s come
    # back as JSON, not SSE.
    await service.get_session(pool, session_id)

    return EventSourceResponse(
        sse_event_stream(db_url, pool, session_id, after_seq=after_seq),
        ping=15,
    )
