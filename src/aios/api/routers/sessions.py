"""Session endpoints: create/list/get sessions, append messages, list events.

Phase 1 runs the harness loop **inline** inside the request handler — there's
no worker yet. This means a `POST /sessions` with an `initial_message` blocks
until the model responds. Phase 2 splits this into a worker enqueue.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, status

from aios.api.deps import AuthDep, PoolDep, VaultDep
from aios.harness.loop import run_session_turn
from aios.models.common import ListResponse
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionCreate, SessionUserMessage
from aios.services import sessions as service

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    body: SessionCreate,
    pool: PoolDep,
    vault: VaultDep,
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
        # Phase 1: run the harness loop inline. Phase 2 enqueues a worker job.
        await run_session_turn(pool, vault, session.id)
        # Re-fetch the session to include the post-turn status.
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
    vault: VaultDep,
    _auth: AuthDep,
) -> Event:
    event = await service.append_user_message(pool, session_id, body.content)
    # Phase 1 runs the harness loop inline. Phase 2 enqueues a worker job.
    await run_session_turn(pool, vault, session_id)
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
