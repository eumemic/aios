"""Connection endpoints — CRUD plus mode-binding transitions.

Created in detached mode; switch to single_session via ``attach`` or
per_chat via ``configure-per-chat``.  ``DELETE`` soft-archives — but
only on detached connections (the service layer enforces this so
operators can't silently break inbound delivery for live single_session
connections or orphan ``spawned_from_connection_id`` pointers on
per_chat-spawned sessions).
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.connections import (
    Connection,
    ConnectionAttach,
    ConnectionConfigurePerChat,
    ConnectionCreate,
)
from aios.services import connections as service

router = APIRouter(prefix="/v1/connections", tags=["connections"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: ConnectionCreate, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.create_connection(
        pool,
        connector=body.connector,
        account=body.account,
        metadata=body.metadata,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    connector: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Connection]:
    items = await service.list_connections(
        pool,
        connector=connector,
        session_id=session_id,
        limit=limit,
        after=after,
    )
    return ListResponse[Connection](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{connection_id}")
async def get(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.get_connection(pool, connection_id)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(connection_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_connection(pool, connection_id)


@router.post("/{connection_id}/attach")
async def attach(
    connection_id: str, body: ConnectionAttach, pool: PoolDep, _auth: AuthDep
) -> Connection:
    return await service.attach_connection(pool, connection_id, session_id=body.session_id)


@router.post("/{connection_id}/detach")
async def detach(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.detach_connection(pool, connection_id)


@router.post("/{connection_id}/configure-per-chat")
async def configure_per_chat(
    connection_id: str, body: ConnectionConfigurePerChat, pool: PoolDep, _auth: AuthDep
) -> Connection:
    return await service.configure_per_chat(
        pool, connection_id, session_template_id=body.session_template_id
    )


@router.post("/{connection_id}/unconfigure")
async def unconfigure(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.unconfigure_connection(pool, connection_id)
