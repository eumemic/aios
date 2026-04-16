"""Connection endpoints + the inbound-message endpoint.

Inbound flow: a connector posts a message for some ``path`` (chat id);
we build the channel ``address`` from ``connector/account/path``, run
the resolver, append a user-message event with ``metadata.channel``
stamped, and defer a wake job.

``DELETE /{id}`` soft-archives by design.  No hard-delete endpoint:
connections are cheap to retain archived for audit, and dropping the
``connections`` row would orphan any historical ``metadata.channel``
references in the event log.  (Vaults expose both because vault
credentials carry encrypted secrets that may need to be purged.)
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.errors import ValidationError
from aios.harness.wake import defer_wake
from aios.models.common import ListResponse
from aios.models.connections import (
    Connection,
    ConnectionCreate,
    ConnectionUpdate,
    InboundMessage,
    InboundMessageResponse,
)
from aios.services import channels as channels_service
from aios.services import connections as service
from aios.services import sessions as sessions_service

router = APIRouter(prefix="/v1/connections", tags=["connections"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: ConnectionCreate, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.create_connection(
        pool,
        connector=body.connector,
        account=body.account,
        mcp_url=body.mcp_url,
        vault_id=body.vault_id,
        metadata=body.metadata,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Connection]:
    items = await service.list_connections(pool, limit=limit, after=after)
    return ListResponse[Connection](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{connection_id}")
async def get(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.get_connection(pool, connection_id)


@router.put("/{connection_id}")
async def update(
    connection_id: str, body: ConnectionUpdate, pool: PoolDep, _auth: AuthDep
) -> Connection:
    return await service.update_connection(
        pool,
        connection_id,
        mcp_url=body.mcp_url,
        vault_id=body.vault_id,
        metadata=body.metadata,
    )


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(connection_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_connection(pool, connection_id)


# ─── inbound message ────────────────────────────────────────────────────────


@router.post("/{connection_id}/messages", status_code=status.HTTP_201_CREATED)
async def post_message(
    connection_id: str,
    body: InboundMessage,
    pool: PoolDep,
    _auth: AuthDep,
) -> InboundMessageResponse:
    # Path is split on '/' to form trailing address segments.  Reject
    # empty leading/trailing/interior segments and '..' to keep the
    # full address (and thus prefix-rule matching) unambiguous.
    segments = body.path.split("/")
    if not segments or any(seg == "" or seg == ".." for seg in segments):
        raise ValidationError(
            "path must be non-empty with no empty segments or '..'",
            detail={"path": body.path},
        )

    connection = await service.get_connection(pool, connection_id)
    address = f"{connection.connector}/{connection.account}/{body.path}"

    resolution = await channels_service.resolve_channel(pool, address)

    metadata = {**body.metadata, "channel": address}
    event = await sessions_service.append_user_message(
        pool, resolution.session_id, body.content, metadata=metadata
    )
    await defer_wake(resolution.session_id, cause="inbound_message")

    return InboundMessageResponse(
        session_id=resolution.session_id,
        event_id=event.id,
        created_session=resolution.created_session,
    )
