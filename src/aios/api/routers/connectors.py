"""Connector-facing endpoints (#301).

A connector container talks to aios via three routes here:

* ``POST /v1/connectors/inbound`` — submit an inbound user message.
* ``POST /v1/connectors/tool-results`` — submit a custom-tool result.
* ``GET /v1/connectors/calls`` — SSE stream of pending tool calls.

All three use ``ConnectorAuthDep`` — the bearer token resolves to a
single ``connection_id``; the request body never carries it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette import EventSourceResponse

from aios.api.deps import ConnectorAuthDep, CryptoBoxDep, DbUrlDep, PoolDep
from aios.api.sse import connector_calls_stream
from aios.db import queries
from aios.errors import (
    AiosError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from aios.harness.wake import defer_wake
from aios.models.connections import ConnectorSecrets
from aios.services import connections as connections_service
from aios.services import inbound as inbound_service
from aios.services import sessions as sessions_service

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


# ─── connector-facing endpoints (#301) ──────────────────────────────────────


class ConnectorInboundRequest(BaseModel):
    """Body for ``POST /v1/connectors/inbound``.

    Authenticated via ``ConnectorAuthDep`` so the connection_id is
    server-resolved from the bearer token — clients don't pick which
    connection their inbound lands on.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(
        description="Client-supplied dedup key (ULID).  Replays return the original event id.",
    )
    chat_id: str
    sender: dict[str, Any] = Field(default_factory=dict)
    content: str
    attachments: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    timestamp: str | None = Field(
        default=None,
        description="Optional ISO-8601 platform timestamp; stored in event metadata.",
    )


class ConnectorInboundResponse(BaseModel):
    """Response for ``POST /v1/connectors/inbound``."""

    appended_event_id: str | None
    session_id: str | None
    deduped: bool


def _inbound_drop_error(drop_reason: str) -> AiosError:
    """Pick the right :class:`AiosError` subclass for a drop_reason.

    Each drop maps onto an existing error type — preserving HTTP status
    contracts without inventing a new error_type for every reason.
    Detail carries ``drop_reason`` so clients can branch on the
    machine-readable value.
    """
    detail = {"drop_reason": drop_reason}
    msg = f"inbound dropped ({drop_reason})"
    if drop_reason == "payload_too_large":
        return PayloadTooLargeError(msg, detail=detail)
    if drop_reason in ("detached", "archived_template"):
        return ValidationError(msg, detail=detail)
    if drop_reason == "session_missing":
        return NotFoundError(msg, detail=detail)
    # attachment_staging_failed (and any unrecognised reason) → 500.
    return AiosError(msg, detail=detail)


@router.post("/inbound", operation_id="post_connector_inbound", status_code=status.HTTP_201_CREATED)
async def post_inbound(
    body: ConnectorInboundRequest,
    pool: PoolDep,
    connection_id: ConnectorAuthDep,
) -> ConnectorInboundResponse:
    """Append an inbound user message to the session bound to the caller's connection.

    Idempotent on ``body.event_id`` — replays return the original
    event id with ``deduped=True``.  Drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).
    """
    result = await inbound_service.handle_inbound(
        pool,
        connection_id=connection_id,
        event_id=body.event_id,
        chat_id=body.chat_id,
        sender=body.sender,
        content=body.content,
        attachments=body.attachments,
        connector_metadata=body.metadata,
        platform_timestamp=body.timestamp,
    )
    if result.drop_reason is not None:
        raise _inbound_drop_error(result.drop_reason.value)
    return ConnectorInboundResponse(
        appended_event_id=result.appended_event_id,
        session_id=result.session_id,
        deduped=result.deduped,
    )


class ConnectorToolResultRequest(BaseModel):
    """Body for ``POST /v1/connectors/tool-results``.

    Mirrors the operator-facing :class:`ToolResultRequest` but adds
    ``session_id`` since connector tokens aren't path-scoped to a
    session.  The handler validates the session is bound to the
    caller's connection (preventing a connector from posting results
    for sessions outside its scope).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tool_call_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False


@router.post(
    "/tool-results",
    operation_id="post_connector_tool_result",
    status_code=status.HTTP_201_CREATED,
)
async def post_tool_result(
    body: ConnectorToolResultRequest,
    pool: PoolDep,
    connection_id: ConnectorAuthDep,
) -> Any:
    """Submit a custom tool result from a connector container.

    Authorization: the session must be bound to the caller's connection
    (single_session attach, per_chat origin, or operator-bound chat).
    Otherwise → 403.
    """
    async with pool.acquire() as conn:
        if not await queries.is_session_bound_to_connection(
            conn, connection_id=connection_id, session_id=body.session_id
        ):
            raise ForbiddenError(
                "session is not bound to this connection",
                detail={"session_id": body.session_id, "connection_id": connection_id},
            )
        event = await sessions_service.append_tool_result(
            conn,
            session_id=body.session_id,
            tool_call_id=body.tool_call_id,
            content=body.content,
            is_error=body.is_error,
        )
    await defer_wake(pool, body.session_id, cause="connector_tool_result")
    return event


@router.get("/secrets", operation_id="get_connector_secrets")
async def get_secrets(
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    connection_id: ConnectorAuthDep,
) -> ConnectorSecrets:
    """Decrypted secrets for the caller's connection.

    The bearer token resolves server-side to one ``connection_id``;
    operators set secrets on that connection via
    ``POST /v1/connections`` or ``PUT /v1/connections/{id}/secrets`` and
    never read them back through the operator surface.  This is the only
    decryption path.

    Returns ``{"secrets": {}}`` when no secrets are configured — the
    connector author decides whether that's acceptable (most need at
    least one credential and should fail loudly).
    """
    secrets = await connections_service.get_connection_secrets(
        pool, connection_id, crypto_box=crypto_box
    )
    return ConnectorSecrets(secrets=secrets)


@router.get("/calls", openapi_extra={"x-codegen": {"targets": []}})
async def get_calls(
    db_url: DbUrlDep,
    pool: PoolDep,
    connection_id: ConnectorAuthDep,
) -> EventSourceResponse:
    """SSE stream of pending custom tool calls for the caller's connection.

    Backfills any pending calls at subscribe time (calls already parked
    in some session's ``stop_reason.custom_tools``), then tails the
    ``connector_calls_<connection_id>`` NOTIFY channel.  Each emitted
    event is keyed ``call`` with a JSON body shaped::

        {
            "session_id": "...",
            "tool_call_id": "...",
            "name": "...",
            "arguments": "...",       // JSON string from the model
            "focal_channel": "..."
        }

    Connector containers dedupe by ``tool_call_id`` client-side so SSE
    reconnects (which replay the backfill) don't double-execute.
    """
    return EventSourceResponse(
        connector_calls_stream(db_url, pool, connection_id),
        ping=15,
    )
