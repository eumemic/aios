"""Connector admin endpoints.

The connector subprocesses live on the worker process; the API process
talks to them indirectly via procrastinate jobs that NOTIFY back when
done.  Each handler:

1. Mints a ``call_id`` ULID and ``LISTEN``s on the result channel
   first, before enqueuing — the LISTEN-before-action invariant from
   :mod:`aios.db.listen` makes sure NOTIFY can't fire into a dead
   subscriber and get dropped.
2. Defers the matching ``harness.connector_*`` task.
3. Awaits one NOTIFY payload with a 60-second ceiling.
4. Translates ``error`` envelopes into HTTP status codes.

408 (Request Timeout) means the worker didn't NOTIFY within 60s — the
job may still be queued / running; the operator should retry or look
at the worker log.  503 (Service Unavailable) means the worker isn't
running connector machinery (no supervisor, name not enabled) or the
connector itself is down.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette import EventSourceResponse

from aios.api.connector_rpc import connector_rpc
from aios.api.deps import AuthDep, ConnectorAuthDep, DbUrlDep, PoolDep
from aios.api.sse import connector_calls_stream
from aios.db import queries
from aios.errors import (
    AiosError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from aios.harness.connector_tasks import (
    defer_connector_call,
    defer_connector_status,
    defer_connector_tools,
)
from aios.harness.wake import defer_wake
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


_RESULT_TIMEOUT_S = 60.0


class ConnectorCallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = {}
    meta: dict[str, Any] | None = None


async def _rpc(db_url: str, defer: Callable[[str], Awaitable[None]]) -> dict[str, Any]:
    """Admin-endpoint round-trip: ``connector_rpc`` + granular error mapping."""
    envelope = await connector_rpc(db_url, defer, timeout_s=_RESULT_TIMEOUT_S)
    _raise_for_error(envelope)
    return envelope


_CODE_TO_STATUS: dict[str, int] = {
    "not_enabled": status.HTTP_404_NOT_FOUND,
    "not_ready": status.HTTP_503_SERVICE_UNAVAILABLE,
    "circuit_open": status.HTTP_503_SERVICE_UNAVAILABLE,
    "transport_error": status.HTTP_503_SERVICE_UNAVAILABLE,
    "tool_error": status.HTTP_502_BAD_GATEWAY,
    "ambiguous_instance": status.HTTP_409_CONFLICT,
}


def _raise_for_error(envelope: dict[str, Any]) -> None:
    """Map the worker's error envelope onto the right HTTP status.

    Producer-side codes are the contract; the human-readable
    ``error`` string is for the operator's eyes.  An unknown / missing
    code falls through to 502 so a future code addition that we don't
    yet recognize doesn't masquerade as success.
    """
    err = envelope.get("error")
    if not err:
        return
    raw_code = envelope.get("code")
    code = raw_code if isinstance(raw_code, str) else ""
    raise HTTPException(
        status_code=_CODE_TO_STATUS.get(code, status.HTTP_502_BAD_GATEWAY),
        detail=err,
    )


@router.get("", openapi_extra={"x-codegen": {"targets": []}})
async def list_(db_url: DbUrlDep, _auth: AuthDep) -> dict[str, Any]:
    """Snapshot every enabled connector instance."""
    return await _rpc(
        db_url, lambda cid: defer_connector_status(call_id=cid, connector=None, instance=None)
    )


@router.get("/{connector}", openapi_extra={"x-codegen": {"targets": []}})
async def list_for_connector(connector: str, db_url: DbUrlDep, _auth: AuthDep) -> dict[str, Any]:
    """Snapshot every instance of one connector type."""
    return await _rpc(
        db_url,
        lambda cid: defer_connector_status(call_id=cid, connector=connector, instance=None),
    )


@router.get("/{connector}/{instance}", openapi_extra={"x-codegen": {"targets": []}})
async def get_instance(
    connector: str, instance: str, db_url: DbUrlDep, _auth: AuthDep
) -> dict[str, Any]:
    """Snapshot a single ``(connector, instance)`` pair."""
    return await _rpc(
        db_url,
        lambda cid: defer_connector_status(call_id=cid, connector=connector, instance=instance),
    )


@router.get("/{connector}/{instance}/accounts", openapi_extra={"x-codegen": {"targets": []}})
async def list_accounts(
    connector: str, instance: str, db_url: DbUrlDep, _auth: AuthDep
) -> dict[str, Any]:
    envelope = await _rpc(
        db_url,
        lambda cid: defer_connector_status(call_id=cid, connector=connector, instance=instance),
    )
    snapshot = envelope["connector"]
    return {
        "connector": snapshot["connector"],
        "instance": snapshot["instance"],
        "accounts": snapshot["accounts"],
    }


@router.get("/{connector}/{instance}/tools", openapi_extra={"x-codegen": {"targets": []}})
async def list_tools(
    connector: str, instance: str, db_url: DbUrlDep, _auth: AuthDep
) -> dict[str, Any]:
    return await _rpc(
        db_url,
        lambda cid: defer_connector_tools(call_id=cid, connector=connector, instance=instance),
    )


@router.post("/{connector}/{instance}/call", openapi_extra={"x-codegen": {"targets": []}})
async def call(
    connector: str,
    instance: str,
    body: ConnectorCallBody,
    db_url: DbUrlDep,
    _auth: AuthDep,
) -> dict[str, Any]:
    return await _rpc(
        db_url,
        lambda cid: defer_connector_call(
            call_id=cid,
            connector=connector,
            instance=instance,
            tool=body.tool,
            arguments=body.arguments,
            meta=body.meta,
        ),
    )
