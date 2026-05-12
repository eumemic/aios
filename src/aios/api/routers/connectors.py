"""Connector-facing endpoints (#301).

A connector container talks to aios via three routes here:

* ``POST /v1/connectors/inbound`` — submit an inbound user message.
* ``POST /v1/connectors/tool-results`` — submit a custom-tool result.
* ``GET /v1/connectors/calls`` — SSE stream of pending tool calls.

All three use ``ConnectorAuthDep`` — the bearer token resolves to a
single ``connection_id``; the request body never carries it.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    ConnectorAuthDep,
    CryptoBoxDep,
    DbUrlDep,
    PoolDep,
    RuntimeAuthDep,
)
from aios.api.sse import (
    connection_discovery_stream,
    connector_calls_stream,
    runtime_connector_calls_stream,
)
from aios.db import queries
from aios.errors import (
    AiosError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from aios.models.connections import Connection, ConnectionSetTools, ConnectorSecrets
from aios.services import connections as connections_service
from aios.services import inbound as inbound_service
from aios.services import sessions as sessions_service
from aios.services.attachment_staging import InboundAttachment
from aios.services.wake import defer_wake

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


# ─── connector-facing endpoints (#301) ──────────────────────────────────────


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


_DEFAULT_ATTACHMENT_CONTENT_TYPE = "application/octet-stream"


def _parse_form_json(field: str, raw: str | None, *, default: Any = None) -> Any:
    """Decode an optional JSON-in-multipart-form field.

    Multipart form values are always text; we expose ``sender`` and
    ``metadata`` as JSON-encoded strings so connector clients keep the
    shape the JSON inbound used (with JSON dicts) without hand-rolled
    field-flattening on either side. Raises :class:`ValidationError` on
    bad JSON so connector authors see the parse error in the response,
    not a 500.
    """
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValidationError(
            f"{field!r} must be a JSON-encoded string",
            detail={"field": field, "error": str(err)},
        ) from err


async def _do_inbound(
    pool: Any,
    *,
    connection_id: str,
    event_id: str,
    chat_id: str,
    content: str,
    sender_json: str | None,
    metadata_json: str | None,
    timestamp: str | None,
    attachments: list[UploadFile] | None,
) -> ConnectorInboundResponse:
    """Shared body for the per-connection and runtime inbound handlers.

    The two handlers diverge only on auth (one trusts the bearer's
    ``connection_id``; the other accepts it as a form field after a
    runtime-scope check).  Everything after that — JSON-form parsing,
    attachment shaping, the handle_inbound call, drop-reason mapping
    — is identical.
    """
    sender_dict: dict[str, Any] = _parse_form_json("sender", sender_json, default={}) or {}
    metadata_dict: dict[str, Any] | None = _parse_form_json("metadata", metadata_json)
    inbound_attachments = [
        InboundAttachment(
            stream=upload,
            filename=upload.filename or "",
            content_type=upload.content_type or _DEFAULT_ATTACHMENT_CONTENT_TYPE,
        )
        for upload in (attachments or [])
    ]
    result = await inbound_service.handle_inbound(
        pool,
        connection_id=connection_id,
        event_id=event_id,
        chat_id=chat_id,
        sender=sender_dict,
        content=content,
        attachments=inbound_attachments,
        connector_metadata=metadata_dict,
        platform_timestamp=timestamp,
    )
    if result.drop_reason is not None:
        raise _inbound_drop_error(result.drop_reason.value)
    return ConnectorInboundResponse(
        appended_event_id=result.appended_event_id,
        session_id=result.session_id,
        deduped=result.deduped,
    )


@router.post(
    "/inbound",
    operation_id="post_connector_inbound",
    status_code=status.HTTP_201_CREATED,
)
async def post_inbound(
    pool: PoolDep,
    connection_id: ConnectorAuthDep,
    event_id: Annotated[str, Form(description="Client-supplied dedup key (ULID).")],
    chat_id: Annotated[str, Form()],
    content: Annotated[str, Form()],
    sender: Annotated[
        str | None,
        Form(description='JSON-encoded sender dict (e.g. {"display_name": "Alice"}).'),
    ] = None,
    metadata: Annotated[
        str | None,
        Form(description="JSON-encoded connector metadata dict."),
    ] = None,
    timestamp: Annotated[
        str | None,
        Form(description="Optional ISO-8601 platform timestamp; stored in event metadata."),
    ] = None,
    attachments: Annotated[
        list[UploadFile] | None,
        File(description="One file part per attachment; filename + content-type read from each."),
    ] = None,
) -> ConnectorInboundResponse:
    """Append an inbound user message to the session bound to the caller's connection.

    Multipart form: the message text + IDs ride as form fields; any
    attachments ride as ``UploadFile`` parts whose bytes the handler
    streams into the per-session attachment dir (no shared-filesystem
    coupling, closes #322 P1).

    Idempotent on ``event_id`` — replays return the original event id
    with ``deduped=True``. Drops surface as 4xx/5xx with a body
    explaining the reason (operator-config issue vs server error vs
    payload).
    """
    return await _do_inbound(
        pool,
        connection_id=connection_id,
        event_id=event_id,
        chat_id=chat_id,
        content=content,
        sender_json=sender,
        metadata_json=metadata,
        timestamp=timestamp,
        attachments=attachments,
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


@router.put("/tools", operation_id="set_connector_tools")
async def set_tools(
    body: ConnectionSetTools,
    pool: PoolDep,
    connection_id: ConnectorAuthDep,
) -> Connection:
    """Publish the connector's tool schemas onto its own connection.

    The connector container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and POSTs them here, replacing whatever was on the
    connection wholesale.  Operators don't hand-write ``tools.json``.

    Authorization: the bearer token resolves to one ``connection_id``;
    a connector can only publish tools for its own connection.  This
    is the connector-scoped twin of operator-scoped
    ``PUT /v1/connections/{id}/tools``.
    """
    return await connections_service.set_connection_tools(pool, connection_id, tools=body.tools)


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


# ─── runtime-scoped endpoints (#328 PR 5) ────────────────────────────────────
#
# These mirror the per-connection routes above but accept a ``runtime``
# bearer token that scopes the caller to one ``connector`` type and N
# of its connections.  The legacy per-connection routes stay alive
# alongside these until PR 7 cuts them; PR 8 drops the table.


class ToolsSchemaUpdate(BaseModel):
    """Body for ``PUT /v1/connectors/{connector}/tools_schema``."""

    model_config = ConfigDict(extra="forbid")

    tools: list[dict[str, Any]]


class RuntimeToolResultRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/tool-results``.

    Like :class:`ConnectorToolResultRequest` but carries ``connection_id``
    explicitly — the bearer scopes the caller to a connector *type*,
    not to one connection.
    """

    model_config = ConfigDict(extra="forbid")

    connection_id: str
    session_id: str
    tool_call_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False


def _check_runtime_scope(auth_connector: str, target_connector: str) -> None:
    """Raise 403 if a runtime bearer reaches outside its connector type."""
    if auth_connector != target_connector:
        raise ForbiddenError(
            "runtime token scoped to a different connector type",
            detail={
                "auth_connector": auth_connector,
                "target_connector": target_connector,
            },
        )


@router.put(
    "/{connector}/tools_schema",
    operation_id="put_connector_tools_schema",
)
async def put_tools_schema(
    connector: str,
    body: ToolsSchemaUpdate,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> None:
    """Publish the runtime container's tool catalog for a connector type.

    The runtime container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and calls this once, replacing whatever was on the
    ``connectors.tools_schema`` row wholesale.  Operators don't
    hand-write the schema.

    Authorization: the runtime bearer's ``connector`` must match the
    path's ``connector``.
    """
    _, auth_connector = auth
    _check_runtime_scope(auth_connector, connector)
    async with pool.acquire() as conn:
        await queries.update_connector_tools_schema(conn, connector, tools_schema=body.tools)


@router.get(
    "/connections",
    openapi_extra={"x-codegen": {"targets": []}},
)
async def get_connection_discovery(
    db_url: DbUrlDep,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> EventSourceResponse:
    """SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {"event": "added" | "removed", "connection_id": "...", "account": "..."}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.
    """
    _, connector = auth
    return EventSourceResponse(
        connection_discovery_stream(db_url, pool, connector),
        ping=15,
    )


@router.post(
    "/runtime/inbound",
    operation_id="post_connector_runtime_inbound",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_inbound(
    pool: PoolDep,
    auth: RuntimeAuthDep,
    connection_id: Annotated[str, Form(description="The connection this inbound belongs to.")],
    event_id: Annotated[str, Form(description="Client-supplied dedup key (ULID).")],
    chat_id: Annotated[str, Form()],
    content: Annotated[str, Form()],
    sender: Annotated[
        str | None,
        Form(description='JSON-encoded sender dict (e.g. {"display_name": "Alice"}).'),
    ] = None,
    metadata: Annotated[
        str | None,
        Form(description="JSON-encoded connector metadata dict."),
    ] = None,
    timestamp: Annotated[
        str | None,
        Form(description="Optional ISO-8601 platform timestamp; stored in event metadata."),
    ] = None,
    attachments: Annotated[
        list[UploadFile] | None,
        File(description="One file part per attachment; filename + content-type read from each."),
    ] = None,
) -> ConnectorInboundResponse:
    """Append an inbound user message to ``connection_id``'s session.

    Runtime-scoped twin of :func:`post_inbound`: the bearer authenticates
    the caller as one connector *type*; ``connection_id`` rides as a
    form field and must belong to that type.  Same multipart shape, same
    dedup-on-``event_id`` semantics, same drop_reason → HTTP mapping.
    """
    _, auth_connector = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
    _check_runtime_scope(auth_connector, connection.connector)
    return await _do_inbound(
        pool,
        connection_id=connection_id,
        event_id=event_id,
        chat_id=chat_id,
        content=content,
        sender_json=sender,
        metadata_json=metadata,
        timestamp=timestamp,
        attachments=attachments,
    )


@router.post(
    "/runtime/tool-results",
    operation_id="post_connector_runtime_tool_result",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_tool_result(
    body: RuntimeToolResultRequest,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> Any:
    """Submit a custom tool result from a runtime container.

    Authorization: the bearer's connector must match ``body.connection_id``'s
    connector, and the session must be bound to that connection.
    """
    _, auth_connector = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id)
        _check_runtime_scope(auth_connector, connection.connector)
        if not await queries.is_session_bound_to_connection(
            conn,
            connection_id=body.connection_id,
            session_id=body.session_id,
        ):
            raise ForbiddenError(
                "session is not bound to this connection",
                detail={
                    "session_id": body.session_id,
                    "connection_id": body.connection_id,
                },
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


@router.get(
    "/runtime/secrets",
    operation_id="get_connector_runtime_secrets",
)
async def get_runtime_secrets(
    connection_id: str,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    auth: RuntimeAuthDep,
) -> ConnectorSecrets:
    """Decrypted secrets for ``connection_id``.

    Runtime-scoped twin of :func:`get_secrets`.  The bearer's connector
    must match the connection's connector type.
    """
    _, auth_connector = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
    _check_runtime_scope(auth_connector, connection.connector)
    secrets = await connections_service.get_connection_secrets(
        pool, connection_id, crypto_box=crypto_box
    )
    return ConnectorSecrets(secrets=secrets)


@router.get(
    "/runtime/calls",
    openapi_extra={"x-codegen": {"targets": []}},
)
async def get_runtime_calls(
    db_url: DbUrlDep,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> EventSourceResponse:
    """SSE stream of pending custom tool calls across every active
    connection of the caller's connector type.

    Backfills at subscribe time, then tails ``connector_calls_<connector>``.
    Each event is keyed ``call`` with the same payload as
    :func:`get_calls` plus an explicit ``connection_id`` field so the
    runtime container can fan out to its per-connection workers.
    """
    _, connector = auth
    return EventSourceResponse(
        runtime_connector_calls_stream(db_url, pool, connector),
        ping=15,
    )
