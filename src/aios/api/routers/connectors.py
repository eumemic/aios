"""Connector-related endpoints — two distinct caller populations.

The file groups three sections:

1. **Runtime-container-facing** (``RuntimeAuthDep``, per-connector-type
   bearer): ``/runtime/inbound``, ``/runtime/tool-results``,
   ``/runtime/calls``, ``/runtime/secrets``, ``/connections``,
   ``/{connector}/tools_schema``, ``/runtime/management-calls``,
   ``/runtime/management-call-results``.  The bearer scopes the caller
   to one ``connector`` type; ``connection_id`` rides as a form/query
   field for the routes that operate on a specific connection.
2. **Operator-facing signal management** (``AuthDep``, operator API key):
   ``/signal/register``, ``/signal/verify``, ``/signal/profile``.  These
   block-await the connector's resolution via the
   ``connector_result_<call_id>`` LISTEN channel.

Section banners (``# ───`) below mark the boundary.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    AuthDep,
    CryptoBoxDep,
    DbUrlDep,
    PoolDep,
    RuntimeAuthDep,
)
from aios.api.sse import (
    connection_discovery_stream,
    management_calls_stream,
    runtime_connector_calls_stream,
)
from aios.db import queries
from aios.errors import (
    AiosError,
    ConnectorCallFailedError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from aios.models.connections import ConnectorSecrets
from aios.services import connections as connections_service
from aios.services import inbound as inbound_service
from aios.services import management_calls
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
    account_id = ""  # PR 4 stub; needs upstream threading
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
        account_id=account_id,
    )
    if result.drop_reason is not None:
        raise _inbound_drop_error(result.drop_reason.value)
    return ConnectorInboundResponse(
        appended_event_id=result.appended_event_id,
        session_id=result.session_id,
        deduped=result.deduped,
    )


# ─── runtime-scoped endpoints (#328 PR 5) ────────────────────────────────────
#
# All routes accept a ``runtime`` bearer token (``RuntimeAuthDep``) that
# scopes the caller to one ``connector`` type and N of its connections.


class ToolsSchemaUpdate(BaseModel):
    """Body for ``PUT /v1/connectors/{connector}/tools_schema``."""

    model_config = ConfigDict(extra="forbid")

    tools: list[dict[str, Any]]


class RuntimeToolResultRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/tool-results``.

    Carries ``connection_id`` explicitly — the bearer scopes the
    caller to a connector *type*, not to one connection, so the body
    has to name the target connection.
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
    account_id = ""  # PR 3 stub for runtime-auth routes; PR 4 derives from connection
    _check_runtime_scope(auth_connector, connector)
    async with pool.acquire() as conn:
        await queries.update_connector_tools_schema(
            conn, connector, tools_schema=body.tools, account_id=account_id
        )


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
    # Default empty so attachment-only / reaction-passthrough / group-update
    # envelopes can flow through.  FastAPI's multipart Form parser treats an
    # empty field value as ``input=null`` (missing), which 422s a required
    # ``content: str``; the explicit ``= ""`` default makes empty bodies a
    # valid first-class shape rather than a server-side validation failure.
    content: Annotated[str, Form()] = "",
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

    The bearer authenticates the caller as one connector *type*;
    ``connection_id`` rides as a form field and must belong to that
    type.  Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).
    """
    _, auth_connector = auth
    account_id = ""  # PR 3 stub for runtime-auth routes; PR 4 derives from connection
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
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
    account_id = ""  # PR 3 stub for runtime-auth routes; PR 4 derives from connection
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id, account_id=account_id)
        _check_runtime_scope(auth_connector, connection.connector)
        if not await queries.is_session_bound_to_connection(
            conn,
            connection_id=body.connection_id,
            session_id=body.session_id,
            account_id=account_id,
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
            account_id=account_id,
        )
    await defer_wake(pool, body.session_id, cause="connector_tool_result", account_id=account_id)
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

    The bearer's connector must match the connection's connector
    type.  Returns ``{"secrets": {}}`` when none are configured —
    callers decide whether that's acceptable.
    """
    _, auth_connector = auth
    account_id = ""  # PR 3 stub for runtime-auth routes; PR 4 derives from connection
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
    _check_runtime_scope(auth_connector, connection.connector)
    secrets = await connections_service.get_connection_secrets(
        pool, connection_id, crypto_box=crypto_box, account_id=account_id
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
    Each event is keyed ``call`` with a JSON body shaped::

        {
            "session_id": "...",
            "tool_call_id": "...",
            "name": "...",
            "arguments": "...",       // JSON string from the model
            "focal_channel": "...",
            "connection_id": "..."
        }

    The ``connection_id`` field lets the runtime container fan out to
    its per-connection workers client-side.
    """
    _, connector = auth
    return EventSourceResponse(
        runtime_connector_calls_stream(db_url, pool, connector),
        ping=15,
    )


@router.get(
    "/runtime/management-calls",
    openapi_extra={"x-codegen": {"targets": []}},
)
async def get_runtime_management_calls(
    db_url: DbUrlDep,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> EventSourceResponse:
    """SSE stream of pending management calls for the caller's connector type.

    Per-connector-type only (no session/connection scope).  Each event is
    keyed ``call`` with body ``{"call_id": "mgmt_...", "method": str, "params": dict}``.
    """
    _, connector = auth
    return EventSourceResponse(
        management_calls_stream(db_url, pool, connector),
        ping=15,
    )


# ─── operator-facing signal management routes ─────────────────────────


class SignalRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    captcha: str | None = None
    voice: bool = False


class SignalRegisterResponse(BaseModel):
    """``status="captcha_required"`` is a 200, not a 4xx — it's an actionable
    next step (solve the captcha, repost with the token), and 4xx would bury
    the URL inside FastAPI's error envelope."""

    account: str
    status: Literal["sms_sent", "voice_sent", "captcha_required"]
    captcha_url: str | None = None


class SignalVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    code: str
    pin: str | None = None


class SignalVerifyResponse(BaseModel):
    account: str
    uuid: str


class SignalProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str
    given_name: str | None = None
    family_name: str | None = None
    about: str | None = None


def _is_captcha_required(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "captcha_required"


async def _signal_management_call(
    db_url: str,
    pool: PoolDep,
    *,
    account_id: str,
    method: str,
    params: dict[str, Any],
    timeout_s: float,
) -> Any:
    result, is_error = await management_calls.submit_call(
        db_url,
        pool,
        connector="signal",
        method=method,
        params=params,
        timeout_s=timeout_s,
        account_id=account_id,
    )
    if is_error and not _is_captcha_required(result):
        raise ConnectorCallFailedError(
            f"signal connector failed {method!r}",
            detail={"method": method, "connector_error": result},
        )
    return result


@router.post(
    "/signal/register",
    operation_id="post_connector_signal_register",
)
async def post_signal_register(
    body: SignalRegisterRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> SignalRegisterResponse:
    """Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.
    """
    account_id, _, _ = _auth
    result = await _signal_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="register",
        params=body.model_dump(exclude_none=True),
        timeout_s=30.0,
    )
    if _is_captcha_required(result):
        return SignalRegisterResponse(
            account=body.account,
            status="captcha_required",
            captcha_url=result["captcha_url"],
        )
    return SignalRegisterResponse(
        account=body.account,
        status="voice_sent" if body.voice else "sms_sent",
    )


@router.post(
    "/signal/verify",
    operation_id="post_connector_signal_verify",
)
async def post_signal_verify(
    body: SignalVerifyRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> SignalVerifyResponse:
    """Submit the SMS / voice verification code.

    signal-cli writes the new account to its ``accounts.json``; the
    running connector picks it up on the next ``verify_phone`` call
    without restart.
    """
    account_id, _, _ = _auth
    result = await _signal_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="verify",
        params=body.model_dump(exclude_none=True),
        timeout_s=60.0,
    )
    return SignalVerifyResponse(account=body.account, uuid=str(result.get("uuid", "")))


@router.post(
    "/signal/profile",
    operation_id="post_connector_signal_profile",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def post_signal_profile(
    body: SignalProfileRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> None:
    """Update ``given_name`` / ``family_name`` / ``about``.  Avatar bytes
    are not supported in v1 (no operator→container file staging surface)."""
    account_id, _, _ = _auth
    await _signal_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="updateProfile",
        params=body.model_dump(exclude_none=True),
        timeout_s=30.0,
    )


# ─── runtime management-call result intake ────────────────────────────


class RuntimeManagementCallResultRequest(BaseModel):
    """Idempotent on ``call_id`` — a replay POST against an already-resolved
    row no-ops (no double-NOTIFY)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    result: Any = None
    is_error: bool = False


@router.post(
    "/runtime/management-call-results",
    operation_id="post_connector_runtime_management_call_result",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_management_call_result(
    body: RuntimeManagementCallResultRequest,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> None:
    _, auth_connector = auth
    account_id = ""  # PR 3 stub for runtime-auth routes; PR 4 derives from connection
    # Autocommit conn (no ``async with conn.transaction()``): the UPDATE
    # commits before the NOTIFY fires.  Don't wrap these in a
    # transaction — subscribers would see uncommitted state.  See
    # db/listen.py for the full rationale.
    async with pool.acquire() as conn:
        row = await queries.get_management_call(conn, body.call_id, account_id=account_id)
        if row is None:
            raise NotFoundError("no such management call", detail={"call_id": body.call_id})
        _check_runtime_scope(auth_connector, row["connector"])
        moved = await queries.mark_management_call_resolved(
            conn,
            call_id=body.call_id,
            result=body.result,
            is_error=body.is_error,
            account_id=account_id,
        )
        if not moved:
            return
        await queries.notify_management_call_result(
            conn, call_id=body.call_id, account_id=account_id
        )
