"""Connector-related endpoints — two distinct caller populations.

The file groups three sections:

1. **Runtime-container-facing** (``RuntimeAuthDep``, per-connector-type
   bearer): ``/runtime/inbound``, ``/runtime/tool-results``,
   ``/runtime/calls``, ``/runtime/secrets``, ``/connections``,
   ``/{connector}/tools_schema``, ``/runtime/management-calls``,
   ``/runtime/management-call-results``.  The bearer scopes the caller
   to one ``connector`` type; ``connection_id`` rides as a form/query
   field for the routes that operate on a specific connection.
2. **Operator-facing per-connector management** (``AuthDep``, operator
   API key):
   * Signal: ``/signal/register``, ``/signal/verify``, ``/signal/profile``.
   * WhatsApp: ``/whatsapp/start-pairing``, ``/whatsapp/confirm-pairing``,
     ``/whatsapp/unpair``.
   These block-await the connector's resolution via the
   ``connector_result_<call_id>`` LISTEN channel.

Section banners (``# ───`) below mark the boundary.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, assert_never

from fastapi import APIRouter, File, Form, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sse_starlette import EventSourceResponse

from aios.api.deps import (
    AccountIdDep,
    CryptoBoxDep,
    DbUrlDep,
    PoolDep,
    RuntimeAuthDep,
)
from aios.api.sse import (
    connection_discovery_stream,
    make_sse_response,
    management_calls_stream,
    preflight_subscription,
    runtime_connector_calls_stream,
)
from aios.db import queries
from aios.db.listen import (
    open_listen_for_connection_discovery,
    open_listen_for_connector_calls_by_type,
    open_listen_for_management_calls,
)
from aios.errors import (
    AiosError,
    ConnectorCallFailedError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitedError,
    ValidationError,
)
from aios.jobs.app import defer_wake
from aios.logging import get_logger
from aios.models.connections import ConnectorSecrets, inbound_orig_channel
from aios.models.connectors import ConnectorCapabilities
from aios.services import connections as connections_service
from aios.services import connectors as connectors_service
from aios.services import inbound as inbound_service
from aios.services import management_calls
from aios.services import sessions as sessions_service
from aios.services.attachment_staging import InboundAttachment
from aios.services.inbound_budget import (
    check_inbound_budget,
    check_inbound_budget_session,
)

log = get_logger("aios.api.routers.connectors")

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


# ─── connector-facing endpoints (#301) ──────────────────────────────────────


class ConnectorInboundResponse(BaseModel):
    """Response for ``POST /v1/connectors/inbound``."""

    appended_event_id: str | None
    session_id: str | None
    deduped: bool


def _inbound_drop_error(drop_reason: inbound_service.InboundDrop) -> AiosError:
    """Pick the right :class:`AiosError` subclass for a drop_reason.

    Each drop maps onto an existing error type — preserving HTTP status
    contracts without inventing a new error_type for every reason.
    Detail carries ``drop_reason`` so clients can branch on the
    machine-readable value.

    The ``match`` is exhaustive over :class:`InboundDrop` by construction:
    every member has an explicit arm and the final ``assert_never`` makes a
    newly-added member a ``mypy`` error rather than a silent catch-all 500.
    """
    detail = {"drop_reason": drop_reason.value}
    msg = f"inbound dropped ({drop_reason.value})"
    match drop_reason:
        case inbound_service.InboundDrop.PAYLOAD_TOO_LARGE:
            return PayloadTooLargeError(msg, detail=detail)
        case inbound_service.InboundDrop.RATE_LIMITED:
            # Per-counterparty inbound budget exceeded (#1504). 429 is a routine,
            # NON-FATAL drop for the connector-http runner: ``_is_fatal_inbound_status``
            # treats only 401/403/5xx as fatal (crash-restarts the container,
            # killing every sibling connection), so a single over-budget stranger
            # cannot take the container down — it just drops one envelope.
            return RateLimitedError(msg, detail=detail)
        case (
            inbound_service.InboundDrop.DETACHED
            | inbound_service.InboundDrop.ARCHIVED_TEMPLATE
            | inbound_service.InboundDrop.DENIED_BY_POLICY
        ):
            # DENIED_BY_POLICY maps to a non-fatal 422 (#1504): a denied stranger
            # must not be able to crash-restart the connector container.
            return ValidationError(msg, detail=detail)
        case inbound_service.InboundDrop.SESSION_MISSING:
            return NotFoundError(msg, detail=detail)
        case inbound_service.InboundDrop.ATTACHMENT_STAGING_FAILED:
            return AiosError(msg, detail=detail)  # base class → 500
        case _ as unreachable:
            assert_never(unreachable)


_DEFAULT_ATTACHMENT_CONTENT_TYPE = "application/octet-stream"


def _parse_form_json(field: str, raw: str | None, *, default: Any = None) -> Any:
    """Decode an optional JSON-in-multipart-form field that must be a dict.

    Multipart form values are always text; we expose ``sender`` and
    ``metadata`` as JSON-encoded strings so connector clients keep the
    shape the JSON inbound used (with JSON dicts) without hand-rolled
    field-flattening on either side. Raises :class:`ValidationError` on
    bad JSON OR on a JSON value that decoded to something other than
    a dict — downstream code (``_do_inbound`` and on into
    ``services.inbound.handle_inbound``) drills into the value with
    ``.get(...)``, so a list/scalar/null would crash at runtime as a
    500 AND poison the connector's retry loop (the ``event_id`` never
    reaches ``try_record_inbound_ack`` because the inbound flow dies
    before the dedup write). Rejecting at the boundary keeps the
    connector author's mistake visible as a 4xx.
    """
    if raw is None:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValidationError(
            f"{field!r} must be a JSON-encoded string",
            detail={"field": field, "error": str(err)},
        ) from err
    if not isinstance(parsed, dict):
        raise ValidationError(
            f"{field!r} must decode to a JSON object",
            detail={"field": field, "actual_type": type(parsed).__name__},
        )
    return parsed


async def _do_inbound(
    pool: Any,
    *,
    account_id: str,
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
        account_id=account_id,
    )
    if result.drop_reason is not None:
        raise _inbound_drop_error(result.drop_reason)
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


class CapabilitiesUpdate(BaseModel):
    """Body for ``PUT /v1/connectors/{connector}/capabilities``.

    A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
    is decoupled from a full ``tools_schema`` republish and the shipped
    ``tools_schema`` body contract stays untouched.
    """

    model_config = ConfigDict(extra="forbid")

    capabilities: ConnectorCapabilities


class RuntimeToolResultRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/tool-results``.

    Carries ``connection_id`` explicitly — the bearer scopes the
    caller to a connector *type*, not to one connection, so the body
    has to name the target connection.
    """

    # extra="ignore" (forward-tolerant): a connector deployed AHEAD of the api
    # on a coupled-schema change can send a field this api does not yet know;
    # ignore it (known fields still validated/processed) instead of 422-ing,
    # which would crash-loop the connector and wedge the session. This is the
    # forward half of the #1398 deploy-skew symmetry (the backward half:
    # defaulting an omitted field for an older connector).
    model_config = ConfigDict(extra="ignore")

    connection_id: str
    session_id: str
    tool_call_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False
    # The connector declares (via ``@tool(fire_and_forget=True)``) that a
    # *successful* result for this tool is a delivery confirmation the model
    # has nothing to react to (a message send/reaction ack).  When set, the
    # result is still appended to the log — the model sees it — but the
    # session is NOT woken to react to its own send (closes the duplicate-send
    # loop).  A failed result NEVER carries this (the runner only sets it
    # on the post-success path); the intake also AND-gates it with
    # ``not is_error`` so a failure always wakes.  Default False keeps a
    # not-yet-redeployed connector's omitted field behaving exactly as before.
    no_reaction: bool = False


class RuntimeLifecycleRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/lifecycle``.

    Lets a connector emit a lifecycle event onto each session bound to
    ``connection_id`` — used today for "the underlying transport just
    went away" notifications (WhatsApp daemon crashed, peer logged the
    device out, etc.) so the model sees the connection-broken state in
    its context instead of silently failing the next outbound.

    ``event`` is a connector-namespaced kind ("whatsapp.connection.lost",
    "signal.daemon.exited") — the connector chooses the vocabulary.
    ``reason`` is an optional short tag the harness surfaces alongside
    the event for the model to act on ("daemon_crashed", "peer_logout").
    ``data`` is an optional free-form dict for connector-specific
    context (current device count, last successful timestamp, etc.).
    """

    # extra="ignore" (forward-tolerant): a connector deployed AHEAD of the api
    # on a coupled-schema change can send a field this api does not yet know;
    # ignore it (known fields still validated/processed) instead of 422-ing,
    # which would crash-loop the connector and wedge the session. This is the
    # forward half of the #1398 deploy-skew symmetry (the backward half:
    # defaulting an omitted field for an older connector).
    model_config = ConfigDict(extra="ignore")

    connection_id: str
    event: str
    reason: str | None = None
    data: dict[str, Any] | None = None


class RuntimeSessionLifecycleRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/session-lifecycle`` (#1261).

    The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
    Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
    notice across *every* session bound to the connection, this appends a
    single ``kind=lifecycle`` event onto **one** named session — the gap
    called out by the SMS design (§3.5 req 1): a delivery failure must reach
    the *originating* session, not be broadcast.

    ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
    isn't merely visible-on-next-turn but actually wakes the session (the
    "give it stimulus" half of the design's option (a)). Defaults ``False``
    so the primitive stays a plain visible-on-next-wake append unless the
    caller opts into the wake.
    """

    # extra="ignore" (forward-tolerant): a connector deployed AHEAD of the api
    # on a coupled-schema change can send a field this api does not yet know;
    # ignore it (known fields still validated/processed) instead of 422-ing,
    # which would crash-loop the connector and wedge the session. This is the
    # forward half of the #1398 deploy-skew symmetry (the backward half:
    # defaulting an omitted field for an older connector).
    model_config = ConfigDict(extra="ignore")

    connection_id: str
    session_id: str
    event: str
    reason: str | None = None
    data: dict[str, Any] | None = None
    wake: bool = False


class RuntimeChatLifecycleRequest(BaseModel):
    """Body for ``POST /v1/connectors/runtime/chat-lifecycle`` (#1260).

    The routing-key variant of :class:`RuntimeSessionLifecycleRequest`.
    Both target a *single* session (not the broadcast fan-out), but where
    the session-lifecycle route needs the caller to already hold the
    resolved ``session_id``, this route carries a per-peer **routing key**
    (``chat_id``) and resolves it through the connection's per-chat binding
    to the originating session server-side.

    This is the second option the SMS design (§3.5 req 1) calls out: "route
    the per-peer failure through the resolver on the callback's ``To``".  A
    Twilio status callback knows the peer number (→ ``chat_id``) but not the
    AIOS ``session_id`` — without this route the connector would have to do
    an extra round-trip (or maintain its own ``chat_id → session_id`` map)
    just to reach the originating per_chat session.  The broadcast
    ``/runtime/lifecycle`` route stays for genuine connection-wide events.

    ``chat_id`` is the connector's per-peer routing key, the same value the
    inbound path stamps onto ``chat_sessions``.  It must resolve to an
    existing per-chat binding on ``connection_id`` — a routing key with no
    bound session 404s rather than fanning a spurious cross-peer notice (the
    design's "if a correlation row is genuinely missing … drop rather than
    fan a spurious cross-peer failure", §3.5).

    ``wake`` mirrors the session-lifecycle route: ``True`` pairs the append
    with a ``defer_wake`` so the failure wakes the originating session;
    defaults ``False`` (visible-on-next-turn).
    """

    # extra="ignore" (forward-tolerant): a connector deployed AHEAD of the api
    # on a coupled-schema change can send a field this api does not yet know;
    # ignore it (known fields still validated/processed) instead of 422-ing,
    # which would crash-loop the connector and wedge the session. This is the
    # forward half of the #1398 deploy-skew symmetry (the backward half:
    # defaulting an omitted field for an older connector).
    model_config = ConfigDict(extra="ignore")

    connection_id: str
    chat_id: str
    event: str
    reason: str | None = None
    data: dict[str, Any] | None = None
    wake: bool = False


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


def _check_runtime_connection_scope(
    auth_connection_ids: list[str] | None,
    target_connection_id: str,
) -> None:
    """Raise 403 if a runtime bearer with a non-``None`` allowlist reaches
    outside its ``connection_ids`` scope (#350).

    ``None`` means the token is unscoped — no check is performed.
    A non-``None`` list (including ``[]``) restricts the bearer to
    its listed IDs only; anything else 403s.
    """
    if auth_connection_ids is None:
        return
    if target_connection_id not in auth_connection_ids:
        raise ForbiddenError(
            "runtime token not authorized for this connection",
            detail={"target_connection_id": target_connection_id},
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
    path's ``connector``.  ``connection_ids`` allowlist is NOT enforced
    here — the tools schema is a connector-type-wide registration, not
    a per-connection operation.
    """
    _, auth_connector, account_id, _scope = auth
    _check_runtime_scope(auth_connector, connector)
    # Authorization: connectors are root-owned (the connector type IS
    # the configuration).  Child tenants only add connections; they
    # must not be able to publish a tools_schema that propagates into
    # every other tenant's session prelude.  The check lives in the
    # service layer so the route stays a thin shim.
    await connectors_service.update_tools_schema(
        pool, connector=connector, account_id=account_id, tools_schema=body.tools
    )


@router.put(
    "/{connector}/capabilities",
    operation_id="put_connector_capabilities",
)
async def put_capabilities(
    connector: str,
    body: CapabilitiesUpdate,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> None:
    """Publish the runtime container's typed capability descriptor for a
    connector type.

    A sibling to :func:`put_tools_schema` (same root-only publication path),
    kept on its own route so capability churn is decoupled from a full
    ``tools_schema`` republish.  The runtime container is the source of truth
    for what richer renderings it supports; it publishes the descriptor at
    startup, replacing whatever was on the ``connectors.capabilities`` row
    wholesale.

    Authorization: the runtime bearer's ``connector`` must match the path's
    ``connector``; publication itself is root-only (enforced in the service
    layer — connectors are root-owned, the same cross-tenant rationale as
    ``tools_schema``).  Capabilities declare NO authority: they constrain
    RENDERING, never what any principal may invoke.
    """
    _, auth_connector, account_id, _scope = auth
    _check_runtime_scope(auth_connector, connector)
    await connectors_service.update_capabilities(
        pool, connector=connector, account_id=account_id, capabilities=body.capabilities
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

        {"event": "added" | "removed",
         "connection_id": "...",
         "external_account_id": "..."}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    When the bearer carries a ``connection_ids`` allowlist (#350), the
    backfill and tail both filter to that set — out-of-scope IDs are
    silently omitted (not 403'd) so the runtime container's discovery
    loop just doesn't see them.
    """
    _, connector, account_id, auth_connection_ids = auth
    subscription = await preflight_subscription(
        open_listen_for_connection_discovery(db_url, connector),
        stream_name="connection_discovery",
        log_key="sse.connection_discovery.preflight_failed",
        log_fields={"connector": connector},
        log=log,
    )
    return make_sse_response(
        subscription,
        connection_discovery_stream(
            subscription,
            pool,
            connector,
            account_id=account_id,
            connection_ids=auth_connection_ids,
        ),
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
    type.  When the bearer carries a ``connection_ids`` allowlist
    (#350), the form field must also be on the list — otherwise 403.
    Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).
    """
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
    _check_runtime_scope(auth_connector, connection.connector)
    _check_runtime_connection_scope(auth_connection_ids, connection_id)
    return await _do_inbound(
        pool,
        account_id=account_id,
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
    "/runtime/lifecycle",
    operation_id="post_connector_runtime_lifecycle",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_lifecycle(
    body: RuntimeLifecycleRequest,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> dict[str, Any]:
    """Append a ``kind=lifecycle`` event onto every session bound to
    ``body.connection_id``.

    Authorization mirrors the inbound + tool-result paths: the bearer's
    connector must match ``body.connection_id``'s connector, and any
    bearer-side allowlist must include the connection_id (#350).

    Returns ``{"appended_session_ids": [...]}`` enumerating the sessions
    that received the event.  An empty list means no sessions were
    bound at the time of the call (e.g. the operator detached every
    session before the connector finished tearing down); not an error.

    When a session is archived between the binding snapshot
    (``list_session_ids_for_connection``, taken outside the append loop)
    and its append, ``append_event`` raises a typed ``NotFoundError``;
    that session is skipped and reported under ``skipped_session_ids``
    (a flat ``list[str]``, present only when non-empty).  This is the one
    benign per-session failure.  Any *other* append failure —
    serialization, statement timeout, pool exhaustion, a broken
    connection — is a real writer fault: it propagates uncaught and 500s
    the call so the connector can retry, rather than being buried in a
    201 body.
    """
    # Per-counterparty inbound budget (#1504): the broadcast lifecycle route is
    # deliberately LEFT UNCAPPED. It is connection-grain operator-control-plane
    # fan-out (it fans a notice across *every* bound session and carries no
    # ``chat_id``), NOT a per-counterparty inbound surface — there is no
    # ``(connection_id, chat_id)`` counterparty to bound. The per-counterparty
    # budget is applied on the two single-target wake routes
    # (``post_runtime_session_lifecycle`` / ``post_runtime_chat_lifecycle``) and
    # on ``handle_inbound`` instead.
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id, account_id=account_id)
        _check_runtime_scope(auth_connector, connection.connector)
        _check_runtime_connection_scope(auth_connection_ids, body.connection_id)
        # Both binding lineages: the active single_session binding on
        # this connection AND any per-chat-spawned chat_sessions rows.
        # list_session_ids_for_connection unions+dedups.  Without the
        # union the smoke (single_session-bound bot) saw the endpoint
        # silently succeed against an empty list.
        session_ids = await queries.list_session_ids_for_connection(
            conn,
            body.connection_id,
            account_id=account_id,
        )
    payload: dict[str, Any] = {
        "event": body.event,
        "connection_id": body.connection_id,
        "connector": connection.connector,
    }
    if body.reason is not None:
        payload["reason"] = body.reason
    if body.data is not None:
        payload["data"] = body.data
    # Archived/missing sessions are the one benign per-session failure: the
    # broadcast snapshot (list_session_ids_for_connection, above) is taken
    # outside the append loop, so a session may be archived between snapshot and
    # append.  append_event raises a TYPED NotFoundError for that case
    # (queries.append_event: UPDATE … WHERE archived_at IS NULL matches no row);
    # skip it and report which sessions were skipped.  EVERY OTHER failure —
    # serialization, statement timeout, pool exhaustion, a broken connection —
    # is a real writer fault the connector must be able to retry on, so it
    # propagates as a 500 (fail hard).  The single-target session-/chat-
    # lifecycle siblings already let append_event surface uncaught; this is the
    # broadcast analogue, narrowed to the benign kind only.
    appended: list[str] = []
    skipped: list[str] = []
    for sess_id in session_ids:
        try:
            await sessions_service.append_event(
                pool,
                sess_id,
                "lifecycle",
                payload,
                account_id=account_id,
            )
            appended.append(sess_id)
        except NotFoundError:
            skipped.append(sess_id)
    result: dict[str, Any] = {"appended_session_ids": appended}
    if skipped:
        result["skipped_session_ids"] = skipped
    return result


@router.post(
    "/runtime/session-lifecycle",
    operation_id="post_connector_runtime_session_lifecycle",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_session_lifecycle(
    body: RuntimeSessionLifecycleRequest,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> dict[str, Any]:
    """Append a ``kind=lifecycle`` event onto **one** session bound to
    ``body.connection_id`` (#1261), optionally waking it.

    The per-session-targeted sibling of the broadcast ``/runtime/lifecycle``
    route: where that fans a notice across every bound session, this targets
    the single ``body.session_id`` — the SMS design's §3.5 req 1 (a delivery
    failure must reach the *originating* session, not be broadcast).

    Authorization mirrors ``post_runtime_tool_result`` exactly: the bearer's
    connector must match ``body.connection_id``'s connector, any bearer-side
    ``connection_ids`` allowlist must include it (#350), and the session must
    be genuinely bound to that connection — so a runtime bearer can only
    target a session within its own connections.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the exact pattern as the tool-result intake) so the failure wakes the
    session rather than merely being visible on its next turn.

    Reserved model-visible ``event`` values a connector may post here:
    ``connector_delivery_failed`` (#1308, the failure path), and its
    success-path complements ``connector_message_delivered`` /
    ``connector_message_edited`` (#1341, informational acks emitted with
    ``wake=False``). All three render as a bracketed user-role notice; any
    other ``event`` string is appended but filtered out of the model context
    by the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist.
    """
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id, account_id=account_id)
        _check_runtime_scope(auth_connector, connection.connector)
        _check_runtime_connection_scope(auth_connection_ids, body.connection_id)
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
    payload: dict[str, Any] = {
        "event": body.event,
        "connection_id": body.connection_id,
        "connector": connection.connector,
    }
    if body.reason is not None:
        payload["reason"] = body.reason
    if body.data is not None:
        payload["data"] = body.data
    # Per-counterparty inbound rate/cost budget (#1504), gated on the
    # wake-bearing path only. The cost the budget bounds is the inference, and
    # a ``wake=False`` append (visible-on-next-turn, no ``defer_wake``) triggers
    # none — so a no-wake lifecycle is NOT throttled. The check runs AFTER the
    # auth/scope checks above (an out-of-scope token still 403s first) and
    # BEFORE the append, so an over-budget wake writes no event row. The
    # session-lifecycle route carries ``session_id`` but no ``chat_id``; a
    # wake-bearing session-lifecycle maps to exactly one session, so the budget
    # is keyed on ``(account_id, session_id)``. The rejection is a non-fatal 429
    # (``_is_fatal_inbound_status(429) is False``) so a throttle never
    # crash-restarts the connector container. Disabled by default (no query).
    if body.wake and not await check_inbound_budget_session(
        pool, account_id=account_id, session_id=body.session_id
    ):
        raise RateLimitedError(
            "inbound rate budget exceeded for this session",
            detail={"drop_reason": "rate_limited", "session_id": body.session_id},
        )
    if body.wake:
        # Stamp the wake intent that fired the ``defer_wake`` so the budget's
        # session-grain window can meter this inference-bearing write (#1558).
        # The no-wake append stays byte-identical (no marker, uncounted); only
        # ``wake=True`` lifecycle rows carry ``wake`` so internal harness
        # lifecycle transitions (no ``wake`` key) never burn a budget.
        payload["wake"] = True
    event = await sessions_service.append_event(
        pool,
        body.session_id,
        "lifecycle",
        payload,
        account_id=account_id,
    )
    if body.wake:
        await defer_wake(
            pool,
            body.session_id,
            cause="connector_lifecycle",
            account_id=account_id,
        )
    # Mirror the broadcast route's payload shape (a single-element list keeps
    # callers that already handle ``appended_session_ids`` uniform), plus the
    # appended event id and whether a wake was enqueued so the connector can
    # log/correlate without a second read.
    return {
        "appended_session_ids": [body.session_id],
        "event_id": event.id,
        "woke": body.wake,
    }


@router.post(
    "/runtime/chat-lifecycle",
    operation_id="post_connector_runtime_chat_lifecycle",
    status_code=status.HTTP_201_CREATED,
)
async def post_runtime_chat_lifecycle(
    body: RuntimeChatLifecycleRequest,
    pool: PoolDep,
    auth: RuntimeAuthDep,
) -> dict[str, Any]:
    """Append a ``kind=lifecycle`` event onto the single session that
    ``body.chat_id`` resolves to on ``body.connection_id`` (#1260),
    optionally waking it.

    The routing-key sibling of ``/runtime/session-lifecycle``: where that
    needs the resolved ``session_id``, this carries the connector's per-peer
    routing key (``chat_id``) and resolves it through the connection's
    per-chat binding server-side — the SMS design's §3.5 req 1 second option
    ("route the per-peer failure through the resolver on the callback's
    ``To``").  Like the session-lifecycle route it targets exactly one
    session, NOT the broadcast fan-out: a per-peer delivery failure must not
    pollute unrelated ``per_chat`` sessions.

    Authorization mirrors the session-lifecycle route: the bearer's
    connector must match ``body.connection_id``'s connector and any
    bearer-side ``connection_ids`` allowlist must include it (#350).  The
    binding lookup itself is the per-session authorization — a ``chat_id``
    that has no per-chat session on this connection 404s (no spurious
    cross-peer append), and the resolution is scoped to the bearer's
    ``account_id``.

    When ``body.wake`` is set, a ``defer_wake`` is enqueued after the append
    (the same pattern as the session-lifecycle and tool-result intakes) so
    the failure wakes the originating session rather than merely being
    visible on its next turn.

    Reserved model-visible ``event`` values mirror the session-lifecycle
    route: ``connector_delivery_failed`` (#1308) and its success-path
    complements ``connector_message_delivered`` / ``connector_message_edited``
    (#1341, informational acks emitted with ``wake=False``).
    """
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id, account_id=account_id)
        _check_runtime_scope(auth_connector, connection.connector)
        _check_runtime_connection_scope(auth_connection_ids, body.connection_id)
        # Resolve the per-peer routing key to its bound session. A missing
        # row means the chat was never bound on this connection (past
        # retention, or a routing key that never spawned a session) — drop
        # with a 404 rather than fanning a spurious cross-peer notice (§3.5).
        row = await queries.get_chat_session_row(
            conn,
            body.connection_id,
            body.chat_id,
            account_id=account_id,
        )
    if row is None:
        raise NotFoundError(
            "no session bound to this chat_id on the connection",
            detail={"connection_id": body.connection_id, "chat_id": body.chat_id},
        )
    _chat_id, session_id, _created_at = row
    payload: dict[str, Any] = {
        "event": body.event,
        "connection_id": body.connection_id,
        "connector": connection.connector,
        "chat_id": body.chat_id,
    }
    if body.reason is not None:
        payload["reason"] = body.reason
    if body.data is not None:
        payload["data"] = body.data
    # Per-counterparty inbound rate/cost budget (#1504), wake-bearing path only.
    # The chat-lifecycle route carries ``body.chat_id`` and the loaded
    # ``connection`` (hence ``connector`` + ``external_account_id``), so the
    # budget keys on the same ``orig_channel`` window as the inbound path — a
    # per-``(connection_id, chat_id)`` counterparty. ``wake=False`` (no
    # inference) is NOT throttled. Runs after auth/scope and before the append,
    # so an over-budget wake writes no event row; the rejection is a non-fatal
    # 429 (a throttle never crash-restarts the connector). Disabled by default.
    if body.wake and not await check_inbound_budget(
        pool,
        account_id=account_id,
        connector=connection.connector,
        external_account_id=connection.external_account_id,
        chat_id=body.chat_id,
    ):
        raise RateLimitedError(
            "inbound rate budget exceeded for this chat",
            detail={
                "drop_reason": "rate_limited",
                "connection_id": body.connection_id,
                "chat_id": body.chat_id,
            },
        )
    # On the wake-bearing path, stamp the wake intent AND the per-counterparty
    # ``orig_channel`` key so the chat-grain budget window counts this
    # inference-bearing write on the same key the inbound path uses (#1558).
    # ``_resolve_event_channel`` returns NULL for any non-``message`` kind, so
    # stamping ``orig_channel`` on this ``kind='lifecycle'`` row is safe for the
    # derived ``channel`` column. The no-wake append stays byte-identical (no
    # marker, no ``orig_channel`` → uncounted).
    lifecycle_orig_channel: str | None = None
    if body.wake:
        payload["wake"] = True
        lifecycle_orig_channel = inbound_orig_channel(
            connection.connector, connection.external_account_id, body.chat_id
        )
    event = await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        payload,
        account_id=account_id,
        orig_channel=lifecycle_orig_channel,
    )
    if body.wake:
        await defer_wake(
            pool,
            session_id,
            cause="connector_lifecycle",
            account_id=account_id,
        )
    # Mirror the session-lifecycle route's shape (single-element list keeps
    # callers handling ``appended_session_ids`` uniform), plus the resolved
    # session_id, the appended event id, and whether a wake was enqueued.
    return {
        "appended_session_ids": [session_id],
        "session_id": session_id,
        "event_id": event.id,
        "woke": body.wake,
    }


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

    Authorization: the bearer's connector must match
    ``body.connection_id``'s connector, the session must be bound to
    that connection, and (when the bearer carries a ``connection_ids``
    allowlist) ``body.connection_id`` must be on the list (#350).
    """
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, body.connection_id, account_id=account_id)
        _check_runtime_scope(auth_connector, connection.connector)
        _check_runtime_connection_scope(auth_connection_ids, body.connection_id)
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
        # ``no_reaction`` stamps the row so the wake GATE excludes it: a successful
        # fire-and-forget result is not itself a stimulus to react to. A failure must
        # react (so the model can recover), hence AND-gate with ``not is_error``.
        no_reaction = body.no_reaction and not body.is_error
        event = await sessions_service.append_tool_result(
            conn,
            session_id=body.session_id,
            tool_call_id=body.tool_call_id,
            content=body.content,
            is_error=body.is_error,
            no_reaction=no_reaction,
            account_id=account_id,
        )
    # Always append (tool-always-appends-result) and always wake: the wake GATE — not
    # this intake — decides whether to infer. The gate excludes the ``no_reaction`` row,
    # so a LONE delivery confirmation makes the woken step re-gate and no-op (settles, no
    # re-inference on its own ack); but a ``no_reaction`` result COMPLETING a mixed batch
    # still surfaces the unreacted real sibling, which must run. This intake can't see the
    # worker's in-flight sibling tasks, so it cannot make that call itself.
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
    type; when the bearer carries a ``connection_ids`` allowlist
    (#350), ``connection_id`` must be on the list.  Returns
    ``{"secrets": {}}`` when none are configured — callers decide
    whether that's acceptable.
    """
    _, auth_connector, account_id, auth_connection_ids = auth
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
    _check_runtime_scope(auth_connector, connection.connector)
    _check_runtime_connection_scope(auth_connection_ids, connection_id)
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
            "connection_id": "...",
            "workspace_path": "..."  // host-side bind-mount source for
                                     // /workspace; the SDK uses it to
                                     // resolve SandboxPath args
        }

    The ``connection_id`` field lets the runtime container fan out to
    its per-connection workers client-side.  When the bearer carries
    a ``connection_ids`` allowlist (#350), backfill and tail both
    filter to that set — out-of-scope calls are silently omitted.
    """
    _, connector, account_id, auth_connection_ids = auth
    subscription = await preflight_subscription(
        open_listen_for_connector_calls_by_type(db_url, connector),
        stream_name="runtime_calls",
        log_key="sse.runtime_calls.preflight_failed",
        log_fields={"connector": connector},
        log=log,
    )
    return make_sse_response(
        subscription,
        runtime_connector_calls_stream(
            subscription,
            pool,
            connector,
            account_id=account_id,
            connection_ids=auth_connection_ids,
        ),
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
    ``connection_ids`` allowlist is NOT enforced here — management
    calls are connector-type-wide, not per-connection.
    """
    _, connector, account_id, _scope = auth
    subscription = await preflight_subscription(
        open_listen_for_management_calls(db_url, connector),
        stream_name="management_calls",
        log_key="sse.management_calls.preflight_failed",
        log_fields={"connector": connector},
        log=log,
    )
    return make_sse_response(
        subscription,
        management_calls_stream(subscription, pool, connector, account_id=account_id),
    )


# ─── operator-facing signal management routes ─────────────────────────


class SignalRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str
    captcha: str | None = None
    voice: bool = False


class SignalRegisterResponse(BaseModel):
    """``status="captcha_required"`` is a 200, not a 4xx — it's an actionable
    next step (solve the captcha, repost with the token), and 4xx would bury
    the URL inside FastAPI's error envelope."""

    external_account_id: str
    status: Literal["sms_sent", "voice_sent", "captcha_required"]
    captcha_url: str | None = None


class SignalVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str
    code: str
    pin: str | None = None


class SignalVerifyResponse(BaseModel):
    external_account_id: str
    uuid: str


class SignalProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str
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
    account_id: AccountIdDep,
) -> SignalRegisterResponse:
    """Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.
    """
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
            external_account_id=body.external_account_id,
            status="captcha_required",
            captcha_url=result["captcha_url"],
        )
    return SignalRegisterResponse(
        external_account_id=body.external_account_id,
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
    account_id: AccountIdDep,
) -> SignalVerifyResponse:
    """Submit the SMS / voice verification code.

    signal-cli writes the new account to its ``accounts.json``; the
    running connector picks it up on the next ``verify_phone`` call
    without restart.
    """
    result = await _signal_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="verify",
        params=body.model_dump(exclude_none=True),
        timeout_s=60.0,
    )
    return SignalVerifyResponse(
        external_account_id=body.external_account_id, uuid=str(result.get("uuid", ""))
    )


@router.post(
    "/signal/profile",
    operation_id="post_connector_signal_profile",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def post_signal_profile(
    body: SignalProfileRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> None:
    """Update ``given_name`` / ``family_name`` / ``about``.  Avatar bytes
    are not supported in v1 (no operator→container file staging surface)."""
    await _signal_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="updateProfile",
        params=body.model_dump(exclude_none=True),
        timeout_s=30.0,
    )


# ─── operator-facing whatsapp management routes ───────────────────────


class WhatsappStartPairingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str


class WhatsappStartPairingResponse(BaseModel):
    external_account_id: str
    code: str


class WhatsappPairingCodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str


class WhatsappPairingCodeResponse(BaseModel):
    """The QR code currently live for the in-flight pairing attempt.

    ``rotation_seq`` increments each time whatsmeow rotates the code
    (~every 20 s); operators poll this endpoint every few seconds and
    re-render the QR when ``rotation_seq`` changes."""

    external_account_id: str
    code: str
    rotation_seq: int


class WhatsappConfirmPairingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str


class WhatsappConfirmPairingResponse(BaseModel):
    """``status`` is the terminal pairing outcome; ``jid`` / ``push_name``
    are populated on success, ``reason`` on error / timeout."""

    external_account_id: str
    status: Literal["success", "timeout", "error"]
    jid: str | None = None
    push_name: str | None = None
    reason: str | None = None


class WhatsappUnpairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_account_id: str


async def _whatsapp_management_call(
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
        connector="whatsapp",
        method=method,
        params=params,
        timeout_s=timeout_s,
        account_id=account_id,
    )
    if is_error:
        # The connector emits a structured ``no_active_connection``
        # payload when the operator's external_account_id doesn't
        # match any running connection.  Surface as 404 so operator
        # tooling can discriminate "wrong target" from "daemon crashed".
        if isinstance(result, dict) and result.get("status") == "no_active_connection":
            raise NotFoundError(
                f"no active whatsapp connection for {result.get('external_account_id')!r}",
                detail={"external_account_id": result.get("external_account_id")},
            )
        raise ConnectorCallFailedError(
            f"whatsapp connector failed {method!r}",
            detail={"method": method, "connector_error": result},
        )
    return result


@router.post(
    "/whatsapp/start-pairing",
    operation_id="post_connector_whatsapp_start_pairing",
)
async def post_whatsapp_start_pairing(
    body: WhatsappStartPairingRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> WhatsappStartPairingResponse:
    """Initiate QR pairing.  Returns the first QR code; the operator
    scans within whatsmeow's QR window (~20 s before rotation, ~100 s
    total).  The follow-up ``/confirm-pairing`` blocks until the
    pairing terminates."""
    result = await _whatsapp_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="startPairing",
        params=body.model_dump(exclude_none=True),
        timeout_s=30.0,
    )
    code = result.get("code") if isinstance(result, dict) else None
    if not code:
        # Fail loud rather than 200 OK with an empty QR — operators
        # printing the response would otherwise show a blank code with
        # no diagnostic.
        raise ConnectorCallFailedError(
            "whatsapp connector startPairing returned no code",
            detail={"method": "startPairing", "connector_result": result},
        )
    return WhatsappStartPairingResponse(
        external_account_id=body.external_account_id,
        code=str(code),
    )


@router.post(
    "/whatsapp/pairing-code",
    operation_id="post_connector_whatsapp_pairing_code",
)
async def post_whatsapp_pairing_code(
    body: WhatsappPairingCodeRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> WhatsappPairingCodeResponse:
    """Return the QR code currently live for the in-flight pairing
    attempt.  whatsmeow rotates the code ~every 20 s over the ~100 s
    attempt; ``/start-pairing`` surfaces only the first.  Operators poll
    this every few seconds and re-render when ``rotation_seq`` changes so
    each rotation is scannable, not just the first window.  404s when no
    attempt is live (none started, or already terminated)."""
    result = await _whatsapp_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="getPairingCode",
        params=body.model_dump(exclude_none=True),
        timeout_s=30.0,
    )
    code = result.get("code") if isinstance(result, dict) else None
    if not code:
        # Fail loud rather than 200 OK with an empty QR — see
        # post_whatsapp_start_pairing for the rationale.
        raise ConnectorCallFailedError(
            "whatsapp connector getPairingCode returned no code",
            detail={"method": "getPairingCode", "connector_result": result},
        )
    return WhatsappPairingCodeResponse(
        external_account_id=body.external_account_id,
        code=str(code),
        rotation_seq=int(result.get("rotation_seq", 0)),
    )


@router.post(
    "/whatsapp/confirm-pairing",
    operation_id="post_connector_whatsapp_confirm_pairing",
)
async def post_whatsapp_confirm_pairing(
    body: WhatsappConfirmPairingRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> WhatsappConfirmPairingResponse:
    """Block until pairing terminates.  Long-poll: up to ~180 s server-side
    so a slow scan doesn't trip the HTTP timeout before the daemon
    itself reports timeout."""
    result = await _whatsapp_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="confirmPairing",
        params=body.model_dump(exclude_none=True),
        timeout_s=240.0,
    )
    raw_status = result.get("status") if isinstance(result, dict) else None
    if raw_status not in ("success", "timeout", "error"):
        # Unknown / empty daemon status: surface as "error" with the
        # raw value as reason so the Pydantic Literal doesn't 500.
        return WhatsappConfirmPairingResponse(
            external_account_id=body.external_account_id,
            status="error",
            reason=(
                f"unrecognized daemon status: {raw_status!r}"
                if raw_status
                else "daemon returned empty status"
            ),
        )
    return WhatsappConfirmPairingResponse(
        external_account_id=body.external_account_id,
        status=raw_status,
        jid=result.get("jid"),
        push_name=result.get("push_name"),
        reason=result.get("reason"),
    )


@router.post(
    "/whatsapp/unpair",
    operation_id="post_connector_whatsapp_unpair",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def post_whatsapp_unpair(
    body: WhatsappUnpairRequest,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> None:
    """Log out the WhatsApp device on the server; clears the local
    sqlstore so the next ``/start-pairing`` provisions a fresh device."""
    await _whatsapp_management_call(
        db_url,
        pool,
        account_id=account_id,
        method="unpair",
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
    _, auth_connector, account_id, _scope = auth
    # ``connection_ids`` allowlist is NOT enforced — management calls
    # are connector-type-wide; the result intake is the matching peer
    # of ``get_runtime_management_calls`` which also doesn't filter.
    #
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
        await queries.notify_management_call_result(conn, call_id=body.call_id)
