"""Connection endpoints — CRUD plus mode-binding transitions.

Created in detached mode; switch to single_session via ``attach`` or
per_chat via ``configure-per-chat``.  ``DELETE`` soft-archives — but
only on detached connections (the service layer enforces this so
operators can't silently break inbound delivery for live single_session
connections or orphan ``spawned_from_connection_id`` pointers on
per_chat-spawned sessions).

``attach`` validates the connection's ``account`` against the connector
subprocess's current snapshot (audit fix #1, design §5.6) — without
this guard an operator can permanently attach a connection for an
account the connector no longer serves, and inbound silently drops with
the ``account_drift`` counter ticking up.  The validation flows through
procrastinate RPC like the rest of ``/v1/connectors/*``: API mints a
ULID, LISTENs on ``connector_result_<call_id>``, defers
``harness.connector_status``, awaits NOTIFY.  ~50-200ms latency on an
admin-only endpoint is invisible.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, CryptoBoxDep, DbUrlDep, PoolDep
from aios.models.common import ListResponse
from aios.models.connections import (
    BindChatRequest,
    BoundChat,
    Connection,
    ConnectionAttach,
    ConnectionConfigurePerChat,
    ConnectionCreate,
    ConnectionMode,
    ConnectionSetSecrets,
    ConnectionSetTools,
    RecentChat,
)
from aios.services import connections as service

router = APIRouter(prefix="/v1/connections", tags=["connections"])


@router.post("", operation_id="create_connection", status_code=status.HTTP_201_CREATED)
async def create(
    body: ConnectionCreate, pool: PoolDep, crypto_box: CryptoBoxDep, _auth: AuthDep
) -> Connection:
    """Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the connector-scoped
    ``GET /v1/connectors/secrets`` route — operator-facing reads return
    ``secrets_set: bool`` instead of values.
    """
    return await service.create_connection(
        pool,
        connector=body.connector,
        account=body.account,
        metadata=body.metadata,
        tools=body.tools,
        secrets=body.secrets,
        crypto_box=crypto_box,
    )


@router.put("/{connection_id}/tools", operation_id="set_connection_tools")
async def set_tools(
    connection_id: str,
    body: ConnectionSetTools,
    pool: PoolDep,
    _auth: AuthDep,
) -> Connection:
    """Replace the connection's tools wholesale (#301).

    Tools declared on a connection become available to the model as
    ``type="custom"`` entries on every session this connection is
    attached to (single_session) or that this connection spawned
    (per_chat).  The model calls them, the session parks in
    ``requires_action``, the connector executes externally and POSTs the
    result back via ``/v1/sessions/:id/tool-results``.
    """
    return await service.set_connection_tools(pool, connection_id, tools=body.tools)


@router.put("/{connection_id}/secrets", operation_id="set_connection_secrets")
async def set_secrets(
    connection_id: str,
    body: ConnectionSetSecrets,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> Connection:
    """Replace the connection's encrypted secrets dict, wholesale.

    Mirrors ``set_tools`` — the request body fully replaces the stored
    blob.  Pass ``{"secrets": {}}`` to clear secrets entirely.
    Operator-facing reads only ever expose ``secrets_set: bool``; the
    decrypted values are exclusively available to the connector
    container that holds a connector token resolving to this connection.
    """
    return await service.set_connection_secrets(
        pool, connection_id, secrets=body.secrets, crypto_box=crypto_box
    )


@router.get("", operation_id="list_connections")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    connector: str | None = None,
    session_id: str | None = None,
    mode: ConnectionMode | None = None,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Connection]:
    """List connections, newest first, excluding archived. Cursor pagination via ``after``.

    Filters: ``connector`` (e.g. ``"telegram"``), ``session_id`` (only
    connections in single_session mode bound to that session), ``mode``
    (``detached`` / ``single_session`` / ``per_chat``). Filters compose.
    """
    items = await service.list_connections(
        pool,
        connector=connector,
        session_id=session_id,
        mode=mode,
        limit=limit,
        after=after,
    )
    return ListResponse[Connection](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{connection_id}", operation_id="get_connection")
async def get(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    """Fetch one connection by id."""
    return await service.get_connection(pool, connection_id)


@router.delete(
    "/{connection_id}",
    operation_id="archive_connection",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete(connection_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    """Archive a connection (DELETE soft-archives, only on detached connections).

    The service layer rejects archive attempts on ``single_session`` or
    ``per_chat`` connections — archiving those would silently break
    inbound delivery for live sessions or orphan
    ``spawned_from_connection_id`` pointers on per_chat-spawned sessions.
    Detach or unconfigure first, then archive.
    """
    await service.archive_connection(pool, connection_id)


@router.post("/{connection_id}/attach", operation_id="attach_connection")
async def attach(
    connection_id: str,
    body: ConnectionAttach,
    pool: PoolDep,
    db_url: DbUrlDep,
    _auth: AuthDep,
) -> Connection:
    """Attach a connection to a session.

    The legacy supervisor's "live snapshot drift check" was removed in
    #301 — the new architecture has connectors as peer services, so the
    api process can't probe a worker-side account snapshot.  Operators
    bear the responsibility of attaching to a real account; an inbound
    referencing an unknown account simply drops at the new
    ``/v1/connectors/inbound`` boundary.
    """
    return await service.attach_connection(pool, connection_id, session_id=body.session_id)


@router.post(
    "/{connection_id}/detach",
    operation_id="detach_connection",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def detach(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    """Detach a single_session connection back to detached mode.

    The bound session is unaffected (continues to exist); only the
    connection's mode/binding changes. Inbound on this connection is
    paused until it's re-attached or configured for per_chat.
    """
    return await service.detach_connection(pool, connection_id)


@router.post(
    "/{connection_id}/configure-per-chat",
    operation_id="configure_connection_per_chat",
)
async def configure_per_chat(
    connection_id: str, body: ConnectionConfigurePerChat, pool: PoolDep, _auth: AuthDep
) -> Connection:
    """Move a detached connection into per_chat mode, pinned to a session template.

    Inbound from new chat partners on this connection will spawn fresh
    sessions using the named template (agent + environment + vaults +
    memory stores). Use ``unconfigure_connection`` to return to detached.
    """
    return await service.configure_per_chat(
        pool, connection_id, session_template_id=body.session_template_id
    )


@router.post(
    "/{connection_id}/unconfigure",
    operation_id="unconfigure_connection",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def unconfigure(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    """Return a per_chat connection to detached mode.

    Already-spawned sessions are unaffected and continue normally;
    operator-bound chat → session rows persist and continue to route
    those specific chats. Only the per_chat fallback (spawn for unseen
    chats) is removed — inbound from new chat partners is paused until
    the connection is reconfigured.
    """
    return await service.unconfigure_connection(pool, connection_id)


@router.post(
    "/{connection_id}/bind-chat",
    operation_id="bind_chat",
    status_code=status.HTTP_201_CREATED,
)
async def bind_chat(
    connection_id: str, body: BindChatRequest, pool: PoolDep, _auth: AuthDep
) -> BoundChat:
    """Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).
    """
    return await service.bind_chat_to_session(
        pool, connection_id, chat_id=body.chat_id, session_id=body.session_id
    )


@router.delete(
    "/{connection_id}/bind-chat/{chat_id}",
    operation_id="unbind_chat",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def unbind_chat(connection_id: str, chat_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    """Drop the operator-curated row.  Idempotent — repeat calls 204."""
    await service.unbind_chat(pool, connection_id, chat_id)


@router.get("/{connection_id}/bound-chats", operation_id="list_bound_chats")
async def bound_chats(connection_id: str, pool: PoolDep, _auth: AuthDep) -> ListResponse[BoundChat]:
    """List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).
    """
    items = await service.list_bound_chats(pool, connection_id)
    return ListResponse[BoundChat](data=items)


@router.get("/{connection_id}/recent-chats", operation_id="list_recent_chats")
async def recent_chats(
    connection_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
) -> ListResponse[RecentChat]:
    """List chats that recently sent inbound on this connection, newest first.

    Useful for picking a ``chat_id`` to bind via ``bind_chat`` —
    enumerates the conversational counterparts that the connector has
    delivered messages from.
    """
    items = await service.list_recent_chats(pool, connection_id, limit=limit)
    return ListResponse[RecentChat](data=items)
