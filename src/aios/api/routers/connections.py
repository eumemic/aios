"""Connection endpoints — CRUD plus mode-binding transitions.

Created in detached mode; switch to single_session via ``attach`` or
per_chat via ``configure-per-chat``.  Both write to the ``bindings``
table (one active row per connection); ``Connection.session_id`` /
``session_template_id`` on the wire are projected from the active
binding via a LEFT JOIN at read time, preserving the pre-#328-PR-7
shape.  ``DELETE`` soft-archives — but only on connections with no
active binding (the service layer enforces this so operators can't
silently break inbound delivery for live single_session connections or
strand the template a per_chat binding spawns from).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, CryptoBoxDep, PoolDep
from aios.models.common import ListResponse
from aios.models.connections import (
    BindChatRequest,
    BoundChat,
    Connection,
    ConnectionAttach,
    ConnectionConfigurePerChat,
    ConnectionCreate,
    ConnectionMode,
    ConnectionReparent,
    ConnectionSetSecrets,
    RecentChat,
)
from aios.services import connections as service

router = APIRouter(prefix="/v1/connections", tags=["connections"])


@router.post("", operation_id="create_connection", status_code=status.HTTP_201_CREATED)
async def create(
    body: ConnectionCreate, pool: PoolDep, crypto_box: CryptoBoxDep, account_id: AccountIdDep
) -> Connection:
    """Create a detached connection, **idempotent on ``(connector, external_account_id)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single
    row: posting twice with the same ``(connector, external_account_id)``
    returns 201 with the existing row rather than 409. The ``id`` may
    differ from a freshly-allocated one if a concurrent writer landed
    first; the response always reflects the canonical active row.

    The active-row partial-unique index is **per-account**, not global
    (migration 0060, in support of the reparent primitive #694): the
    same ``(connector, external_account_id)`` may live in multiple
    accounts simultaneously. The 409 ``conflict`` only fires within
    the caller's own account — cross-account collisions are no longer
    rejected here.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route — operator-facing
    reads return ``secrets_set: bool`` instead of values.
    """
    return await service.create_connection(
        pool,
        connector=body.connector,
        external_account_id=body.external_account_id,
        metadata=body.metadata,
        secrets=body.secrets,
        crypto_box=crypto_box,
        account_id=account_id,
    )


@router.put("/{connection_id}/secrets", operation_id="set_connection_secrets")
async def set_secrets(
    connection_id: str,
    body: ConnectionSetSecrets,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> Connection:
    """Replace the connection's encrypted secrets dict, wholesale.

    The request body fully replaces the stored blob.  Pass
    ``{"secrets": {}}`` to clear secrets entirely.  Operator-facing reads
    only ever expose ``secrets_set: bool``; the decrypted values are
    exclusively available via the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route to a connector container
    holding a runtime token for this connector type.
    """
    return await service.set_connection_secrets(
        pool, connection_id, secrets=body.secrets, crypto_box=crypto_box, account_id=account_id
    )


@router.get("", operation_id="list_connections")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    connector: str | None = None,
    session_id: str | None = None,
    mode: ConnectionMode | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
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
        limit=limit + 1,
        after=after,
        account_id=account_id,
    )
    return ListResponse[Connection].paginate(items, limit, cursor=lambda x: x.id)


@router.get("/{connection_id}", operation_id="get_connection")
async def get(connection_id: str, pool: PoolDep, account_id: AccountIdDep) -> Connection:
    """Fetch one connection by id."""
    return await service.get_connection(pool, connection_id, account_id=account_id)


@router.post("/{connection_id}/reparent", operation_id="reparent_connection")
async def reparent(
    connection_id: str,
    body: ConnectionReparent,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> Connection:
    """Transfer a connection to a different account. Root operator only.

    Moves ``connection.account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    daemon state (signal-cli's ``account.dat``, whatsmeow's
    ``sqlstore.db``, telegram webhook config) carries over without
    recreation. Encrypted secrets are re-keyed from the source
    account's derived subkey to the destination's inside the same
    transaction, so the post-reparent connection decrypts correctly
    under the destination context. The per-account partial unique
    index on ``(account_id, connector, external_account_id) WHERE
    archived_at IS NULL`` enforces no-collision at the destination
    automatically; a colliding destination returns 409.

    Authorization (v1): root operator only — the caller's account must
    have ``parent_account_id IS NULL``. Multi-tenant consent semantics
    ("both source and destination owners must approve") are deferred
    to v2; v1 is the operator-only escape hatch that unblocks the
    jarbot v2 ``ExternalIdentity`` transfer flow.

    **Daemon-cache caveat (v1)**: this is a database-only reparent.
    Connector daemons cache ``account_id`` in memory at attach time
    and do NOT receive a rebind event. Restart the connector container
    after reparent — the in-memory cache is otherwise stale until the
    next restart.
    """
    return await service.reparent_connection(
        pool,
        connection_id,
        destination_account_id=body.destination_account_id,
        requester_account_id=account_id,
        crypto_box=crypto_box,
    )


@router.delete(
    "/{connection_id}",
    operation_id="archive_connection",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete(connection_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive a connection (DELETE soft-archives, only on detached connections).

    The service layer rejects archive attempts while an active binding
    exists — archiving those would silently break inbound delivery for
    live sessions or strand the template a per_chat binding spawns
    from. Detach or unconfigure first, then archive.
    """
    await service.archive_connection(pool, connection_id, account_id=account_id)


@router.post("/{connection_id}/attach", operation_id="attach_connection")
async def attach(
    connection_id: str,
    body: ConnectionAttach,
    pool: PoolDep,
    account_id: AccountIdDep,
) -> Connection:
    """Attach a connection to a session (single_session mode).

    Inserts an active ``bindings`` row.  Operators bear the
    responsibility of binding a connection to a real, ongoing session;
    an inbound referencing an unknown account simply drops at the
    inbound boundary.
    """
    return await service.attach_connection(
        pool, connection_id, session_id=body.session_id, account_id=account_id
    )


@router.post(
    "/{connection_id}/detach",
    operation_id="detach_connection",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def detach(connection_id: str, pool: PoolDep, account_id: AccountIdDep) -> Connection:
    """Detach a single_session connection back to detached mode.

    The bound session is unaffected (continues to exist); only the
    connection's mode/binding changes. Inbound on this connection is
    paused until it's re-attached or configured for per_chat.
    """
    return await service.detach_connection(pool, connection_id, account_id=account_id)


@router.post(
    "/{connection_id}/configure-per-chat",
    operation_id="configure_connection_per_chat",
)
async def configure_per_chat(
    connection_id: str, body: ConnectionConfigurePerChat, pool: PoolDep, account_id: AccountIdDep
) -> Connection:
    """Move a detached connection into per_chat mode, pinned to a session template.

    Inbound from new chat partners on this connection will spawn fresh
    sessions using the named template (agent + environment + vaults +
    memory stores). Use ``unconfigure_connection`` to return to detached.
    """
    return await service.configure_per_chat(
        pool, connection_id, session_template_id=body.session_template_id, account_id=account_id
    )


@router.post(
    "/{connection_id}/unconfigure",
    operation_id="unconfigure_connection",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def unconfigure(connection_id: str, pool: PoolDep, account_id: AccountIdDep) -> Connection:
    """Return a per_chat connection to detached mode.

    Already-spawned sessions are unaffected and continue normally;
    operator-bound chat → session rows persist and continue to route
    those specific chats. Only the per_chat fallback (spawn for unseen
    chats) is removed — inbound from new chat partners is paused until
    the connection is reconfigured.
    """
    return await service.unconfigure_connection(pool, connection_id, account_id=account_id)


@router.post(
    "/{connection_id}/bind-chat",
    operation_id="bind_chat",
    status_code=status.HTTP_201_CREATED,
)
async def bind_chat(
    connection_id: str, body: BindChatRequest, pool: PoolDep, account_id: AccountIdDep
) -> BoundChat:
    """Operator-curate a chat → session mapping (#215).

    A row in ``chat_sessions`` overrides the connection's
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
        pool, connection_id, chat_id=body.chat_id, session_id=body.session_id, account_id=account_id
    )


@router.delete(
    "/{connection_id}/bind-chat/{chat_id}",
    operation_id="unbind_chat",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def unbind_chat(
    connection_id: str, chat_id: str, pool: PoolDep, account_id: AccountIdDep
) -> None:
    """Drop the operator-curated row.  Idempotent — repeat calls 204."""
    await service.unbind_chat(pool, connection_id, chat_id, account_id=account_id)


@router.get("/{connection_id}/bound-chats", operation_id="list_bound_chats")
async def bound_chats(
    connection_id: str, pool: PoolDep, account_id: AccountIdDep
) -> ListResponse[BoundChat]:
    """List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).
    """
    items = await service.list_bound_chats(pool, connection_id, account_id=account_id)
    return ListResponse[BoundChat](data=items)


@router.get("/{connection_id}/recent-chats", operation_id="list_recent_chats")
async def recent_chats(
    connection_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ListResponse[RecentChat]:
    """List chats that recently sent inbound on this connection, newest first.

    Useful for picking a ``chat_id`` to bind via ``bind_chat`` —
    enumerates the conversational counterparts that the connector has
    delivered messages from.
    """
    items = await service.list_recent_chats(pool, connection_id, limit=limit, account_id=account_id)
    return ListResponse[RecentChat](data=items)
