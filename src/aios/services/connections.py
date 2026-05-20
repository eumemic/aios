"""Business logic for connection resources.

Thin wrapper over :mod:`aios.db.queries`.  The single business rule
lives in :func:`archive_connection`: refuse to archive while the
connection is in single_session or per_chat mode.  Operators must
``detach`` (or ``unconfigure``) first — silently dropping the routing
binding on archive would interrupt outbound delivery for single_session
and stop spawning fresh sessions for per_chat.

Routing curation writes to the ``bindings`` table (one active row per
connection); reads of ``Connection.session_id`` / ``session_template_id``
project that row through a LEFT JOIN at query time, preserving the
pre-#328-PR-7 wire shape.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.crypto.vault import CryptoBox, EncryptedBlob
from aios.db import queries
from aios.errors import ConflictError, NotFoundError
from aios.models.connections import (
    BindingMode,
    BoundChat,
    Connection,
    ConnectionMode,
    RecentChat,
)


def _encrypt_secrets(
    secrets: dict[str, str] | None, crypto_box: CryptoBox, *, account_id: str
) -> EncryptedBlob | None:
    """Encrypt a secrets dict, or ``None`` if the dict is missing or empty.

    Empty / missing → no blob → schema columns NULL → ``secrets_set: False``.
    The operator surface treats ``None`` and ``{}`` identically (both
    "clear secrets") so create + PUT produce the same row state.

    Encryption is keyed to ``account_id`` via :meth:`CryptoBox.derive_account_subkey`
    so a connection's secrets can't be decrypted by another tenant's
    derived key even if the row leaks.
    """
    if not secrets:
        return None
    return crypto_box.derive_account_subkey(account_id).encrypt_dict(secrets)


async def create_connection(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connector: str,
    external_account_id: str,
    metadata: dict[str, Any],
    secrets: dict[str, str] | None = None,
    crypto_box: CryptoBox,
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.insert_connection(
            conn,
            connector=connector,
            external_account_id=external_account_id,
            metadata=metadata,
            secrets_blob=_encrypt_secrets(secrets, crypto_box, account_id=account_id),
            account_id=account_id,
        )


async def set_connection_secrets(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    account_id: str,
    secrets: dict[str, str],
    crypto_box: CryptoBox,
) -> Connection:
    """Replace a connection's encrypted secrets dict, wholesale.

    Pass an empty dict to clear secrets (columns go NULL, ``secrets_set``
    flips back to ``False``).  Running connector containers cache the
    decrypted dict at startup — restart them to pick up rotated values.
    """
    async with pool.acquire() as conn:
        return await queries.set_connection_secrets(
            conn,
            connection_id,
            secrets_blob=_encrypt_secrets(secrets, crypto_box, account_id=account_id),
            account_id=account_id,
        )


async def get_connection_secrets(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    account_id: str,
    crypto_box: CryptoBox,
) -> dict[str, str]:
    """Read and decrypt a connection's secrets.

    Returns an empty dict when no secrets are configured.  Raises
    :class:`NotFoundError` if the connection itself is missing or
    archived — connector containers shouldn't see "no secrets" when the
    underlying connection is gone; that's a deployment-state mismatch
    they should fail loudly on.
    """
    async with pool.acquire() as conn:
        blob = await queries.get_connection_secret_blob(conn, connection_id, account_id=account_id)
    if blob is None:
        return {}
    decoded = crypto_box.derive_account_subkey(account_id).decrypt_dict(blob)
    return {str(k): str(v) for k, v in decoded.items()}


async def list_tools_for_session(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> list[dict[str, Any]]:
    """Custom tool specs from every active connection bound to ``session_id``.

    Reached from the harness's per-step prelude via the ``ToolProvider``
    Protocol (#328): :class:`aios_connectors.providers.SubsystemToolProvider`
    delegates here, and ``compute_step_prelude`` calls the provider
    rather than this function directly. PR 7 swaps the SQL onto the
    new ``bindings`` ⨝ ``connectors.tools_schema`` tables.
    """
    async with pool.acquire() as conn:
        return await queries.list_connection_tools_for_session(
            conn, session_id, account_id=account_id
        )


async def get_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.get_connection(conn, connection_id, account_id=account_id)


async def list_connections(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connector: str | None = None,
    session_id: str | None = None,
    mode: ConnectionMode | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Connection]:
    async with pool.acquire() as conn:
        return await queries.list_connections(
            conn,
            connector=connector,
            session_id=session_id,
            mode=mode,
            limit=limit,
            after=after,
            account_id=account_id,
        )


async def attach_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str, session_id: str
) -> Connection:
    """Bind the connection in single_session mode by inserting an
    active ``bindings`` row.

    Returns the freshly-read connection with its derived
    ``session_id`` populated.  Race-safe via the partial-unique index
    on ``bindings (connection_id) WHERE archived_at IS NULL``.

    Refuses binding to an archived session — the FK constraint accepts
    archived rows (no partial ``WHERE archived_at IS NULL`` on the FK),
    so without this check ``insert_binding`` would succeed and the
    operator would receive a 200 with the archived ``session_id``.
    Subsequent inbounds would DETACH at the resolver (post-#541), but
    that's a delayed signal from a different actor; bind-time is the
    only point where the operator can correct the action that caused
    the misconfiguration.
    """
    async with pool.acquire() as conn:
        session = await queries.get_session(conn, session_id, account_id=account_id)
        if session.archived_at is not None:
            raise ConflictError(
                f"session {session_id} is archived; cannot attach a connection to it",
                detail={"id": connection_id, "session_id": session_id},
            )
        await queries.insert_binding(
            conn,
            connection_id=connection_id,
            mode="single_session",
            session_id=session_id,
            account_id=account_id,
        )
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
    # Second acquire so the NOTIFY fires OUTSIDE the insert's implicit
    # transaction — subscribers must never see a payload for an
    # uncommitted row.
    async with pool.acquire() as conn:
        await queries.notify_connection_change(
            conn,
            connector=connection.connector,
            connection_id=connection.id,
            external_account_id=connection.external_account_id,
            event="added",
            account_id=account_id,
        )
    return connection


async def detach_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str
) -> Connection:
    """Archive the connection's active single_session binding, returning
    the now-detached connection.

    Refuses (404 / 409) if the connection is missing, archived, or not
    in single_session mode.
    """
    async with pool.acquire() as conn:
        await _archive_binding_or_raise(
            conn, connection_id, expected_mode="single_session", account_id=account_id
        )
        return await queries.get_connection(conn, connection_id, account_id=account_id)


async def configure_per_chat(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str, session_template_id: str
) -> Connection:
    """Bind the connection in per_chat mode by inserting an active
    ``bindings`` row pointing at the session template.
    """
    async with pool.acquire() as conn:
        await queries.insert_binding(
            conn,
            connection_id=connection_id,
            mode="per_chat",
            session_template_id=session_template_id,
            account_id=account_id,
        )
        return await queries.get_connection(conn, connection_id, account_id=account_id)


async def unconfigure_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str
) -> Connection:
    """Archive the connection's active per_chat binding, returning the
    now-detached connection.

    Refuses (404 / 409) if the connection is missing, archived, or not
    in per_chat mode.
    """
    async with pool.acquire() as conn:
        await _archive_binding_or_raise(
            conn, connection_id, expected_mode="per_chat", account_id=account_id
        )
        return await queries.get_connection(conn, connection_id, account_id=account_id)


async def _archive_binding_or_raise(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
    expected_mode: BindingMode,
) -> None:
    """Archive the connection's active binding, raising on a mode mismatch.

    Happy path is a single mode-guarded UPDATE-RETURNING.  On miss we
    follow up with a connection read to diagnose: NotFound > archived
    > wrong mode (or detached).
    """
    archived = await queries.archive_active_binding(
        conn, connection_id, expected_mode=expected_mode, account_id=account_id
    )
    if archived is not None:
        return
    existing = await queries.get_connection(conn, connection_id, account_id=account_id)
    if existing.archived_at is not None:
        raise ConflictError(
            f"connection {connection_id} is archived",
            detail={"id": connection_id},
        )
    # The mode-guarded UPDATE may have missed either because there's no
    # active binding (detached) or because the active binding is in the
    # *other* mode.  Surface that distinction so operator dashboards can
    # branch — both are 409s, but with different recovery actions.
    current = await queries.get_active_binding(conn, connection_id, account_id=account_id)
    current_mode = current.mode if current is not None else "detached"
    raise ConflictError(
        f"connection {connection_id} is in {current_mode} mode, not {expected_mode}",
        detail={"id": connection_id, "current_mode": current_mode},
    )


async def bind_chat_to_session(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    account_id: str,
    chat_id: str,
    session_id: str,
) -> BoundChat:
    """Insert an operator-curated ``chat_sessions`` row.

    On conflict (a row already exists for this ``(connection_id, chat_id)``,
    either operator-bound earlier or per-chat-spawned) the existing row is
    preserved — the call is idempotent.  The returned ``BoundChat``
    reflects whichever ``session_id`` is now stored, which may differ from
    the requested one if the conflict path triggered.
    """
    async with pool.acquire() as conn, conn.transaction():
        # Validate both FKs at the service boundary — without this,
        # asyncpg surfaces FK violations as 500s instead of clean 4xxs.
        await queries.get_connection(conn, connection_id, account_id=account_id)
        session = await queries.get_session(conn, session_id, account_id=account_id)
        # Refuse archived sessions at bind time — the FK accepts archived
        # rows so without this check the ledger stamp succeeds, the
        # operator receives a 201 ``BoundChat`` with the archived id, and
        # subsequent inbounds DETACH at the resolver's tier-1 check
        # (post-#541). Bind-time is the only point where the operator
        # learns of the misconfiguration at the action that caused it
        # (mirrors the sibling ``attach_connection`` fix in #549).
        if session.archived_at is not None:
            raise ConflictError(
                f"session {session_id} is archived; cannot bind a chat to it",
                detail={"connection_id": connection_id, "session_id": session_id},
            )
        await queries.insert_chat_session(
            conn,
            connection_id=connection_id,
            chat_id=chat_id,
            session_id=session_id,
            account_id=account_id,
        )
        row = await queries.get_chat_session_row(
            conn, connection_id, chat_id, account_id=account_id
        )
    if row is None:
        raise NotFoundError(
            f"bound chat ({connection_id}, {chat_id}) not found after insert",
            detail={"connection_id": connection_id, "chat_id": chat_id},
        )
    row_chat_id, row_session_id, row_created_at = row
    return BoundChat(
        chat_id=row_chat_id,
        session_id=row_session_id,
        created_at=row_created_at,
    )


async def unbind_chat(
    pool: asyncpg.Pool[Any], connection_id: str, chat_id: str, *, account_id: str
) -> bool:
    """Delete a ``chat_sessions`` row.  Returns whether one was actually
    present (idempotent — repeat calls are no-ops)."""
    async with pool.acquire() as conn:
        return await queries.delete_chat_session(
            conn, connection_id, chat_id, account_id=account_id
        )


async def list_bound_chats(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str
) -> list[BoundChat]:
    """All ``chat_sessions`` rows for ``connection_id``, operator-bound
    and per-chat-spawned together.  An unknown ``connection_id`` 404s
    rather than returning ``[]`` so the operator surface is symmetric
    with the sibling endpoints."""
    async with pool.acquire() as conn:
        await queries.get_connection(conn, connection_id, account_id=account_id)
        rows = await queries.list_chat_sessions_for_connection(
            conn, connection_id, account_id=account_id
        )
    return [
        BoundChat(chat_id=chat_id, session_id=session_id, created_at=created_at)
        for chat_id, session_id, created_at in rows
    ]


async def list_recent_chats(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str, limit: int = 50
) -> list[RecentChat]:
    """Distinct chat_ids that have produced inbound on this connection's
    external identity, ordered most-recent first.  Used as the input to
    ``bind-chat`` when an operator needs to find a specific chat's
    ``chat_id`` without digging through event logs.
    """
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
        rows = await queries.list_recent_chat_ids(
            conn,
            connection.connector,
            connection.external_account_id,
            limit=limit,
            account_id=account_id,
        )
    return [
        RecentChat(chat_id=chat_id, last_seen_at=last_seen_at) for chat_id, last_seen_at in rows
    ]


async def archive_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, account_id: str
) -> Connection:
    """Archive a connection, refusing while it still has an active binding.

    Operators must ``detach`` / ``unconfigure`` first — this prevents an
    archive from silently dropping the inbound delivery target for a
    live single_session, or from stranding the template a per_chat
    binding spawns from.
    """
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)
        if connection.archived_at is not None:
            return await queries.archive_connection(conn, connection_id, account_id=account_id)
        binding = await queries.get_active_binding(conn, connection_id, account_id=account_id)
        if binding is not None:
            raise ConflictError(
                f"connection {connection_id} is in {binding.mode} mode; "
                f"detach or unconfigure before archiving",
                detail={"id": connection_id, "mode": binding.mode},
            )
        archived = await queries.archive_connection(conn, connection_id, account_id=account_id)
    async with pool.acquire() as conn:
        await queries.notify_connection_change(
            conn,
            connector=archived.connector,
            connection_id=archived.id,
            external_account_id=archived.external_account_id,
            event="removed",
            account_id=account_id,
        )
    return archived
