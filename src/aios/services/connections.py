"""Business logic for connection resources.

Thin wrapper over :mod:`aios.db.queries`.  The single business rule
lives in :func:`archive_connection`: refuse to archive while the
connection is in single_session or per_chat mode.  Operators must
``detach`` (or ``unconfigure``) first — silently dropping the routing
binding on archive would orphan the spawned-from-connection_id pointers
on per_chat sessions and would interrupt outbound delivery for
single_session.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.crypto.vault import CryptoBox, EncryptedBlob
from aios.db import queries
from aios.errors import ConflictError, NotFoundError
from aios.models.agents import ToolSpec
from aios.models.connections import (
    BoundChat,
    Connection,
    ConnectionMode,
    RecentChat,
)


def _encrypt_secrets(secrets: dict[str, str] | None, crypto_box: CryptoBox) -> EncryptedBlob | None:
    """Encrypt a secrets dict, or ``None`` if the dict is missing or empty.

    Empty / missing → no blob → schema columns NULL → ``secrets_set: False``.
    The operator surface treats ``None`` and ``{}`` identically (both
    "clear secrets") so create + PUT produce the same row state.
    """
    if not secrets:
        return None
    return crypto_box.encrypt_dict(secrets)


async def create_connection(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account: str,
    metadata: dict[str, Any],
    tools: list[ToolSpec] | None = None,
    secrets: dict[str, str] | None = None,
    crypto_box: CryptoBox,
) -> Connection:
    tools_payload = [t.model_dump(exclude_none=True) for t in (tools or [])]
    async with pool.acquire() as conn:
        return await queries.insert_connection(
            conn,
            connector=connector,
            account=account,
            metadata=metadata,
            tools=tools_payload,
            secrets_blob=_encrypt_secrets(secrets, crypto_box),
        )


async def set_connection_tools(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    tools: list[ToolSpec],
) -> Connection:
    """Replace a connection's tools.  Caller validates ToolSpec types via
    the request model (see :class:`ConnectionSetTools`).
    """
    payload = [t.model_dump(exclude_none=True) for t in tools]
    async with pool.acquire() as conn:
        return await queries.set_connection_tools(conn, connection_id, tools=payload)


async def set_connection_secrets(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
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
            secrets_blob=_encrypt_secrets(secrets, crypto_box),
        )


async def get_connection_secrets(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
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
        blob = await queries.get_connection_secret_blob(conn, connection_id)
    if blob is None:
        return {}
    decoded = crypto_box.decrypt_dict(blob)
    return {str(k): str(v) for k, v in decoded.items()}


async def list_tools_for_session(pool: asyncpg.Pool[Any], session_id: str) -> list[dict[str, Any]]:
    """Custom tool specs from every active connection bound to ``session_id``.

    Reached from the harness's per-step prelude via the ``ToolProvider``
    Protocol (#328): :class:`aios_connectors.providers.SubsystemToolProvider`
    delegates here, and ``compute_step_prelude`` calls the provider
    rather than this function directly. PR 7 swaps the SQL onto the
    new ``bindings`` ⨝ ``connectors.tools_schema`` tables.
    """
    async with pool.acquire() as conn:
        return await queries.list_connection_tools_for_session(conn, session_id)


async def get_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        return await queries.get_connection(conn, connection_id)


async def list_connections(
    pool: asyncpg.Pool[Any],
    *,
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
        )


async def _notify_connection_change(
    pool: asyncpg.Pool[Any], connection: Connection, *, event: str
) -> None:
    """Thin wrapper acquiring a fresh pool connection for the NOTIFY emit.

    Runs OUTSIDE the caller's transaction (the txn that just committed
    the attach/archive) so a subscriber never sees a payload for an
    uncommitted row.
    """
    async with pool.acquire() as conn:
        await queries.notify_connection_change(
            conn,
            connector=connection.connector,
            connection_id=connection.id,
            account=connection.account,
            event=event,
        )


async def attach_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, session_id: str
) -> Connection:
    async with pool.acquire() as conn:
        connection = await queries.attach_connection(conn, connection_id, session_id=session_id)
    await _notify_connection_change(pool, connection, event="added")
    return connection


async def detach_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        return await queries.detach_connection(conn, connection_id)


async def configure_per_chat(
    pool: asyncpg.Pool[Any], connection_id: str, *, session_template_id: str
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.configure_per_chat_connection(
            conn, connection_id, session_template_id=session_template_id
        )


async def unconfigure_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        return await queries.unconfigure_connection(conn, connection_id)


async def bind_chat_to_session(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    chat_id: str,
    session_id: str,
) -> BoundChat:
    """Insert an operator-curated ``connection_chat_sessions`` row.

    On conflict (a row already exists for this ``(connection_id, chat_id)``,
    either operator-bound earlier or supervisor-spawned via per_chat) the
    existing row is preserved — the call is idempotent.  The returned
    ``BoundChat`` reflects whichever ``session_id`` is now stored, which
    may differ from the requested one if the conflict path triggered.
    """
    async with pool.acquire() as conn, conn.transaction():
        # Validate both FKs at the service boundary — without this,
        # asyncpg surfaces FK violations as 500s instead of clean 4xxs.
        await queries.get_connection(conn, connection_id)
        await queries.get_session(conn, session_id)
        await queries.insert_chat_session(
            conn,
            connection_id=connection_id,
            chat_id=chat_id,
            session_id=session_id,
        )
        row = await queries.get_chat_session_row(conn, connection_id, chat_id)
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


async def unbind_chat(pool: asyncpg.Pool[Any], connection_id: str, chat_id: str) -> bool:
    """Delete a ``connection_chat_sessions`` row.  Returns whether one
    was actually present (idempotent — repeat calls are no-ops)."""
    async with pool.acquire() as conn:
        return await queries.delete_chat_session(conn, connection_id, chat_id)


async def list_bound_chats(pool: asyncpg.Pool[Any], connection_id: str) -> list[BoundChat]:
    """All ``connection_chat_sessions`` rows for ``connection_id``,
    operator-bound and supervisor-spawned together.  An unknown
    ``connection_id`` 404s rather than returning ``[]`` so the operator
    surface is symmetric with the sibling endpoints."""
    async with pool.acquire() as conn:
        await queries.get_connection(conn, connection_id)
        rows = await queries.list_chat_sessions_for_connection(conn, connection_id)
    return [
        BoundChat(chat_id=chat_id, session_id=session_id, created_at=created_at)
        for chat_id, session_id, created_at in rows
    ]


async def list_recent_chats(
    pool: asyncpg.Pool[Any], connection_id: str, *, limit: int = 50
) -> list[RecentChat]:
    """Distinct chat_ids that have produced inbound on this connection's
    account, ordered most-recent first.  Used as the input to
    ``bind-chat`` when an operator needs to find a specific chat's
    ``chat_id`` without digging through event logs.
    """
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
        rows = await queries.list_recent_chat_ids(
            conn, connection.connector, connection.account, limit=limit
        )
    return [
        RecentChat(chat_id=chat_id, last_seen_at=last_seen_at) for chat_id, last_seen_at in rows
    ]


async def archive_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    """Archive a connection, refusing while it's still bound to a session
    or template.

    Operators must ``detach`` / ``unconfigure`` first — this prevents an
    archive from silently dropping the inbound delivery target for a
    live single_session, or orphaning the ``spawned_from_connection_id``
    pointer on per_chat-spawned sessions.
    """
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
        if connection.archived_at is not None:
            return await queries.archive_connection(conn, connection_id)
        if connection.session_id is not None or connection.session_template_id is not None:
            mode = "single_session" if connection.session_id is not None else "per_chat"
            raise ConflictError(
                f"connection {connection_id} is in {mode} mode; "
                f"detach or unconfigure before archiving",
                detail={"id": connection_id, "mode": mode},
            )
        archived = await queries.archive_connection(conn, connection_id)
    await _notify_connection_change(pool, archived, event="removed")
    return archived
