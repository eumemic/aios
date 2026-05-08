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

from aios.db import queries
from aios.errors import ConflictError, NotFoundError
from aios.models.agents import ToolSpec
from aios.models.connections import (
    BoundChat,
    Connection,
    ConnectionMode,
    RecentChat,
)


async def create_connection(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account: str,
    metadata: dict[str, Any],
    tools: list[ToolSpec] | None = None,
) -> Connection:
    tools_payload = [t.model_dump(exclude_none=True) for t in (tools or [])]
    async with pool.acquire() as conn:
        return await queries.insert_connection(
            conn,
            connector=connector,
            account=account,
            metadata=metadata,
            tools=tools_payload,
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


async def list_tools_for_session(pool: asyncpg.Pool[Any], session_id: str) -> list[dict[str, Any]]:
    """Custom tool specs from every active connection bound to ``session_id``.

    Used by :func:`aios.harness.step_context.compute_step_prelude` to
    surface connection-declared tools to the model alongside agent +
    MCP + connector-subprocess tools (#301).
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


async def attach_connection(
    pool: asyncpg.Pool[Any], connection_id: str, *, session_id: str
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.attach_connection(conn, connection_id, session_id=session_id)


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


async def validate_account_for_session(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    connector: str,
    account: str,
) -> bool:
    """Return True iff ``session_id`` is allowed to act on ``(connector, account)``.

    Authorization model: a session may invoke connector tools targeted
    at accounts that are either bound to it via ``connections.session_id``
    (single_session attach), spawned it via
    ``sessions.spawned_from_connection_id`` (per_chat origin), or
    operator-curated to it via a row in ``connection_chat_sessions``
    (#215).  Used by the outbound MCP dispatch — see
    :mod:`aios.harness.tool_dispatch`.
    """
    async with pool.acquire() as conn:
        return await queries.session_authorizes_connector_account(
            conn, session_id, connector, account
        )


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
        return await queries.archive_connection(conn, connection_id)
