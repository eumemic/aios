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
from aios.errors import ConflictError
from aios.models.connections import Connection


async def create_connection(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account: str,
    metadata: dict[str, Any],
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.insert_connection(
            conn,
            connector=connector,
            account=account,
            metadata=metadata,
        )


async def get_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        return await queries.get_connection(conn, connection_id)


async def list_connections(
    pool: asyncpg.Pool[Any],
    *,
    connector: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Connection]:
    async with pool.acquire() as conn:
        return await queries.list_connections(
            conn,
            connector=connector,
            session_id=session_id,
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
