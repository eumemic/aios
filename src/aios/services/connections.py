"""Business logic for connection resources.

Thin wrapper over :mod:`aios.db.queries`. The only business rule lives
in :func:`archive_connection`, which refuses to archive a connection
while channel bindings under its ``(connector, account)`` prefix are
still active — archiving would silently drop the connection-provided
MCP tools from any live session bound to those channels.
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
    mcp_url: str,
    vault_id: str,
    metadata: dict[str, Any],
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.insert_connection(
            conn,
            connector=connector,
            account=account,
            mcp_url=mcp_url,
            vault_id=vault_id,
            metadata=metadata,
        )


async def get_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        return await queries.get_connection(conn, connection_id)


async def list_connections(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Connection]:
    async with pool.acquire() as conn:
        return await queries.list_connections(conn, limit=limit, after=after)


async def update_connection(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    mcp_url: str | None = None,
    vault_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Connection:
    async with pool.acquire() as conn:
        return await queries.update_connection(
            conn,
            connection_id,
            mcp_url=mcp_url,
            vault_id=vault_id,
            metadata=metadata,
        )


async def archive_connection(pool: asyncpg.Pool[Any], connection_id: str) -> Connection:
    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
        if connection.archived_at is not None:
            # Let the query raise its canonical "already archived" error.
            return await queries.archive_connection(conn, connection_id)
        active = await queries.count_active_bindings_for_connection(
            conn, connector=connection.connector, account=connection.account
        )
        if active > 0:
            raise ConflictError(
                f"connection {connection_id} has {active} active channel binding"
                f"{'s' if active != 1 else ''} under {connection.connector}/"
                f"{connection.account}; archive the bindings first to avoid "
                f"silently dropping MCP tools from live sessions",
                detail={
                    "id": connection_id,
                    "active_bindings": active,
                    "connector": connection.connector,
                    "account": connection.account,
                },
            )
        return await queries.archive_connection(conn, connection_id)
