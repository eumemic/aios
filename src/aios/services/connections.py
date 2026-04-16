"""Business logic for connection resources.

Thin wrapper over :mod:`aios.db.queries` — connections themselves carry
no business rules at this phase. The actual routing logic lives in
:mod:`aios.services.channels`; the inbound-message endpoint composes
this service with that one.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
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
        return await queries.archive_connection(conn, connection_id)
