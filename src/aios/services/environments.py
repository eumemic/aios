"""Business logic for environments. v1 environments are name-only."""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.environments import Environment


async def create_environment(pool: asyncpg.Pool[Any], *, name: str) -> Environment:
    async with pool.acquire() as conn:
        return await queries.insert_environment(conn, name=name)


async def get_environment(pool: asyncpg.Pool[Any], env_id: str) -> Environment:
    async with pool.acquire() as conn:
        return await queries.get_environment(conn, env_id)


async def list_environments(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Environment]:
    async with pool.acquire() as conn:
        return await queries.list_environments(conn, limit=limit, after=after)


async def archive_environment(pool: asyncpg.Pool[Any], env_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_environment(conn, env_id)
