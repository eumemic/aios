"""Business logic for environments."""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.environments import Environment, EnvironmentConfig


async def create_environment(
    pool: asyncpg.Pool[Any],
    *,
    name: str,
    config: EnvironmentConfig | None = None,
) -> Environment:
    async with pool.acquire() as conn:
        return await queries.insert_environment(conn, name=name, config=config)


async def get_environment(pool: asyncpg.Pool[Any], env_id: str) -> Environment:
    async with pool.acquire() as conn:
        return await queries.get_environment(conn, env_id)


async def list_environments(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Environment]:
    async with pool.acquire() as conn:
        return await queries.list_environments(conn, limit=limit, after=after)


async def update_environment(
    pool: asyncpg.Pool[Any],
    env_id: str,
    *,
    name: str | None = None,
    config: EnvironmentConfig | None = None,
) -> Environment:
    async with pool.acquire() as conn:
        return await queries.update_environment(conn, env_id, name=name, config=config)


async def archive_environment(pool: asyncpg.Pool[Any], env_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_environment(conn, env_id)
