"""Business logic for agents. Phase 1: mutable single rows.

Phase 4 will replace this module with versioning logic that allocates a new
agent_versions row on every update.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.agents import Agent, ToolSpec


async def create_agent(
    pool: asyncpg.Pool[Any],
    *,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
    credential_id: str | None,
    description: str | None,
    metadata: dict[str, Any],
    window_min: int,
    window_max: int,
) -> Agent:
    if window_min >= window_max:
        from aios.errors import ValidationError

        raise ValidationError(
            "window_min must be strictly less than window_max",
            detail={"window_min": window_min, "window_max": window_max},
        )
    async with pool.acquire() as conn:
        return await queries.insert_agent(
            conn,
            name=name,
            model=model,
            system=system,
            tools=tools,
            credential_id=credential_id,
            description=description,
            metadata=metadata,
            window_min=window_min,
            window_max=window_max,
        )


async def get_agent(pool: asyncpg.Pool[Any], agent_id: str) -> Agent:
    async with pool.acquire() as conn:
        return await queries.get_agent(conn, agent_id)


async def list_agents(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Agent]:
    async with pool.acquire() as conn:
        return await queries.list_agents(conn, limit=limit, after=after)


async def archive_agent(pool: asyncpg.Pool[Any], agent_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_agent(conn, agent_id)
