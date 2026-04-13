"""Business logic for agents.

Agents are versioned: every update creates a new immutable version.
The ``agents`` table holds the latest config; ``agent_versions`` stores
the full history.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.agents import Agent, AgentVersion, ToolSpec


async def create_agent(
    pool: asyncpg.Pool[Any],
    *,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
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


async def update_agent(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    *,
    expected_version: int,
    name: str | None = None,
    model: str | None = None,
    system: str | None = None,
    tools: list[ToolSpec] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
) -> Agent:
    async with pool.acquire() as conn:
        return await queries.update_agent(
            conn,
            agent_id,
            expected_version=expected_version,
            name=name,
            model=model,
            system=system,
            tools=tools,
            description=description,
            metadata=metadata,
            window_min=window_min,
            window_max=window_max,
        )


async def get_agent_version(pool: asyncpg.Pool[Any], agent_id: str, version: int) -> AgentVersion:
    async with pool.acquire() as conn:
        return await queries.get_agent_version(conn, agent_id, version)


async def list_agent_versions(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    *,
    limit: int = 50,
    after: int | None = None,
) -> list[AgentVersion]:
    async with pool.acquire() as conn:
        return await queries.list_agent_versions(conn, agent_id, limit=limit, after=after)
