"""Business logic for agents.

Agents are versioned: every update creates a new immutable version.
The ``agents`` table holds the latest config; ``agent_versions`` stores
the full history.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.agents import Agent, AgentVersion, McpServerSpec, ToolSpec
from aios.models.skills import AgentSkillRef
from aios.services import skills as skills_service


async def create_agent(
    pool: asyncpg.Pool[Any],
    *,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
    skills: list[AgentSkillRef] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    description: str | None,
    metadata: dict[str, Any],
    litellm_extra: dict[str, Any] | None = None,
    window_min: int,
    window_max: int,
) -> Agent:
    if window_min >= window_max:
        from aios.errors import ValidationError

        raise ValidationError(
            "window_min must be strictly less than window_max",
            detail={"window_min": window_min, "window_max": window_max},
        )
    skill_refs = skills or []
    resolved = await skills_service.resolve_skill_refs(pool, skill_refs)
    snapshot_json = skills_service.serialize_skills_for_snapshot(skill_refs, resolved)
    async with pool.acquire() as conn:
        return await queries.insert_agent(
            conn,
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills_json=snapshot_json,
            mcp_servers=mcp_servers or [],
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra or {},
            window_min=window_min,
            window_max=window_max,
        )


async def get_agent(pool: asyncpg.Pool[Any], agent_id: str) -> Agent:
    async with pool.acquire() as conn:
        return await queries.get_agent(conn, agent_id)


async def list_agents(
    pool: asyncpg.Pool[Any],
    *,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Agent]:
    async with pool.acquire() as conn:
        return await queries.list_agents(conn, limit=limit, after=after, name=name)


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
    skills: list[AgentSkillRef] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    litellm_extra: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
) -> Agent:
    skills_json_str: str | None = None
    if skills is not None:
        resolved = await skills_service.resolve_skill_refs(pool, skills)
        skills_json_str = skills_service.serialize_skills_for_snapshot(skills, resolved)
    async with pool.acquire() as conn:
        return await queries.update_agent(
            conn,
            agent_id,
            expected_version=expected_version,
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills_json=skills_json_str,
            mcp_servers=mcp_servers,
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra,
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
