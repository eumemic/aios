"""Business logic for session templates.

Thin wrapper over :mod:`aios.db.queries` — no business rules beyond
what the schema enforces.
"""

from __future__ import annotations

from types import EllipsisType
from typing import Any

import asyncpg

from aios.db import queries
from aios.models.session_templates import SessionTemplate


async def create_session_template(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    name: str,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
    vault_ids: list[str],
    memory_store_ids: list[str],
    metadata: dict[str, Any],
    archive_when_idle: bool = False,
) -> SessionTemplate:
    async with pool.acquire() as conn:
        return await queries.insert_session_template(
            conn,
            name=name,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            vault_ids=vault_ids,
            memory_store_ids=memory_store_ids,
            metadata=metadata,
            archive_when_idle=archive_when_idle,
            account_id=account_id,
        )


async def get_session_template(
    pool: asyncpg.Pool[Any], template_id: str, *, account_id: str
) -> SessionTemplate:
    async with pool.acquire() as conn:
        return await queries.get_session_template(conn, template_id, account_id=account_id)


async def list_session_templates(
    pool: asyncpg.Pool[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[SessionTemplate]:
    async with pool.acquire() as conn:
        return await queries.list_session_templates(
            conn, limit=limit, after=after, account_id=account_id
        )


async def update_session_template(
    pool: asyncpg.Pool[Any],
    template_id: str,
    *,
    account_id: str,
    name: str | None = None,
    agent_id: str | None = None,
    agent_version: int | None | EllipsisType = ...,
    environment_id: str | None = None,
    vault_ids: list[str] | None = None,
    memory_store_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    archive_when_idle: bool | None = None,
) -> SessionTemplate:
    async with pool.acquire() as conn:
        return await queries.update_session_template(
            conn,
            template_id,
            name=name,
            agent_id=agent_id,
            agent_version=agent_version,
            environment_id=environment_id,
            vault_ids=vault_ids,
            memory_store_ids=memory_store_ids,
            metadata=metadata,
            archive_when_idle=archive_when_idle,
            account_id=account_id,
        )


async def archive_session_template(
    pool: asyncpg.Pool[Any], template_id: str, *, account_id: str
) -> SessionTemplate:
    async with pool.acquire() as conn:
        return await queries.archive_session_template(conn, template_id, account_id=account_id)
