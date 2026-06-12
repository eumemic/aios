"""Business logic for session templates.

Thin wrapper over :mod:`aios.db.queries` — the only business rule beyond
the schema is account-scoped ownership validation of every referenced
resource (agent, environment, vaults, memory stores) on create/update.
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
    async with pool.acquire() as conn, conn.transaction():
        # Validate the environment is account-owned before binding the template
        # to it. A bare FK would accept another tenant's env id and leak its
        # image / env-vars / networking into spawned sessions — mirrors
        # create_session / create_run (issue #755).
        await queries.get_environment(conn, environment_id, account_id=account_id)
        # Validate every referenced resource is account-owned before binding.
        # agent_id/environment_id have existence-only FKs (no ownership);
        # vault_ids/memory_store_ids are plain text[] with NO FK at all, so a
        # foreign id would silently bind. Mirror the #755 env guard (issue #851).
        await queries.get_agent(conn, agent_id, account_id=account_id)
        for vault_id in vault_ids:
            await queries.get_vault(conn, vault_id, account_id=account_id)
        for store_id in memory_store_ids:
            await queries.get_memory_store(conn, store_id, account_id=account_id)
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
    async with pool.acquire() as conn, conn.transaction():
        # Validate ownership only when the caller supplies a new value — omitting
        # a field preserves the current binding, so no check is needed there.
        # Without these guards the create-clean-then-update-dirty path bypasses
        # the create_session_template checks: agent_id has an existence-only FK,
        # and vault_ids/memory_store_ids are plain text[] with no FK at all, so a
        # foreign id would silently rebind. Mirrors create_session_template and
        # the #755 env guard (issue #851).
        if environment_id is not None:
            await queries.get_environment(conn, environment_id, account_id=account_id)
        if agent_id is not None:
            await queries.get_agent(conn, agent_id, account_id=account_id)
        if vault_ids is not None:
            for vault_id in vault_ids:
                await queries.get_vault(conn, vault_id, account_id=account_id)
        if memory_store_ids is not None:
            for store_id in memory_store_ids:
                await queries.get_memory_store(conn, store_id, account_id=account_id)
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
