"""Shared fixtures for ``tests/integration/`` (DB-backed, testcontainer-Postgres).

Anything that seeds reusable account / tenant state belongs here so
individual test modules don't re-roll the same scaffolding.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import register_jsonb_codec
from aios.models.agents import Agent, ToolSpec
from aios.models.environments import Environment
from aios.models.sessions import Session
from aios.services import agents as agents_service
from aios.services import environments as environments_service


async def seed_agent_env_session(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    prefix: str,
    tools: list[ToolSpec] | None = None,
) -> tuple[Agent, Environment, Session]:
    """Seed a default ``(agent, env, session)`` trio scoped to ``account_id``.

    Used by integration tests that need a session-shaped scaffold but
    don't care about the specifics of the agent / environment / session
    rows. The agent name is ``{prefix}-agent``; the env name is
    ``{prefix}-env``. Other agent settings (``model="openrouter/test"``,
    ``window_min=50_000``, ``window_max=150_000``, empty system /
    description / metadata) match the long-standing conventions across
    the existing integration tests.
    """
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"{prefix}-agent",
        model="openrouter/test",
        system="",
        tools=tools or [],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"{prefix}-env"
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
    return agent, env, session


@pytest.fixture
async def conn_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg conn with one root + two child tenants (``acc_a``, ``acc_b``).

    The partial unique index ``accounts_one_active_root`` permits only
    a single non-archived ``parent_account_id IS NULL`` row at a time,
    so the root + two children layout is the minimum that supports
    cross-tenant tests.
    """
    conn = await asyncpg.connect(migrated_db_url)
    # Mirror the production pool: query functions read jsonb as native Python.
    await register_jsonb_codec(conn)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,      TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                   ('acc_b',    'acc_root', FALSE, 'tenant-b')
            """
        )
        yield conn
    finally:
        await conn.close()


@pytest.fixture
async def live_conn(request: pytest.FixtureRequest) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg connection to a migrated DB, TRUNCATEd before the test.

    Resolves its target in two ways so the same test body runs both in CI and
    on a workstation with no Docker:

    * ``AIOS_TEST_LIVE_DB_URL`` — an externally provided, already-migrated
      Postgres (local dev / sandboxes where testcontainers can't run);
    * otherwise the standard session-scoped ``migrated_db_url`` testcontainer
      fixture, so these tests execute on the normal integration shard.
    """
    import os

    url = os.environ.get("AIOS_TEST_LIVE_DB_URL") or request.getfixturevalue("migrated_db_url")
    conn = await asyncpg.connect(url)
    await register_jsonb_codec(conn)
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
    )
    if rows:
        await conn.execute(
            "TRUNCATE " + ", ".join(r["tablename"] for r in rows) + " RESTART IDENTITY CASCADE"
        )
    try:
        yield conn
    finally:
        await conn.close()
