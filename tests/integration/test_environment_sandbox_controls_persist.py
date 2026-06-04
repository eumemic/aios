"""The per-environment sandbox controls (issues #724, #725) round-trip
through the ``environments.config`` JSONB column without any schema
migration.

``image``, ``disk_bytes``, and ``bash_timeout_seconds`` are stored
inside the existing schemaless JSONB ``config`` column (added by
migration 0004), so adding them to :class:`EnvironmentConfig` is purely
additive at the DB layer. These tests prove an insert carrying the new
fields reads back intact, and that an environment with none of them set
reads back with all three ``None`` (current behavior preserved).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.environments import EnvironmentConfig
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_with_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        yield pool, "acc_test"
    finally:
        await pool.close()


async def test_new_fields_round_trip_through_jsonb(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """An environment created with image + disk + bash-timeout overrides
    reads back with all three intact — no migration required."""
    pool, account_id = pool_with_account

    config = EnvironmentConfig(
        image="ghcr.io/eumemic/aios-dev-env:pinned",
        disk_bytes=8 * 1024 * 1024 * 1024,
        bash_timeout_seconds=600,
        packages={"pip": ["pytest"]},
    )
    created = await environments_service.create_environment(
        pool, account_id=account_id, name="dev-env", config=config
    )
    assert created.config.image == "ghcr.io/eumemic/aios-dev-env:pinned"
    assert created.config.disk_bytes == 8 * 1024 * 1024 * 1024
    assert created.config.bash_timeout_seconds == 600

    # Read back through a fresh query to confirm the JSONB persisted, not
    # just the in-memory echo returned by the insert.
    fetched = await environments_service.get_environment(pool, created.id, account_id=account_id)
    assert fetched.config == config


async def test_unset_fields_persist_as_none(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """An environment with none of the new fields set reads back with all
    three ``None`` — current behavior (global defaults) is preserved."""
    pool, account_id = pool_with_account

    created = await environments_service.create_environment(
        pool,
        account_id=account_id,
        name="plain-env",
        config=EnvironmentConfig(packages={"pip": ["pandas"]}),
    )
    fetched = await environments_service.get_environment(pool, created.id, account_id=account_id)
    assert fetched.config.image is None
    assert fetched.config.disk_bytes is None
    assert fetched.config.bash_timeout_seconds is None


async def test_get_environment_config_for_session_carries_new_fields(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """The session-scoped read used by the sandbox spec builder surfaces
    the new fields, so ``build_spec_from_session`` can resolve the
    per-env image / disk / bash-timeout overrides (issues #724, #725)."""
    pool, account_id = pool_with_account

    config = EnvironmentConfig(
        image="ghcr.io/eumemic/aios-dev-env:pinned",
        disk_bytes=4 * 1024 * 1024 * 1024,
        bash_timeout_seconds=300,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name="bound-env", config=config
    )
    # A session needs an agent FK; seed one via the canonical service path
    # (same shape as test_session_cross_tenant_isolation.py).
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name="dev-agent",
        model="openrouter/some-model",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
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
        loaded = await queries.get_environment_config_for_session(
            conn, session.id, account_id=account_id
        )

    assert loaded is not None
    assert loaded.image == "ghcr.io/eumemic/aios-dev-env:pinned"
    assert loaded.disk_bytes == 4 * 1024 * 1024 * 1024
    assert loaded.bash_timeout_seconds == 300
