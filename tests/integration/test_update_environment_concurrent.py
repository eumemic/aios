"""Concurrent partial writers to the same environment row must not lose
updates. Pre-fix, ``update_environment`` read ``current`` outside any
transaction, merged in Python, then wrote BOTH columns back — so a
config-only writer would clobber a concurrent name-only writer's
rename (and vice versa)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.models.environments import (
    Environment,
    EnvironmentConfig,
    LimitedNetworking,
    UnrestrictedNetworking,
)
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_with_environment(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, Environment]]:
    """Yield ``(pool, account_id, env)`` for a freshly-seeded environment."""
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        env = await environments_service.create_environment(
            pool,
            account_id="acc_test",
            name="initial",
            config=EnvironmentConfig(
                packages={"pip": ["pandas"]},
                networking=UnrestrictedNetworking(),
            ),
        )
        yield pool, "acc_test", env
    finally:
        await pool.close()


async def test_concurrent_partial_updates_preserve_both_changes(
    pool_with_environment: tuple[asyncpg.Pool[Any], str, Environment],
) -> None:
    pool, account_id, env = pool_with_environment

    new_config = EnvironmentConfig(
        packages={"pip": ["pandas"]},
        networking=LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
    )

    # Pool min_size=2 so both coroutines can hold connections concurrently —
    # the race is real, not artificially serialized by pool exhaustion.
    name_result, config_result = await asyncio.gather(
        environments_service.update_environment(
            pool, env.id, account_id=account_id, name="renamed"
        ),
        environments_service.update_environment(
            pool, env.id, account_id=account_id, config=new_config
        ),
    )

    final = await environments_service.get_environment(pool, env.id, account_id=account_id)
    assert final.name == "renamed"
    assert final.config == new_config
    assert name_result.name == "renamed"
    assert config_result.config == new_config


async def test_sequential_partial_updates_preserve_omitted_fields(
    pool_with_environment: tuple[asyncpg.Pool[Any], str, Environment],
) -> None:
    pool, account_id, env = pool_with_environment

    renamed = await environments_service.update_environment(
        pool, env.id, account_id=account_id, name="renamed"
    )
    assert renamed.name == "renamed"
    assert renamed.config == env.config

    new_config = EnvironmentConfig(packages={"pip": ["numpy"]})
    reconfigured = await environments_service.update_environment(
        pool, env.id, account_id=account_id, config=new_config
    )
    assert reconfigured.name == "renamed"
    assert reconfigured.config == new_config
