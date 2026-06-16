"""Integration test: the ``outbound_suppression`` session field round-trips
through the DB column added in migration 0106 (#710).

Exercises the create-with-mode, the default, the PUT flip (and that the flip
recycles the cached sandbox), and the idempotent re-PUT (same mode → no
recycle). Needs the real Postgres testcontainer so the migration's column +
CHECK constraint and the INSERT/UPDATE plumbing are actually run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def suppression_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_supp', NULL, TRUE, 'outbound-suppression-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_supp",
            name="supp-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_supp", name="supp-env"
        )
        yield pool, "acc_supp", agent.id, env.id
    finally:
        await pool.close()


async def test_default_is_off(
    suppression_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, agent_id, env_id = suppression_env
    session = await sessions_service.create_session(
        pool,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
    )
    assert session.outbound_suppression == "off"
    # Round-trips on a fresh read, too.
    fetched = await sessions_service.get_session_basic(pool, session.id, account_id=account_id)
    assert fetched.outbound_suppression == "off"


async def test_create_with_suppression_on(
    suppression_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, agent_id, env_id = suppression_env
    session = await sessions_service.create_session(
        pool,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
        outbound_suppression="on",
    )
    assert session.outbound_suppression == "on"


async def test_flip_via_update_recycles_sandbox(
    suppression_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, agent_id, env_id = suppression_env
    session = await sessions_service.create_session(
        pool,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
    )
    with patch.object(sessions_service, "_evict_sandbox_for_resource_change") as evict:
        updated = await sessions_service.update_session(
            pool, session.id, account_id=account_id, outbound_suppression="on"
        )
    assert updated.outbound_suppression == "on"
    evict.assert_called_once_with(session.id)

    # Idempotent re-PUT (same mode) must not recycle.
    with patch.object(sessions_service, "_evict_sandbox_for_resource_change") as evict2:
        again = await sessions_service.update_session(
            pool, session.id, account_id=account_id, outbound_suppression="on"
        )
    assert again.outbound_suppression == "on"
    evict2.assert_not_called()


async def test_check_constraint_rejects_bad_value(
    suppression_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, agent_id, env_id = suppression_env
    session = await sessions_service.create_session(
        pool,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
    )
    with pytest.raises(asyncpg.PostgresError):
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET outbound_suppression = 'bogus' WHERE id = $1",
                session.id,
            )
