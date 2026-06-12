"""Integration tests: ``create_session_template`` must validate that every
referenced resource (agent, vault, memory store) is account-owned.

``environment_id`` already gets a #755 ownership guard, but ``agent_id`` was
FK-only and ``vault_ids`` / ``memory_store_ids`` are plain ``text[]`` columns
with NO FK at all — a foreign id would silently bind. These tests pin the
ownership guards added in issue #851.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.services import memory_stores as memory_stores_service
from aios.services import session_templates as session_templates_service
from aios.services import vaults as vaults_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_three_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool seeded with one root + two child tenants (``acc_a``, ``acc_b``)."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_root', NULL, TRUE, 'root'), "
                "('acc_a', 'acc_root', FALSE, 'a'), "
                "('acc_b', 'acc_root', FALSE, 'b')"
            )
        yield pool
    finally:
        await pool.close()


async def test_create_template_rejects_foreign_agent(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """acc_b's agent + acc_a's own env → NotFoundError (agent ownership guard)."""
    pool = pool_three_accounts
    _agent_a, env_a, _s = await seed_agent_env_session(pool, account_id="acc_a", prefix="st")
    agent_b, _env_b, _sb = await seed_agent_env_session(pool, account_id="acc_b", prefix="stb")

    with pytest.raises(NotFoundError):
        await session_templates_service.create_session_template(
            pool,
            account_id="acc_a",
            name="tmpl-foreign-agent",
            agent_id=agent_b.id,
            environment_id=env_a.id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )


async def test_create_template_rejects_foreign_vault(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """acc_a agent+env, vault_ids=[acc_b vault] → NotFoundError (vault ownership guard)."""
    pool = pool_three_accounts
    agent_a, env_a, _s = await seed_agent_env_session(pool, account_id="acc_a", prefix="st")
    vault_b = await vaults_service.create_vault(
        pool, account_id="acc_b", display_name="b-vault", metadata={}
    )

    with pytest.raises(NotFoundError):
        await session_templates_service.create_session_template(
            pool,
            account_id="acc_a",
            name="tmpl-foreign-vault",
            agent_id=agent_a.id,
            environment_id=env_a.id,
            agent_version=None,
            vault_ids=[vault_b.id],
            memory_store_ids=[],
            metadata={},
        )


async def test_create_template_rejects_foreign_memory_store(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """acc_a agent+env, memory_store_ids=[acc_b store] → NotFoundError (store ownership guard)."""
    pool = pool_three_accounts
    agent_a, env_a, _s = await seed_agent_env_session(pool, account_id="acc_a", prefix="st")
    store_b = await memory_stores_service.create_store(
        pool, account_id="acc_b", name="b-store", description="", metadata={}
    )

    with pytest.raises(NotFoundError):
        await session_templates_service.create_session_template(
            pool,
            account_id="acc_a",
            name="tmpl-foreign-store",
            agent_id=agent_a.id,
            environment_id=env_a.id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[store_b.id],
            metadata={},
        )


async def test_create_template_with_own_resources_succeeds(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Control: all acc_a-owned ids → a template that round-trips its bindings."""
    pool = pool_three_accounts
    agent_a, env_a, _s = await seed_agent_env_session(pool, account_id="acc_a", prefix="st")
    own_vault = await vaults_service.create_vault(
        pool, account_id="acc_a", display_name="a-vault", metadata={}
    )
    own_store = await memory_stores_service.create_store(
        pool, account_id="acc_a", name="a-store", description="", metadata={}
    )

    tmpl = await session_templates_service.create_session_template(
        pool,
        account_id="acc_a",
        name="tmpl-own",
        agent_id=agent_a.id,
        environment_id=env_a.id,
        agent_version=None,
        vault_ids=[own_vault.id],
        memory_store_ids=[own_store.id],
        metadata={},
    )
    assert tmpl.vault_ids == [own_vault.id]
    assert tmpl.memory_store_ids == [own_store.id]


async def _own_template(pool: asyncpg.Pool[Any]) -> str:
    """Create a clean acc_a-owned template and return its id (the clean half
    of the create-clean-then-update-dirty bypass these tests pin shut)."""
    agent_a, env_a, _s = await seed_agent_env_session(pool, account_id="acc_a", prefix="stu")
    tmpl = await session_templates_service.create_session_template(
        pool,
        account_id="acc_a",
        name="tmpl-update-base",
        agent_id=agent_a.id,
        environment_id=env_a.id,
        agent_version=None,
        vault_ids=[],
        memory_store_ids=[],
        metadata={},
    )
    return tmpl.id


async def test_update_template_rejects_foreign_agent(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Updating a clean acc_a template to bind acc_b's agent → NotFoundError."""
    pool = pool_three_accounts
    template_id = await _own_template(pool)
    agent_b, _env_b, _sb = await seed_agent_env_session(pool, account_id="acc_b", prefix="stub")

    with pytest.raises(NotFoundError):
        await session_templates_service.update_session_template(
            pool,
            template_id,
            account_id="acc_a",
            agent_id=agent_b.id,
        )


async def test_update_template_rejects_foreign_vault(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Updating a clean acc_a template to bind acc_b's vault → NotFoundError."""
    pool = pool_three_accounts
    template_id = await _own_template(pool)
    vault_b = await vaults_service.create_vault(
        pool, account_id="acc_b", display_name="b-vault", metadata={}
    )

    with pytest.raises(NotFoundError):
        await session_templates_service.update_session_template(
            pool,
            template_id,
            account_id="acc_a",
            vault_ids=[vault_b.id],
        )


async def test_update_template_rejects_foreign_memory_store(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Updating a clean acc_a template to bind acc_b's store → NotFoundError."""
    pool = pool_three_accounts
    template_id = await _own_template(pool)
    store_b = await memory_stores_service.create_store(
        pool, account_id="acc_b", name="b-store", description="", metadata={}
    )

    with pytest.raises(NotFoundError):
        await session_templates_service.update_session_template(
            pool,
            template_id,
            account_id="acc_a",
            memory_store_ids=[store_b.id],
        )
