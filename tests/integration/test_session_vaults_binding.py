"""Integration tests: the session-vault binding write path must validate
vault OWNERSHIP, and ``create_session`` must validate agent ownership.

``set_session_vaults`` stamps the caller's ``account_id`` onto each bind row
but the ``vault_id`` FK alone proves only existence, not ownership — a
foreign-but-existing vault would bind, a secret-injection vector. ``create_session``
likewise bound ``agent_id`` by FK only. These tests pin the ownership guards
(issue #851), the run analog of ``test_wf_run_vaults.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.services import sessions as sessions_service
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


async def _session_vault_count(pool: asyncpg.Pool[Any], session_id: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM session_vaults WHERE session_id = $1", session_id
            )
        )


async def test_set_session_vaults_rejects_foreign_vault(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Binding acc_b's (existing) vault to acc_a's session raises NotFoundError —
    the FK alone would accept it; the ownership guard rejects it, no row leaks."""
    pool = pool_three_accounts
    _agent, _env, session = await seed_agent_env_session(pool, account_id="acc_a", prefix="sv")
    vault_b = await vaults_service.create_vault(
        pool, account_id="acc_b", display_name="b-vault", metadata={}
    )

    async with pool.acquire() as conn:
        with pytest.raises(NotFoundError):
            await queries.set_session_vaults(conn, session.id, [vault_b.id], account_id="acc_a")
    assert await _session_vault_count(pool, session.id) == 0


async def test_set_session_vaults_rejects_nonexistent_vault(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """A plainly nonexistent vault id is likewise NotFound, with no bind row."""
    pool = pool_three_accounts
    _agent, _env, session = await seed_agent_env_session(pool, account_id="acc_a", prefix="sv")

    async with pool.acquire() as conn:
        with pytest.raises(NotFoundError):
            await queries.set_session_vaults(
                conn, session.id, ["vlt_does_not_exist"], account_id="acc_a"
            )
    assert await _session_vault_count(pool, session.id) == 0


async def test_set_session_vaults_binds_own_vault(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """Control: binding acc_a's own vault succeeds and reads back."""
    pool = pool_three_accounts
    _agent, _env, session = await seed_agent_env_session(pool, account_id="acc_a", prefix="sv")
    own = await vaults_service.create_vault(
        pool, account_id="acc_a", display_name="a-vault", metadata={}
    )

    async with pool.acquire() as conn:
        await queries.set_session_vaults(conn, session.id, [own.id], account_id="acc_a")
        bound = await queries.get_session_vault_ids(conn, session.id, account_id="acc_a")
    assert bound == [own.id]


async def test_create_session_rejects_foreign_agent(
    pool_three_accounts: asyncpg.Pool[Any],
) -> None:
    """``create_session`` for acc_a binding acc_b's agent (with acc_a's own env)
    raises NotFoundError — the agent FK proves existence, not ownership — and no
    acc_a session row is created."""
    pool = pool_three_accounts
    _agent_a, env_a, _session = await seed_agent_env_session(pool, account_id="acc_a", prefix="sv")
    agent_b, _env_b, _session_b = await seed_agent_env_session(
        pool, account_id="acc_b", prefix="svb"
    )

    async def _acc_a_session_count() -> int:
        async with pool.acquire() as conn:
            return int(
                await conn.fetchval("SELECT count(*) FROM sessions WHERE account_id = 'acc_a'")
            )

    before = await _acc_a_session_count()
    with pytest.raises(NotFoundError):
        await sessions_service.create_session(
            pool,
            account_id="acc_a",
            agent_id=agent_b.id,
            environment_id=env_a.id,
            title=None,
            metadata={},
        )
    assert await _acc_a_session_count() == before
