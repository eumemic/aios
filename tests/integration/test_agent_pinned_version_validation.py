"""Integration tests: a pinned ``agent_version`` is validated at write time.

Pre-fix, create/update of a session or session_template wrote a
caller-supplied ``agent_version`` with no existence check — the bare
``agent_id`` FK constrains the agent, not the version integer. A bad pin
(e.g. a version greater than the agent's current one) was accepted at
write time and returned 200/201, then the first ``wake_session`` step's
``load_for_session`` called ``get_agent_version``, raised
``NotFoundError``, and the session burned its retry budget into the
terminal ``errored`` state. ``agent_versions`` is append-only, so the pin
never materializes — a permanent, unrecoverable brick from a write that
looked healthy.

The fix validates the *resolved* pin in all four writers
(``create_session``, ``update_session``, ``create_session_template``,
``update_session_template``) via the shared
``validate_pinned_agent_version``, turning a silent mid-run brick into a
clean ``NotFoundError`` at the write the operator initiated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.services import session_templates as templates_service
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

# A version that cannot exist for a freshly-seeded agent (which is at version 1).
_BAD_PIN = 999


@pytest.fixture
async def pool_and_account(
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


class TestCreateSessionPinValidation:
    async def test_nonexistent_pin_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(pool, account_id=account_id, prefix="pin-cs")
        with pytest.raises(NotFoundError):
            await sessions_service.create_session(
                pool,
                account_id=account_id,
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=_BAD_PIN,
                title=None,
                metadata={},
            )

    async def test_valid_pin_accepted(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """The happy path is untouched: pinning the agent's real version succeeds."""
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(
            pool, account_id=account_id, prefix="pin-cs-ok"
        )
        session = await sessions_service.create_session(
            pool,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
        assert session.agent_version == agent.version

    async def test_none_pin_accepted(self, pool_and_account: tuple[asyncpg.Pool[Any], str]) -> None:
        """``agent_version=None`` ("latest") needs no version and must pass."""
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(
            pool, account_id=account_id, prefix="pin-cs-none"
        )
        session = await sessions_service.create_session(
            pool,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=None,
            title=None,
            metadata={},
        )
        assert session.agent_version is None


class TestUpdateSessionPinValidation:
    async def test_repin_to_nonexistent_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """A re-pin via update (agent_version supplied, agent_id omitted) is
        validated against the resolved current agent — the path that could
        brick an already-running, previously-healthy session."""
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(pool, account_id=account_id, prefix="pin-us")
        session = await sessions_service.create_session(
            pool,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=None,
            title=None,
            metadata={},
        )
        with pytest.raises(NotFoundError):
            await sessions_service.update_session(
                pool, session.id, account_id=account_id, agent_version=_BAD_PIN
            )


class TestCreateTemplatePinValidation:
    async def test_nonexistent_pin_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(pool, account_id=account_id, prefix="pin-ct")
        with pytest.raises(NotFoundError):
            await templates_service.create_session_template(
                pool,
                account_id=account_id,
                name="pinned-template",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=_BAD_PIN,
                vault_ids=[],
                memory_store_ids=[],
                metadata={},
            )


class TestUpdateTemplatePinValidation:
    async def test_repin_to_nonexistent_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        pool, account_id = pool_and_account
        agent, env, _ = await seed_agent_env_session(pool, account_id=account_id, prefix="pin-ut")
        template = await templates_service.create_session_template(
            pool,
            account_id=account_id,
            name="repin-template",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        with pytest.raises(NotFoundError):
            await templates_service.update_session_template(
                pool, template.id, account_id=account_id, agent_version=_BAD_PIN
            )
