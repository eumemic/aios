"""Integration tests: session-row reads that JOIN to ``environments``
or ``agents`` must filter the joined row by ``account_id``, not just
the session's ``account_id``.

``insert_session`` does not currently validate that ``environment_id``
or ``agent_id`` belongs to the same account as the session — the FK
only requires existence in the target table. Without account-side
predicates in the JOIN, a session row with a cross-tenant
``environment_id`` or ``agent_id`` would surface the foreign tenant's
config / bound model into the worker's step context every step. This
is a cross-tenant data leak [security].

Two read paths are covered:

* ``get_environment_config_for_session`` — exposes
  :class:`EnvironmentConfig` (env vars, networking rules, package
  list). Called from ``sandbox/spec.py:109`` per step.
* ``get_session_model`` — exposes the bound model string (which may
  itself encode a sensitive routing target / LiteLLM credentialed
  route). Called from worker dispatch per step.

The upstream write-side gap (``insert_session`` / ``update_session``
not validating cross-tenant FKs) is a separate broken-invariant
defect. These tests isolate the read-side defenses; once write-side
validation lands, the fixture below will need to inject the bad row
via direct SQL INSERT (legacy rows from before that validation could
still exist on long-lived deployments).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_acc_a_session_with_acc_b_refs(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, session_id)`` for an acc_a session whose
    ``environment_id`` AND ``agent_id`` both reference acc_b-owned
    resources.

    The cross-tenant FK references are constructed via
    ``queries.insert_session`` directly, sidestepping any service-layer
    validation — the test exercises the read path's defenses against a
    pre-existing invalid row, regardless of how it got there. Once
    write-side validation lands, this fixture will need to inject via
    direct SQL INSERT instead.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                       ('acc_b',    'acc_root', FALSE, 'tenant-b')
                """
            )

        # Tenant B owns both the agent and the environment — neither
        # should ever surface in a read scoped to tenant A.
        agent_b = await agents_service.create_agent(
            pool,
            account_id="acc_b",
            name="b-agent",
            model="openrouter/secret-tenant-b-model",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env_b = await environments_service.create_environment(
            pool,
            account_id="acc_b",
            name="b-env",
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_a",
                agent_id=agent_b.id,
                environment_id=env_b.id,
                agent_version=agent_b.version,
                title=None,
                metadata={},
            )
        yield pool, session.id
    finally:
        await pool.close()


class TestSessionCrossTenantIsolation:
    async def test_get_environment_config_for_session_blocks_cross_tenant_env(
        self,
        pool_acc_a_session_with_acc_b_refs: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """A session for acc_a whose ``environment_id`` points at an
        acc_b environment must NOT load that environment's config. The
        SQL must filter both ``s.account_id`` AND ``e.account_id``;
        without the latter, the read returns acc_b's config to acc_a's
        worker — a cross-tenant exfiltration of env vars, networking
        rules, and package configuration.
        """
        pool, session_id = pool_acc_a_session_with_acc_b_refs
        async with pool.acquire() as conn:
            result = await queries.get_environment_config_for_session(
                conn, session_id, account_id="acc_a"
            )
        assert result is None, (
            "cross-tenant environment-config leak: acc_a's session "
            "carrying acc_b's environment_id returned acc_b's "
            "EnvironmentConfig. The JOIN in "
            "get_environment_config_for_session must filter "
            "e.account_id, not just s.account_id."
        )

    async def test_get_session_model_blocks_cross_tenant_agent(
        self,
        pool_acc_a_session_with_acc_b_refs: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """A session for acc_a whose ``agent_id`` points at an acc_b
        agent must NOT resolve to acc_b's bound model. The SQL must
        filter ``s.account_id``, ``a.account_id``, and
        ``av.account_id`` — without these the worker would dispatch
        model calls under tenant B's routing for a tenant A session.
        Raising :class:`NotFoundError` is the correct fail-closed
        outcome (the row "doesn't exist" from acc_a's vantage).
        """
        pool, session_id = pool_acc_a_session_with_acc_b_refs
        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.get_session_model(conn, session_id, account_id="acc_a")
