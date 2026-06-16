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
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

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


# ─── api->session invoke edge: caller-supplied environment_id ownership gate ───
# (#1130, ships with #1128's POST /v1/invocations.)
#
# Unlike session-> / run-> callers (which inherit the launcher's env), the API
# caller has no launcher to inherit from, so it supplies ``environment_id``
# directly — re-opening the caller-chosen-env surface ``create_run`` forecloses.
# #1130 gates that supply with the SAME ownership check ``create_session`` /
# ``create_run`` already enforce (#755): the supplied env must be owned by the
# *authenticated caller's* account (``get_environment(..., account_id=<caller>)``),
# never a body-supplied account. These tests pin that gate on the invoke path.


_API_ROOT = "acc_invoke_gate_root"
_CALLER = "acc_invoke_gate_caller"
_OTHER = "acc_invoke_gate_other"


@pytest.fixture
async def pool_invoke_gate(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, agent_id, caller_account_id)`` for the invoke ownership gate.

    Two sibling tenants under one root: ``_CALLER`` (the authenticated caller,
    owns an agent) and ``_OTHER`` (owns the env the gate must refuse). The
    session-wake defer is patched out — the gate fires before any wake, and
    these tests cover the ownership refusal, not the worker stepping.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ($1, NULL, TRUE,  'invoke-gate-root'),
                       ($2, $1,   FALSE, 'invoke-gate-caller'),
                       ($3, $1,   FALSE, 'invoke-gate-other')
                """,
                _API_ROOT,
                _CALLER,
                _OTHER,
            )
        agent, _env, _session = await seed_agent_env_session(
            pool, account_id=_CALLER, prefix="invoke-gate"
        )
        with mock.patch("aios.services.wake.defer_wake", new=AsyncMock()):
            yield pool, agent.id, _CALLER
    finally:
        runtime.pool = prev
        await pool.close()


class TestInvokeEnvironmentOwnershipGate:
    async def test_invoke_refuses_cross_tenant_environment_id(
        self,
        pool_invoke_gate: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """``invoke`` (POST /v1/invocations) must refuse a foreign ``environment_id``.

        The caller authenticates as ``_CALLER`` but supplies an env owned by
        ``_OTHER``. The ownership gate (``get_environment(..., account_id=_CALLER)``
        inside ``create_session``) raises :class:`NotFoundError` — the env
        "doesn't exist" from the caller's vantage — and NO session is created.
        Without the gate, a bare FK would accept the foreign id and leak its
        image / env-vars / networking into the caller's servicer session.
        """
        pool, agent_id, caller = pool_invoke_gate
        other_env = await environments_service.create_environment(
            pool, account_id=_OTHER, name="invoke-gate-foreign-env"
        )
        before = {s.id for s in await sessions_service.list_sessions(pool, account_id=caller)}
        with pytest.raises(NotFoundError):
            await sessions_service.invoke(
                pool,
                account_id=caller,
                target_kind="agent",
                target=agent_id,
                input="x",
                environment_id=other_env.id,
            )
        # Fail-closed: the refused invoke spawned NO new servicer session (the
        # gate raises inside create_session's transaction, before any insert).
        after = {s.id for s in await sessions_service.list_sessions(pool, account_id=caller)}
        assert after == before

    async def test_invoke_accepts_self_owned_environment_id(
        self,
        pool_invoke_gate: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """A self-owned ``environment_id`` passes the gate and binds the servicer.

        The caller supplies an env it owns; the ownership gate accepts it and
        the servicer session is created bound to that env.
        """
        pool, agent_id, caller = pool_invoke_gate
        own_env = await environments_service.create_environment(
            pool, account_id=caller, name="invoke-gate-own-env"
        )
        handle = await sessions_service.invoke(
            pool,
            account_id=caller,
            target_kind="agent",
            target=agent_id,
            input="x",
            environment_id=own_env.id,
        )
        assert handle.servicer_kind == "session"
        session = await sessions_service.get_session(pool, handle.servicer_id, account_id=caller)
        assert session.environment_id == own_env.id
