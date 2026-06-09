"""Workflow runs as credentialed principals — the credential substrate + attenuation.

Slice 1 of the credentialed-runs lane, against a real Postgres. No tool() call site
yet: the proof is in the asymmetries —
  * a run resolves a credential through its OWN bound vaults (``resolve_run_credential``),
    and a run with no binding resolves nothing;
  * **launch-time attenuation** — an agent launching a run can only bind vaults it holds;
  * **create-time attenuation** — an agent authoring a workflow can only declare a tool
    surface it holds;
  * the declared surface round-trips through the ``workflows`` columns.

``defer_run_wake`` is patched out — these tests exercise creation + resolution, not the
run loop.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.harness import runtime
from aios.mcp.client import resolve_auth_for_target_url_run
from aios.models.agents import Agent, HttpServerSpec, McpServerSpec
from aios.models.vaults import VaultCredentialCreate
from aios.services import agents as agents_service
from aios.services import vaults as vaults_service
from aios.services import workflows as wf_service

pytestmark = pytest.mark.integration

ACC = "acc_v"
ENV = "env_v"
_SCRIPT = "async def main(i):\n    return i\n"


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(32))


@pytest.fixture
async def vault_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + a seeded ``(account, environment)``; wakes patched out."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'vault-root')",
                ACC,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'vault-env', '{}'::jsonb, $2)",
                ENV,
                ACC,
            )
        with mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()):
            yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _make_agent(
    pool: asyncpg.Pool[Any], name: str, *, mcp_servers: list[McpServerSpec] | None = None
) -> Agent:
    return await agents_service.create_agent(
        pool,
        account_id=ACC,
        name=name,
        model="test/dummy",
        system="x",
        tools=[],
        mcp_servers=mcp_servers,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )


async def _make_session(
    pool: asyncpg.Pool[Any], agent: Agent, *, vault_ids: list[str] | None = None
) -> str:
    """Insert a bare session row (no sandbox/crypto) bound to ``agent``, optionally to vaults."""
    async with pool.acquire() as conn:
        session = await db_queries.insert_session(
            conn,
            account_id=ACC,
            agent_id=agent.id,
            environment_id=ENV,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
        if vault_ids:
            await db_queries.set_session_vaults(conn, session.id, vault_ids, account_id=ACC)
    return session.id


async def _make_vault(pool: asyncpg.Pool[Any], name: str) -> str:
    vault = await vaults_service.create_vault(pool, account_id=ACC, display_name=name, metadata={})
    return vault.id


async def _run_count(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM wf_runs WHERE account_id = $1", ACC))


async def _workflow_count(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM workflows WHERE account_id = $1", ACC))


async def test_resolver_asymmetry(vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox) -> None:
    """A run resolves the credential in its OWN bound vaults; an unbound run resolves nothing."""
    pool = vault_pool
    vault_id = await _make_vault(pool, "v-x")
    url = "https://api.example/mcp"
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault_id,
        body=VaultCredentialCreate(
            target_url=url, auth_type="bearer_header", token=SecretStr("tok-123")
        ),
    )
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="w-res", script=_SCRIPT)
    bound = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, vault_ids=[vault_id]
    )
    unbound = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV
    )

    resolved_vault_id, headers = await resolve_auth_for_target_url_run(
        pool, crypto_box, bound.id, url, account_id=ACC
    )
    assert resolved_vault_id == vault_id
    assert headers == {"Authorization": "Bearer tok-123"}

    none_id, none_headers = await resolve_auth_for_target_url_run(
        pool, crypto_box, unbound.id, url, account_id=ACC
    )
    assert none_id is None
    assert none_headers == {}

    # The resolve path is account-scoped: the same bound run resolved under a different
    # account sees nothing (a dropped account filter on the join would leak here).
    foreign = await resolve_auth_for_target_url_run(
        pool, crypto_box, bound.id, url, account_id="acc_intruder"
    )
    assert foreign == (None, {})


async def test_launch_time_attenuation(vault_pool: asyncpg.Pool[Any]) -> None:
    """A run can only bind vaults the launching agent holds; a breach rolls back."""
    pool = vault_pool
    vx = await _make_vault(pool, "v-x")
    vy = await _make_vault(pool, "v-y")
    agent = await _make_agent(pool, "launcher-agent")
    launcher = await _make_session(pool, agent, vault_ids=[vx])
    both = await _make_session(pool, agent, vault_ids=[vx, vy])
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="w-launch", script=_SCRIPT)

    # Held vault → succeeds and binds.
    ok = await wf_service.create_run(
        pool,
        account_id=ACC,
        workflow_id=wf.id,
        environment_id=ENV,
        vault_ids=[vx],
        launcher_session_id=launcher,
    )
    async with pool.acquire() as conn:
        assert await wf_queries.get_run_vault_ids(conn, ok.id, account_id=ACC) == [vx]

    # Strict subset of a richer launcher → succeeds.
    sub = await wf_service.create_run(
        pool,
        account_id=ACC,
        workflow_id=wf.id,
        environment_id=ENV,
        vault_ids=[vx],
        launcher_session_id=both,
    )
    async with pool.acquire() as conn:
        assert await wf_queries.get_run_vault_ids(conn, sub.id, account_id=ACC) == [vx]

    # Ungranted vault → ForbiddenError, and no run row leaks (txn rolled back).
    before = await _run_count(pool)
    with pytest.raises(ForbiddenError):
        await wf_service.create_run(
            pool,
            account_id=ACC,
            workflow_id=wf.id,
            environment_id=ENV,
            vault_ids=[vy],
            launcher_session_id=launcher,
        )
    assert await _run_count(pool) == before

    # Operator path (no launcher) binds anything, account-scoped.
    op = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, vault_ids=[vy]
    )
    async with pool.acquire() as conn:
        assert await wf_queries.get_run_vault_ids(conn, op.id, account_id=ACC) == [vy]


async def test_create_time_attenuation(vault_pool: asyncpg.Pool[Any]) -> None:
    """A workflow may declare only servers/tools the creating agent itself has."""
    pool = vault_pool
    s1 = McpServerSpec(name="s1", url="https://s1.example")
    s2 = McpServerSpec(name="s2", url="https://s2.example")
    creator_agent = await _make_agent(pool, "creator-agent", mcp_servers=[s1])
    creator = await _make_session(pool, creator_agent)

    # Declares a subset of the creator's surface → ok.
    ok = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-ct-ok",
        script=_SCRIPT,
        mcp_servers=[s1],
        creator_session_id=creator,
    )
    assert [m.url for m in ok.mcp_servers] == ["https://s1.example"]

    # Declares a server the creator lacks → ForbiddenError, no workflow row leaks.
    before = await _workflow_count(pool)
    with pytest.raises(ForbiddenError):
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-ct-bad",
            script=_SCRIPT,
            mcp_servers=[s2],
            creator_session_id=creator,
        )
    assert await _workflow_count(pool) == before

    # Operator path (no creator) declares anything.
    op = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-ct-op",
        script=_SCRIPT,
        mcp_servers=[s2],
        creator_session_id=None,
    )
    assert [m.url for m in op.mcp_servers] == ["https://s2.example"]


async def test_declared_surface_round_trips(vault_pool: asyncpg.Pool[Any]) -> None:
    """mcp_servers / http_servers persist on the workflow and read back equal."""
    pool = vault_pool
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-surface",
        script=_SCRIPT,
        mcp_servers=[McpServerSpec(name="s", url="https://s.example")],
        http_servers=[HttpServerSpec(name="h", base_url="https://h.example")],
    )
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    assert [m.url for m in fetched.mcp_servers] == ["https://s.example"]
    assert [h.base_url for h in fetched.http_servers] == ["https://h.example"]


async def test_run_cannot_bind_foreign_or_missing_vault(vault_pool: asyncpg.Pool[Any]) -> None:
    """A run binds only its own account's vaults; a foreign/unknown id rolls the insert back.

    This is the operator path (no launcher) — the only path where an over-broad vault set
    isn't already bounded by a launcher's holdings — so the account-scoped binding guard is
    the load-bearing check. It also exercises the genuine rollback: ``insert_wf_run`` lands,
    then ``set_run_vaults`` raises, and the outer transaction must leave no run row.
    """
    pool = vault_pool
    async with pool.acquire() as conn:
        # A child account (only one active root is allowed — accounts_one_active_root).
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_other', $1, FALSE, 'other-tenant')",
            ACC,
        )
    foreign_vault = await vaults_service.create_vault(
        pool, account_id="acc_other", display_name="other-v", metadata={}
    )
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="w-foreign", script=_SCRIPT)

    before = await _run_count(pool)
    # Another account's (existing) vault — the FK alone would accept it; the ownership
    # guard rejects it, and the insert that already ran rolls back.
    with pytest.raises(NotFoundError):
        await wf_service.create_run(
            pool,
            account_id=ACC,
            workflow_id=wf.id,
            environment_id=ENV,
            vault_ids=[foreign_vault.id],
        )
    assert await _run_count(pool) == before

    # A plainly nonexistent vault id is likewise NotFound, no run row.
    with pytest.raises(NotFoundError):
        await wf_service.create_run(
            pool,
            account_id=ACC,
            workflow_id=wf.id,
            environment_id=ENV,
            vault_ids=["vlt_does_not_exist"],
        )
    assert await _run_count(pool) == before


async def test_update_time_attenuation(vault_pool: asyncpg.Pool[Any]) -> None:
    """An agent-actor may only update a workflow whose merged final surface it could have
    declared itself — enforced even for a script-only edit (else an agent could rewrite the
    script of an operator-created workflow with a broader surface and wield it). The
    operator path (no actor) updates anything."""
    pool = vault_pool
    s1 = McpServerSpec(name="s1", url="https://s1.example")
    s2 = McpServerSpec(name="s2", url="https://s2.example")
    agent = await _make_agent(pool, "updater-agent", mcp_servers=[s1])
    actor = await _make_session(pool, agent)

    # Operator creates a workflow whose surface exceeds the agent's (s2 ∉ agent's).
    broad = await wf_service.create_workflow(
        pool, account_id=ACC, name="w-broad", script=_SCRIPT, mcp_servers=[s2]
    )
    # Script-only edit by the agent → ForbiddenError (merged surface still ⊉ agent's).
    with pytest.raises(ForbiddenError):
        await wf_service.update_workflow(
            pool,
            broad.id,
            account_id=ACC,
            expected_version=1,
            script="async def main(i):\n    return 2\n",
            actor_session_id=actor,
        )

    # On a workflow within the agent's surface, the same edit succeeds.
    narrow = await wf_service.create_workflow(
        pool, account_id=ACC, name="w-narrow", script=_SCRIPT, mcp_servers=[s1]
    )
    ok = await wf_service.update_workflow(
        pool,
        narrow.id,
        account_id=ACC,
        expected_version=1,
        script="async def main(i):\n    return 2\n",
        actor_session_id=actor,
    )
    assert ok.version == 2

    # The agent cannot widen a surface beyond its own.
    with pytest.raises(ForbiddenError):
        await wf_service.update_workflow(
            pool,
            narrow.id,
            account_id=ACC,
            expected_version=2,
            mcp_servers=[s1, s2],
            actor_session_id=actor,
        )

    # An actor's token must match the version the attenuation read observes — a FUTURE
    # token 409s up front (else: pass attenuation against today's surface, let a
    # concurrent broadening update bump to exactly that token, and land unchecked).
    # Probed against the BROAD workflow so the assertion discriminates the service-layer
    # pin specifically: with the pin, ConflictError fires before attenuation; without
    # it, attenuation against the broad surface would raise ForbiddenError instead.
    with pytest.raises(ConflictError):
        await wf_service.update_workflow(
            pool,
            broad.id,
            account_id=ACC,
            expected_version=2,
            script="async def main(i):\n    return 3\n",
            actor_session_id=actor,
        )

    # Operator (no actor) updates anything, including the broad workflow.
    op = await wf_service.update_workflow(
        pool, broad.id, account_id=ACC, expected_version=1, description="op-edit"
    )
    assert op.version == 2
