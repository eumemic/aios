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
from aios.errors import ConflictError, ForbiddenError, NotFoundError, RateLimitedError
from aios.harness import runtime
from aios.mcp.client import resolve_auth_for_target_url_run
from aios.models.agents import (
    Agent,
    HttpPermissionPolicy,
    HttpRouteSpec,
    HttpServerSpec,
    McpServerSpec,
    ToolSpec,
)
from aios.models.attenuation import surface_of
from aios.models.vaults import VaultCredentialCreate
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services import sessions as sessions_service
from aios.services import vaults as vaults_service
from aios.services import workflows as wf_service
from aios.services.sessions import create_child_session
from aios.tools import workflow_management as wm
from aios.workflows import service as wf_service_module
from aios.workflows.child_id import child_session_id
from aios.workflows.service import INVOKE_MAX_DEPTH, WorkflowRunDepthExceededError

pytestmark = pytest.mark.integration

ACC = "acc_v"
ENV = "env_v"
_SCRIPT = "async def main(i):\n    return i\n"


async def _create_run_via_session(
    pool: asyncpg.Pool[Any], session_id: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Launch a run inheriting the caller session's env + lineage — the wiring the
    removed ``create_run`` builtin did — so these vault-attenuation tests still
    exercise the same ``wf_service.create_run`` clamp (now reached via call_workflow)."""
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    run = await wf_service.create_run(
        pool,
        account_id=account_id,
        workflow_id=args["workflow_id"],
        environment_id=session.environment_id,
        input=args.get("input"),
        vault_ids=args.get("vault_ids", []),
        launcher_session_id=session_id,
        parent_run_id=session.parent_run_id,
        budget_usd=args.get("budget_usd"),
    )
    return run.model_dump(mode="json")


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
        # Both defer_run_wake bindings: create_run uses aios.workflows.service's,
        # cancel_run uses aios.services.workflows' own import (same split the e2e
        # conftest patches).
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _make_agent(
    pool: asyncpg.Pool[Any],
    name: str,
    *,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
) -> Agent:
    return await agents_service.create_agent(
        pool,
        account_id=ACC,
        name=name,
        model="test/dummy",
        system="x",
        tools=tools or [],
        mcp_servers=mcp_servers,
        http_servers=http_servers,
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
    with pytest.raises(ForbiddenError, match=r"exceeds the acting agent's permissions"):
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


async def test_create_time_attenuation_rejects_per_tool_widening(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """The fixpoint predicate rejects *widening* a tool's policy, not just adding tools.

    The agent holds ``bash`` at ``always_ask``. A workflow declaring the same ``bash``
    at ``always_allow`` is a looser policy on an identically-named tool — membership
    alone would admit it, but it widens the agent's surface, so the fixpoint predicate
    rejects it. Re-declaring ``always_ask`` is a fixpoint and passes. This is the half
    the old membership check missed.
    """
    pool = vault_pool
    agent = await _make_agent(
        pool, "pin-agent", tools=[ToolSpec(type="bash", permission="always_ask")]
    )
    creator = await _make_session(pool, agent)

    before = await _workflow_count(pool)
    with pytest.raises(ForbiddenError):
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-widen",
            script=_SCRIPT,
            tools=[ToolSpec(type="bash", permission="always_allow")],
            creator_session_id=creator,
        )
    assert await _workflow_count(pool) == before

    # Re-declaring the identical policy is a fixpoint → admitted.
    ok = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-pin",
        script=_SCRIPT,
        tools=[ToolSpec(type="bash", permission="always_ask")],
        creator_session_id=creator,
    )
    assert ok.name == "w-pin"


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


# ─── #939: http_servers admitted by identity, inherited launcher-frozen ──────
#
# The authoring gate matches http_servers on identity (name + base_url) like tools/mcp,
# and INHERITS the agent's frozen routes into workflow storage — so a workflow declaring
# the right server but partial/empty routes is admitted and stored with the agent's full
# routes. Membership still gates: a server the agent lacks is still forbidden. Run-time
# stays parent-wins-frozen (run-launch clamps to launcher-verbatim), so relaxing the
# AUTHORING gate grants no new run-time authority.


async def test_create_time_http_identity_inherits_frozen_routes(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Create-time: a workflow declaring the agent's http server by identity (partial/empty
    routes) is admitted, and stores the agent's full routes launcher-frozen."""
    pool = vault_pool
    agent_http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/v1/**",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            )
        ],
    )
    agent = await _make_agent(pool, "http-creator", http_servers=[agent_http])
    creator = await _make_session(pool, agent)

    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-http-ct",
        script=_SCRIPT,
        http_servers=[HttpServerSpec(name="api", base_url="https://api", routes=[])],
        creator_session_id=creator,
    )
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    # Routes inherited launcher-frozen — equals the agent's full spec, NOT the declared
    # empty-routes one.
    assert fetched.http_servers == [agent_http]


async def test_create_time_http_missing_server_still_forbidden(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Create-time: a server the agent lacks (different identity) is still forbidden — the
    identity relaxation is membership-keyed, not a blanket admit. No workflow row leaks."""
    pool = vault_pool
    agent_http = HttpServerSpec(name="api", base_url="https://api")
    agent = await _make_agent(pool, "http-creator-2", http_servers=[agent_http])
    creator = await _make_session(pool, agent)

    before = await _workflow_count(pool)
    with pytest.raises(ForbiddenError, match=r"exceeds the acting agent's permissions") as ei:
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-http-missing",
            script=_SCRIPT,
            http_servers=[HttpServerSpec(name="other", base_url="https://other")],
            creator_session_id=creator,
        )
    assert ei.value.detail["exceeds"]["http_servers"] == ["other"]
    assert await _workflow_count(pool) == before


async def test_update_time_http_identity_inherits_frozen_routes(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Update-time: an actor re-declaring the agent's http server by identity (empty routes)
    is admitted and the stored routes stay the agent's full, launcher-frozen spec."""
    pool = vault_pool
    agent_http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/v1/**",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            )
        ],
    )
    agent = await _make_agent(pool, "http-updater", http_servers=[agent_http])
    actor = await _make_session(pool, agent)
    # Operator-create a workflow storing the server with EMPTY routes verbatim (operator
    # path), version 1 — so the actor's inherited-frozen update is a genuine change.
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-http-up",
        script=_SCRIPT,
        http_servers=[HttpServerSpec(name="api", base_url="https://api", routes=[])],
    )
    assert wf.http_servers == [HttpServerSpec(name="api", base_url="https://api", routes=[])]

    updated = await wf_service.update_workflow(
        pool,
        wf.id,
        account_id=ACC,
        expected_version=1,
        http_servers=[HttpServerSpec(name="api", base_url="https://api", routes=[])],
        actor_session_id=actor,
    )
    assert updated.version == 2
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    # The actor declared empty routes but the agent's full routes are inherited frozen.
    assert fetched.http_servers == [agent_http]


async def test_update_time_http_missing_server_still_forbidden(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Update-time: re-pointing to a server the agent lacks is still forbidden; the workflow
    stays at version 1 (no write)."""
    pool = vault_pool
    agent_http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/v1/**",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            )
        ],
    )
    agent = await _make_agent(pool, "http-updater-2", http_servers=[agent_http])
    actor = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="w-http-ghost", script=_SCRIPT, http_servers=[agent_http]
    )

    with pytest.raises(ForbiddenError, match=r"exceeds the acting agent's permissions"):
        await wf_service.update_workflow(
            pool,
            wf.id,
            account_id=ACC,
            expected_version=1,
            http_servers=[HttpServerSpec(name="ghost", base_url="https://ghost")],
            actor_session_id=actor,
        )
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    assert fetched.version == 1


# ─── #953: names-only http_servers declaration (Ask 3 follow-up) ─────────────
#
# A workflow author may reference the acting agent's http server by NAME ALONE —
# ``http_servers=["api"]`` — instead of constructing a full HttpServerSpec whose
# (name, base_url) must identity-match the agent's. The bare name resolves against
# the acting agent (the agent's base_url + frozen routes inherited into storage),
# exactly as the #949 identity-match path does. An unknown name fails closed; the
# operator path (no acting agent) rejects bare names. Aliasing (a divergent local
# name) is the open fork and is NOT shipped — run-time resolution is untouched.


async def test_create_time_http_names_only_resolves_against_agent(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Create-time: a names-only entry resolves to the agent's server and stores the
    agent's full launcher-frozen routes — identical to declaring the spec by identity."""
    pool = vault_pool
    agent_http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/v1/**",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            )
        ],
    )
    agent = await _make_agent(pool, "http-names-only", http_servers=[agent_http])
    creator = await _make_session(pool, agent)

    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-http-names-only",
        script=_SCRIPT,
        http_servers=["api"],  # names-only sugar
        creator_session_id=creator,
    )
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    assert fetched.http_servers == [agent_http]


async def test_create_time_http_names_only_unknown_name_forbidden(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Create-time: a bare name the agent does not grant fails closed (no row leaks)."""
    pool = vault_pool
    agent_http = HttpServerSpec(name="api", base_url="https://api")
    agent = await _make_agent(pool, "http-names-only-2", http_servers=[agent_http])
    creator = await _make_session(pool, agent)

    before = await _workflow_count(pool)
    with pytest.raises(ForbiddenError, match=r"exceeds the acting agent's permissions") as ei:
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-http-names-ghost",
            script=_SCRIPT,
            http_servers=["ghost"],
            creator_session_id=creator,
        )
    assert ei.value.detail["exceeds"]["http_servers"] == ["ghost"]
    assert await _workflow_count(pool) == before


async def test_create_time_http_names_only_rejected_on_operator_path(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Operator path (no creator): names-only sugar has no agent to resolve against and is
    rejected; the operator must declare a full HttpServerSpec."""
    pool = vault_pool
    before = await _workflow_count(pool)
    with pytest.raises(ForbiddenError, match=r"names-only http_servers require an acting agent"):
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-http-op-names",
            script=_SCRIPT,
            http_servers=["api"],
        )
    assert await _workflow_count(pool) == before


async def test_update_time_http_names_only_resolves_against_agent(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Update-time: a names-only entry resolves against the acting agent and stores the
    agent's full launcher-frozen routes."""
    pool = vault_pool
    agent_http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/v1/**",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            )
        ],
    )
    agent = await _make_agent(pool, "http-names-up", http_servers=[agent_http])
    actor = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="w-http-names-up",
        script=_SCRIPT,
        http_servers=[HttpServerSpec(name="api", base_url="https://api", routes=[])],
    )

    updated = await wf_service.update_workflow(
        pool,
        wf.id,
        account_id=ACC,
        expected_version=1,
        http_servers=["api"],  # names-only sugar
        actor_session_id=actor,
    )
    assert updated.version == 2
    fetched = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    assert fetched.http_servers == [agent_http]


async def test_create_time_http_name_mismatch_reports_base_url(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """A full spec declaring the agent's base_url under a DIFFERENT name is rejected; the
    exceeds detail reads as a name mismatch at the base_url, not a bare absent name (#953
    legibility nicety). The authoring gate is unchanged — still rejected."""
    pool = vault_pool
    agent_http = HttpServerSpec(name="api", base_url="https://api")
    agent = await _make_agent(pool, "http-mismatch", http_servers=[agent_http])
    creator = await _make_session(pool, agent)

    with pytest.raises(ForbiddenError, match=r"exceeds the acting agent's permissions") as ei:
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="w-http-mismatch",
            script=_SCRIPT,
            http_servers=[HttpServerSpec(name="aliased", base_url="https://api")],
            creator_session_id=creator,
        )
    assert ei.value.detail["exceeds"]["http_servers"] == ["name mismatch at https://api"]


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


# ─── agent-acting workflow builtins (slice 3) ────────────────────────────────
#
# The handlers call the same attenuated services, supplying the executing session as
# the trusted actor. These drive the handlers directly (the model dispatch path is
# what passes session_id; here the test plays that role).


async def test_create_workflow_builtin_attenuates(vault_pool: asyncpg.Pool[Any]) -> None:
    """create_workflow via the builtin: the declared surface ⊆ the executing agent's."""
    pool = vault_pool
    s1 = McpServerSpec(name="s1", url="https://s1.example")
    s2 = McpServerSpec(name="s2", url="https://s2.example")
    agent = await _make_agent(pool, "builtin-author", mcp_servers=[s1])
    sess = await _make_session(pool, agent)

    out = await wm.create_workflow_handler(
        sess, {"name": "wf-bi-ok", "script": _SCRIPT, "mcp_servers": [s1.model_dump(mode="json")]}
    )
    assert out["name"] == "wf-bi-ok" and "script" not in out  # heavy field trimmed

    # A server the agent lacks → ForbiddenError, which the dispatch layer renders as a
    # clean model-visible refusal (the handler propagates it; see test_tool_dispatch).
    with pytest.raises(ForbiddenError):
        await wm.create_workflow_handler(
            sess,
            {"name": "wf-bi-bad", "script": _SCRIPT, "mcp_servers": [s2.model_dump(mode="json")]},
        )


async def test_create_run_builtin_vault_attenuation_and_env(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """create_run via the builtin: vaults ⊆ the session's; the run inherits the caller's env."""
    pool = vault_pool
    vx = await _make_vault(pool, "vx")
    vy = await _make_vault(pool, "vy")
    agent = await _make_agent(pool, "builtin-launcher")
    sess = await _make_session(pool, agent, vault_ids=[vx])
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-bi-run", script=_SCRIPT)

    out = await _create_run_via_session(pool, sess, {"workflow_id": wf.id, "vault_ids": [vx]})
    assert out["status"] == "pending"
    assert out["environment_id"] == ENV  # inherited from the caller session (not model input)
    assert out["parent_run_id"] is None  # a foreground session launches a root run
    async with pool.acquire() as conn:
        assert await wf_queries.get_run_vault_ids(conn, out["id"], account_id=ACC) == [vx]

    # A vault the session doesn't hold → ForbiddenError (dispatch renders it model-visible),
    # no run row leaks.
    before = await _run_count(pool)
    with pytest.raises(ForbiddenError):
        await _create_run_via_session(pool, sess, {"workflow_id": wf.id, "vault_ids": [vy]})
    assert await _run_count(pool) == before


async def test_create_run_builtin_threads_parent_run_id(vault_pool: asyncpg.Pool[Any]) -> None:
    """The handler threads the caller session's parent_run_id, recording run lineage."""
    pool = vault_pool
    agent = await _make_agent(pool, "builtin-nested")
    sess = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-bi-nest", script=_SCRIPT)
    root = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)

    # Make the session a real child: parent_run_id + a frozen surface (the run-spawn
    # machinery populates both; load_for_session fails closed on a parent_run_id session
    # that is not surface_frozen). This asserts the handler READS the lineage and threads it.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET parent_run_id = $1, surface_frozen = TRUE, "
            "tools = '[]'::jsonb, mcp_servers = '[]'::jsonb, http_servers = '[]'::jsonb "
            "WHERE id = $2",
            root.id,
            sess,
        )

    out = await _create_run_via_session(pool, sess, {"workflow_id": wf.id})
    assert out["parent_run_id"] == root.id


async def test_create_run_depth_cap(vault_pool: asyncpg.Pool[Any]) -> None:
    """The DOWN-counting depth budget (#1124): a parent_run_id chain may take
    INVOKE_MAX_DEPTH hops, no more. The edgeless root seeds at the full budget,
    each child carries ``parent.depth - 1``, and the INVOKE_MAX_DEPTH-th run
    bottoms out at depth 1 — the next launch refuses before any row is written."""
    pool = vault_pool
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-depth", script=_SCRIPT)

    # Build a chain: root seeds at INVOKE_MAX_DEPTH, then INVOKE_MAX_DEPTH-1 children
    # decrementing to depth 1. INVOKE_MAX_DEPTH runs total, identical to the old up-walk.
    parent: str | None = None
    expected_depth = INVOKE_MAX_DEPTH
    for _ in range(INVOKE_MAX_DEPTH):
        run = await wf_service.create_run(
            pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, parent_run_id=parent
        )
        assert run.depth == expected_depth  # the down-counter decrements each hop
        parent = run.id
        expected_depth -= 1

    # The leaf bottomed out at depth 1 → one more would be depth 0, refused before write.
    before = await _run_count(pool)
    with pytest.raises(WorkflowRunDepthExceededError):
        await wf_service.create_run(
            pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, parent_run_id=parent
        )
    assert await _run_count(pool) == before


async def test_create_run_edgeless_root_seeds_full_budget(vault_pool: asyncpg.Pool[Any]) -> None:
    """An edgeless root — the operator/HTTP ``POST /runs`` path with no parent — is
    seeded at the full shared budget (#1124), so a chain launched off it still has
    INVOKE_MAX_DEPTH hops before it bottoms out."""
    pool = vault_pool
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-root", script=_SCRIPT)

    root = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, parent_run_id=None
    )
    assert root.depth == INVOKE_MAX_DEPTH

    # The persisted column agrees with the returned row — the read-side the next hop uses.
    async with pool.acquire() as conn:
        assert await wf_queries.get_run_depth(conn, root.id, account_id=ACC) == INVOKE_MAX_DEPTH


async def test_get_run_depth_account_scoped(vault_pool: asyncpg.Pool[Any]) -> None:
    """``get_run_depth`` is account-scoped (#1124): a foreign id raises NotFoundError,
    preserving the same-account trust the deleted ``run_ancestor_depth`` CTE enforced
    per hop — a foreign parent can never launder a fresh full budget."""
    pool = vault_pool
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-scope", script=_SCRIPT)
    root = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, parent_run_id=None
    )
    async with pool.acquire() as conn:
        with pytest.raises(NotFoundError):
            await wf_queries.get_run_depth(conn, root.id, account_id="acc_other")


async def test_create_run_rejects_foreign_environment(vault_pool: asyncpg.Pool[Any]) -> None:
    """F2: a run's environment_id must be account-owned — a bare FK would accept a
    foreign tenant's env and leak its image/env-vars. Bounds the HTTP path too."""
    pool = vault_pool
    async with pool.acquire() as conn:
        # A second tenant (a child account — ACC is already the one active root).
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_foreign', $1, FALSE, 'foreign')",
            ACC,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, config, account_id) "
            "VALUES ('env_foreign', 'fe', '{}'::jsonb, 'acc_foreign')"
        )
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-env", script=_SCRIPT)

    before = await _run_count(pool)
    with pytest.raises(NotFoundError):
        await wf_service.create_run(
            pool, account_id=ACC, workflow_id=wf.id, environment_id="env_foreign"
        )
    assert await _run_count(pool) == before


# ─── horizontal fan-out caps + launcher-attenuated cancel ────────────────────


def _cap_settings(monkeypatch: pytest.MonkeyPatch, **caps: int) -> None:
    """Patch ``aios.workflows.service.get_settings`` with lowered fan-out caps."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update=caps)
    monkeypatch.setattr("aios.workflows.service.get_settings", lambda: capped)


async def _force_terminal(pool: asyncpg.Pool[Any], run_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute("UPDATE wf_runs SET status = 'completed' WHERE id = $1", run_id)


async def test_launcher_fanout_cap(
    vault_pool: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launcher session may hold at most N outstanding runs; slots free as runs
    reach a terminal status. The operator path is exempt from the launcher cap."""
    pool = vault_pool
    _cap_settings(monkeypatch, workflow_runs_per_launcher_max=2)
    agent = await _make_agent(pool, "fanout-agent")
    sess = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-cap", script=_SCRIPT)

    first = await _create_run_via_session(pool, sess, {"workflow_id": wf.id})
    assert first["launcher_session_id"] == sess  # persisted lineage (the cap's count key)
    await _create_run_via_session(pool, sess, {"workflow_id": wf.id})

    before = await _run_count(pool)
    with pytest.raises(RateLimitedError, match="outstanding-run cap"):
        await _create_run_via_session(pool, sess, {"workflow_id": wf.id})
    assert await _run_count(pool) == before  # refusal leaves no row

    # Self-healing: a run reaching terminal frees the slot.
    await _force_terminal(pool, first["id"])
    await _create_run_via_session(pool, sess, {"workflow_id": wf.id})

    # Operator launches carry no launcher (and are exempt from the launcher cap).
    op = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    assert op.launcher_session_id is None


async def test_account_fanout_cap_binds_every_launch(
    vault_pool: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The account cap is the backstop: it counts agent AND operator launches."""
    pool = vault_pool
    _cap_settings(monkeypatch, workflow_runs_per_account_max=2)
    agent = await _make_agent(pool, "fanout-acct-agent")
    sess = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-acap", script=_SCRIPT)

    await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    await _create_run_via_session(pool, sess, {"workflow_id": wf.id})  # agent launch counts too

    # Third launch refused on BOTH paths.
    with pytest.raises(RateLimitedError, match="account at outstanding-run cap"):
        await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    with pytest.raises(RateLimitedError, match="account at outstanding-run cap"):
        await _create_run_via_session(pool, sess, {"workflow_id": wf.id})


async def test_cancel_run_builtin_only_own_runs(vault_pool: asyncpg.Pool[Any]) -> None:
    """cancel_run cancels only runs the executing session launched — the self-service
    escape for the launcher cap; operator-launched runs are out of its reach."""
    pool = vault_pool
    agent = await _make_agent(pool, "canceller-agent")
    sess = await _make_session(pool, agent)
    other = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-cxl", script=_SCRIPT)

    own = await _create_run_via_session(pool, sess, {"workflow_id": wf.id})
    out = await wm.cancel_run_handler(sess, {"run_id": own["id"]})
    assert out["id"] == own["id"]  # accepted; the run finalizes on its next wake
    async with pool.acquire() as conn:
        signal = await conn.fetchrow(
            "SELECT 1 FROM wf_run_signals WHERE run_id = $1 AND kind = 'cancel'", own["id"]
        )
    assert signal is not None

    # Another session's run and an operator run are both forbidden.
    op = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    with pytest.raises(ForbiddenError):
        await wm.cancel_run_handler(other, {"run_id": own["id"]})
    with pytest.raises(ForbiddenError):
        await wm.cancel_run_handler(sess, {"run_id": op.id})


# ─── #794: the run→child materialize edge (frozen, run-attenuated surface) ────
#
# A run's agent() child must wield agent ∩ run, frozen at spawn, with the run's vaults —
# closing the "agent-shaped side door". These spawn children the way step.py does
# (clamp(agent, run) then create_child_session) and read the frozen surface back through
# load_for_session, the one chokepoint every reader inherits.


async def _spawn_child(
    pool: asyncpg.Pool[Any],
    run_id: str,
    agent: Agent,
    *,
    surface: Any,
    vault_ids: list[str] | None = None,
) -> Any:
    """Spawn one child for ``run_id`` (as step.py does) and return its Session."""
    cid = child_session_id(run_id, "0")
    await create_child_session(
        pool,
        session_id=cid,
        account_id=ACC,
        agent_id=agent.id,
        environment_id=ENV,
        agent_version=agent.version,
        model=None,
        parent_run_id=run_id,
        surface=surface,
        vault_ids=vault_ids or [],
        request_id="0",
        input="hi",
    )
    return await sessions_service.get_session_basic(pool, cid, account_id=ACC)


async def test_child_surface_is_the_frozen_clamp_not_the_live_agent(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """The setuid regression: a child wields agent ∩ run, NOT the agent's full surface.

    The agent has bash + read; the run only bash. The child — read back through
    load_for_session, the chokepoint every reader uses — has bash but NOT read, even
    though its agent does. Anything the run can't do, its child can't do either.
    """
    pool = vault_pool
    agent = await _make_agent(
        pool, "db-admin", tools=[ToolSpec(type="bash"), ToolSpec(type="read")]
    )
    # Operator-authored workflow with the run's (narrower) surface; launched operator-side.
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="wf-run-bash", script=_SCRIPT, tools=[ToolSpec(type="bash")]
    )
    run = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    child_surface = attenuation_service.clamp(surface_of(agent), surface_of(run))

    child = await _spawn_child(pool, run.id, agent, surface=child_surface)
    loaded = await agents_service.load_for_session(pool, child, account_id=ACC)
    assert {t.type for t in loaded.tools} == {"bash"}  # read dropped — the side door closed


async def test_child_vaults_copied_from_run(vault_pool: asyncpg.Pool[Any]) -> None:
    """The credential half (#794 P1=A): a child's session_vaults are the run's, frozen."""
    pool = vault_pool
    vx = await _make_vault(pool, "v-x")
    agent = await _make_agent(pool, "vault-child-agent")
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-vc", script=_SCRIPT)
    run = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, vault_ids=[vx]
    )
    async with pool.acquire() as conn:
        run_vaults = await wf_queries.get_run_vault_ids(conn, run.id, account_id=ACC)

    child = await _spawn_child(pool, run.id, agent, surface=surface_of(agent), vault_ids=run_vaults)
    async with pool.acquire() as conn:
        child_vaults = await db_queries.get_session_vault_ids(conn, child.id, account_id=ACC)
    assert child_vaults == [vx]


async def test_frozen_surface_wins_over_a_later_agent_edit(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """Replay-soundness: the child reads its spawn-time frozen surface, not the live
    agent, even after the agent's surface is rewritten (a new version)."""
    pool = vault_pool
    agent = await _make_agent(pool, "evolving-agent", tools=[ToolSpec(type="bash")])
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="wf-frozen", script=_SCRIPT, tools=[ToolSpec(type="bash")]
    )
    run = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    child = await _spawn_child(
        pool, run.id, agent, surface=attenuation_service.clamp(surface_of(agent), surface_of(run))
    )

    # Rewrite the agent's surface (a new version) AFTER the child froze.
    await agents_service.update_agent(
        pool,
        agent.id,
        account_id=ACC,
        expected_version=agent.version,
        tools=[ToolSpec(type="read")],
    )

    loaded = await agents_service.load_for_session(pool, child, account_id=ACC)
    assert {t.type for t in loaded.tools} == {"bash"}  # the frozen surface, not the new [read]


async def test_load_for_session_fails_closed_on_unfrozen_child(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """A parent_run_id session with no frozen snapshot fails closed — never the full agent."""
    pool = vault_pool
    agent = await _make_agent(pool, "ghost-child-agent", tools=[ToolSpec(type="bash")])
    sess_id = await _make_session(pool, agent)
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="wf-ghost", script=_SCRIPT)
    run = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    # Make it look like a child (parent_run_id) WITHOUT a frozen surface — the corrupt state.
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET parent_run_id = $1 WHERE id = $2", run.id, sess_id)
    sess = await sessions_service.get_session_basic(pool, sess_id, account_id=ACC)
    with pytest.raises(RuntimeError, match="no frozen surface"):
        await agents_service.load_for_session(pool, sess, account_id=ACC)


async def test_create_run_clamps_top_edge_to_launcher(vault_pool: asyncpg.Pool[Any]) -> None:
    """The top edge: an agent-launched run is clamped to the launcher's surface; the
    operator path snapshots the workflow verbatim."""
    pool = vault_pool
    agent = await _make_agent(pool, "narrow-launcher", tools=[ToolSpec(type="bash")])
    launcher = await _make_session(pool, agent)
    # Operator authors a BROADER workflow (the agent couldn't, but the operator can).
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="wf-broad-run",
        script=_SCRIPT,
        tools=[ToolSpec(type="bash"), ToolSpec(type="read")],
    )
    launched = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV, launcher_session_id=launcher
    )
    assert {t.type for t in launched.tools} == {"bash"}  # read clamped away

    operator = await wf_service.create_run(
        pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV
    )
    assert {t.type for t in operator.tools} == {"bash", "read"}  # operator = top — verbatim


async def test_create_run_clamps_to_launcher_revoked_mid_launch(
    vault_pool: asyncpg.Pool[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """#835: the launcher surface is read INSIDE the run txn, not before it.

    A live-read launcher (``agent_version=None``) holds ``[bash, read]``. The operator
    authors a broad ``[bash, read]`` workflow. We inject an ``update_agent`` that revokes
    ``read`` at the txn boundary — patching ``get_environment`` (the first in-txn query)
    to commit the revoke before calling through. The snapshot must clamp to the
    post-revoke surface ``{bash}``.

    Discriminates the bug: pre-fix, ``load_for_session`` runs BEFORE the txn and reads the
    stale-broad ``[bash, read]`` (the revoke commits too late) → snapshot ``{bash, read}``.
    Post-fix, the in-txn read happens AFTER the revoke barrier → snapshot ``{bash}``.
    """
    pool = vault_pool
    agent = await _make_agent(
        pool, "revoked-launcher", tools=[ToolSpec(type="bash"), ToolSpec(type="read")]
    )
    # Live/latest-read launcher session (agent_version=None) — the only kind with a live
    # surface read to race; a pinned/frozen session would read an immutable version.
    async with pool.acquire() as conn:
        launcher_session = await db_queries.insert_session(
            conn,
            account_id=ACC,
            agent_id=agent.id,
            environment_id=ENV,
            agent_version=None,
            title=None,
            metadata={},
        )
    launcher = launcher_session.id
    # Operator authors a BROAD workflow (the run will clamp against the launcher).
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="wf-revoke-mid",
        script=_SCRIPT,
        tools=[ToolSpec(type="bash"), ToolSpec(type="read")],
    )

    # The real in-txn env read (imported from aios.db.queries; we patch the name bound
    # in the service module, so capture the original from its source here).
    real_get_environment = db_queries.get_environment
    revoked = False

    async def revoking_get_environment(conn: Any, environment_id: str, *, account_id: str) -> Any:
        nonlocal revoked
        if not revoked:
            revoked = True
            # Commit the revoke on a fresh pool connection (its own txn), simulating a
            # concurrent operator edit landing at the txn boundary.
            await agents_service.update_agent(
                pool, agent.id, account_id=ACC, expected_version=1, tools=[ToolSpec(type="bash")]
            )
        return await real_get_environment(conn, environment_id, account_id=account_id)

    monkeypatch.setattr(wf_service_module, "get_environment", revoking_get_environment)

    run = await wf_service.create_run(
        pool,
        account_id=ACC,
        workflow_id=wf.id,
        environment_id=ENV,
        launcher_session_id=launcher,
    )
    assert {t.type for t in run.tools} == {"bash"}  # clamped to the post-revoke surface


async def test_subrun_composes_against_child_frozen_clamp(
    vault_pool: asyncpg.Pool[Any],
) -> None:
    """A sub-run launched by a child cannot exceed the child's frozen surface —
    composition with no ancestor walk: the child's load_for_session returns its clamp,
    and create_run meets the sub-workflow against it."""
    pool = vault_pool
    agent = await _make_agent(pool, "nesting-agent", tools=[ToolSpec(type="bash")])
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="wf-parent", script=_SCRIPT, tools=[ToolSpec(type="bash")]
    )
    run = await wf_service.create_run(pool, account_id=ACC, workflow_id=wf.id, environment_id=ENV)
    child = await _spawn_child(
        pool, run.id, agent, surface=attenuation_service.clamp(surface_of(agent), surface_of(run))
    )

    # The child launches a sub-run of a broader operator workflow; it clamps to [bash].
    sub_wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="wf-sub-broad",
        script=_SCRIPT,
        tools=[ToolSpec(type="bash"), ToolSpec(type="read")],
    )
    subrun = await wf_service.create_run(
        pool,
        account_id=ACC,
        workflow_id=sub_wf.id,
        environment_id=ENV,
        launcher_session_id=child.id,
        parent_run_id=run.id,
    )
    assert {t.type for t in subrun.tools} == {"bash"}  # composed clamp: read dropped
