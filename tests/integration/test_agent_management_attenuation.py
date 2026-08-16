"""Integration coverage for create-time agent surface attenuation (T1, #1470).

Drives the **service path** (``aios.services.agents.create_agent`` /
``update_agent``) against a real Postgres, asserting the acceptance criteria:

* an agent authored by a session with surface S **cannot** declare a surface ⊄ S
  (``ForbiddenError``, ``detail.exceeds``) — nor may an update widen past the editor;
* an authored agent with surface ⊆ S **persists**;
* the create-time clamp is independent of the #794/#823 spawn-edge reclamp — the
  spawn edge still re-clamps ``agent ∩ run`` at spawn (asserted via the pure
  ``attenuation`` operator the spawn edge calls, against the persisted authored agent).

The tool-handler path (``create_agent_handler`` in ``tools.agent_management``) is a
thin wrapper that calls the same service function with ``creator_session_id`` =
the executing session, so enforcing here enforces there too; the F1/F2 invariants and
return shapes are covered without a DB in ``tests/unit/test_agent_management.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.errors import ForbiddenError
from aios.harness import runtime
from aios.models.agents import Agent, HttpServerSpec, McpServerSpec, ToolSpec
from aios.models.attenuation import surface_of
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service

pytestmark = pytest.mark.integration

ACC = "acc_agm"
ENV = "env_agm"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = p
    try:
        async with p.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_agm', NULL, TRUE, 'agm-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_agm', 'agm-env', '{}'::jsonb, 'acc_agm')"
            )
        yield p
    finally:
        runtime.pool = prev
        await p.close()


async def _make_agent(
    pool: asyncpg.Pool[Any], name: str, *, tools: list[ToolSpec] | None = None
) -> Agent:
    return await agents_service.create_agent(
        pool,
        account_id=ACC,
        name=name,
        model="test/dummy",
        system="x",
        tools=tools or [],
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )


async def _make_session(pool: asyncpg.Pool[Any], agent: Agent) -> str:
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
    return session.id


# ─── create-time clamp: authored surface ⊆ creator ───────────────────────────


async def test_create_rejects_surface_exceeding_creator(pool: asyncpg.Pool[Any]) -> None:
    # Creator agent holds only `read`. Authoring a child with `read`+`write` exceeds it.
    creator = await _make_agent(pool, "creator", tools=[ToolSpec(type="read")])
    session_id = await _make_session(pool, creator)
    with pytest.raises(ForbiddenError) as exc:
        await agents_service.create_agent(
            pool,
            account_id=ACC,
            name="child-too-broad",
            model="test/dummy",
            system="x",
            tools=[ToolSpec(type="read"), ToolSpec(type="write")],
            mcp_servers=None,
            http_servers=None,
            description=None,
            metadata={},
            window_min=1000,
            window_max=100000,
            creator_session_id=session_id,
        )
    assert exc.value.detail is not None
    assert "write" in exc.value.detail["exceeds"]["tools"]
    # Not created.
    assert (await agents_service.list_agents(pool, account_id=ACC, name="child-too-broad")) == []


async def test_create_accepts_surface_subset_of_creator(pool: asyncpg.Pool[Any]) -> None:
    creator = await _make_agent(
        pool, "creator2", tools=[ToolSpec(type="read"), ToolSpec(type="write")]
    )
    session_id = await _make_session(pool, creator)
    child = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="child-ok",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="read")],  # ⊆ creator's {read, write}
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
        creator_session_id=session_id,
    )
    assert child.id
    stored = await agents_service.get_agent(pool, child.id, account_id=ACC)
    assert {t.type for t in stored.tools} == {"read"}


async def test_create_equal_surface_persists(pool: asyncpg.Pool[Any]) -> None:
    creator = await _make_agent(pool, "creator3", tools=[ToolSpec(type="bash")])
    session_id = await _make_session(pool, creator)
    child = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="child-equal",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="bash")],
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
        creator_session_id=session_id,
    )
    assert {t.type for t in child.tools} == {"bash"}


async def test_operator_path_skips_clamp(pool: asyncpg.Pool[Any]) -> None:
    # No creator_session_id (the HTTP/operator path): any surface may be declared.
    agent = await _make_agent(
        pool, "operator-made", tools=[ToolSpec(type="read"), ToolSpec(type="write")]
    )
    assert {t.type for t in agent.tools} == {"read", "write"}


# ─── update-time clamp: merged surface ⊆ editor ──────────────────────────────


async def test_update_rejects_widening_past_editor(pool: asyncpg.Pool[Any]) -> None:
    editor = await _make_agent(pool, "editor", tools=[ToolSpec(type="read")])
    session_id = await _make_session(pool, editor)
    # A target agent the editor authored within its surface.
    target = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="target",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="read")],
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
        creator_session_id=session_id,
    )
    with pytest.raises(ForbiddenError) as exc:
        await agents_service.update_agent(
            pool,
            target.id,
            account_id=ACC,
            expected_version=target.version,
            tools=[ToolSpec(type="read"), ToolSpec(type="write")],  # ⊄ editor's {read}
            editor_session_id=session_id,
        )
    assert exc.value.detail is not None
    assert "write" in exc.value.detail["exceeds"]["tools"]
    # Unchanged (still at its original version, still read-only).
    after = await agents_service.get_agent(pool, target.id, account_id=ACC)
    assert after.version == target.version and {t.type for t in after.tools} == {"read"}


async def test_update_preserves_target_surface_the_editor_lacks(
    pool: asyncpg.Pool[Any],
) -> None:
    editor = await _make_agent(pool, "delta-editor", tools=[ToolSpec(type="read")])
    session_id = await _make_session(pool, editor)
    target = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="delta-target",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="write")],
        mcp_servers=[McpServerSpec(name="target-mcp", url="https://mcp.example.test")],
        http_servers=[HttpServerSpec(name="target-http", base_url="https://api.example.test")],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )

    updated = await agents_service.update_agent(
        pool,
        target.id,
        account_id=ACC,
        expected_version=target.version,
        window_max=120000,
        editor_session_id=session_id,
    )

    assert updated.window_max == 120000
    assert {t.type for t in updated.tools} == {"write"}
    assert [server.name for server in updated.mcp_servers] == ["target-mcp"]
    assert [server.name for server in updated.http_servers] == ["target-http"]


async def test_update_allows_shrinking_surface_the_editor_lacks(
    pool: asyncpg.Pool[Any],
) -> None:
    editor = await _make_agent(pool, "shrink-editor", tools=[])
    session_id = await _make_session(pool, editor)
    target = await _make_agent(
        pool, "shrink-target", tools=[ToolSpec(type="read"), ToolSpec(type="write")]
    )

    updated = await agents_service.update_agent(
        pool,
        target.id,
        account_id=ACC,
        expected_version=target.version,
        tools=[ToolSpec(type="read")],
        editor_session_id=session_id,
    )

    assert {t.type for t in updated.tools} == {"read"}


async def test_update_within_editor_surface_persists(pool: asyncpg.Pool[Any]) -> None:
    editor = await _make_agent(
        pool, "editor2", tools=[ToolSpec(type="read"), ToolSpec(type="write")]
    )
    session_id = await _make_session(pool, editor)
    target = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="target2",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="read")],
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
        creator_session_id=session_id,
    )
    updated = await agents_service.update_agent(
        pool,
        target.id,
        account_id=ACC,
        expected_version=target.version,
        tools=[ToolSpec(type="read"), ToolSpec(type="write")],  # ⊆ editor's {read, write}
        editor_session_id=session_id,
    )
    assert {t.type for t in updated.tools} == {"read", "write"}
    assert updated.version == target.version + 1


# ─── independence from the #794/#823 spawn-edge reclamp ──────────────────────


async def test_spawn_edge_reclamp_is_independent_and_unchanged(pool: asyncpg.Pool[Any]) -> None:
    # An agent authored with surface ⊆ creator can still hold a tool a *narrower* run
    # lacks; the spawn-edge clamp (agent ∩ run, #794) — the same pure operator the
    # spawn path calls — strips it at spawn, independent of the create-time clamp.
    creator = await _make_agent(
        pool, "spawn-creator", tools=[ToolSpec(type="read"), ToolSpec(type="write")]
    )
    session_id = await _make_session(pool, creator)
    child = await agents_service.create_agent(
        pool,
        account_id=ACC,
        name="spawn-child",
        model="test/dummy",
        system="x",
        tools=[ToolSpec(type="read"), ToolSpec(type="write")],  # ⊆ creator → persists
        mcp_servers=None,
        http_servers=None,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
        creator_session_id=session_id,
    )
    # A run with only `read` — the spawn edge computes clamp(agent ∩ run).
    run_surface = surface_of(await _make_agent(pool, "spawn-run", tools=[ToolSpec(type="read")]))
    effective = attenuation_service.clamp(surface_of(child), run_surface)
    # `write` survives create-time (⊆ creator) but is stripped at spawn (⊄ run).
    assert {t.type for t in effective.tools} == {"read"}
