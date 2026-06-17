"""Integration coverage for create-time workflow-script validation (#1285).

Drives the **service path** (``aios.services.workflows.create_workflow`` /
``update_workflow``) against a real Postgres, asserting the acceptance criteria hold
at create AND update time, on both the operator path (no actor) and the agent-author
path (criterion 5, the named-``agent(agent_id=…)`` surface union — which needs a live
agent to resolve, so it cannot live in the pure unit tests).

The tool-handler path (``create_workflow_handler`` in ``tools.workflow_management``)
is a thin wrapper that calls this same service function, so enforcing here enforces
there too (criterion 8); the pure structural criteria (1-4, 6, 7) are additionally
covered without a DB in ``tests/unit/test_workflow_script_validation.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.errors import ValidationError
from aios.harness import runtime
from aios.models.agents import Agent, ToolSpec
from aios.services import agents as agents_service
from aios.services import workflows as wf_service

pytestmark = pytest.mark.integration

ACC = "acc_wfv"
ENV = "env_wfv"

_VALID = "async def main(input):\n    return input\n"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = p
    try:
        async with p.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_wfv', NULL, TRUE, 'wfv-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_wfv', 'wfv-env', '{}'::jsonb, 'acc_wfv')"
            )
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
        ):
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


# ─── create: criteria 1-4, 6, 7 on the operator path ─────────────────────────


async def test_create_rejects_syntax_error(pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError) as exc:
        await wf_service.create_workflow(
            pool, account_id=ACC, name="bad", script="async def main(input):\n    return (\n"
        )
    assert "compile" in str(exc.value)
    # Not created.
    assert (await wf_service.list_workflows(pool, account_id=ACC)) == []


async def test_create_rejects_missing_main(pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError) as exc:
        await wf_service.create_workflow(pool, account_id=ACC, name="nm", script="x = 1\n")
    assert "main" in str(exc.value)
    assert (await wf_service.list_workflows(pool, account_id=ACC)) == []


async def test_create_rejects_plain_def_main(pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError):
        await wf_service.create_workflow(
            pool, account_id=ACC, name="pd", script="def main(input):\n    return 1\n"
        )
    assert (await wf_service.list_workflows(pool, account_id=ACC)) == []


async def test_create_rejects_bad_arity_main(pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError):
        await wf_service.create_workflow(
            pool, account_id=ACC, name="ar", script="async def main(a, b):\n    return 1\n"
        )
    assert (await wf_service.list_workflows(pool, account_id=ACC)) == []


async def test_create_rejects_under_declared_tool(pool: asyncpg.Pool[Any]) -> None:
    script = "async def main(input):\n    return await tool('bash', {'command': 'ls'})\n"
    with pytest.raises(ValidationError) as exc:
        await wf_service.create_workflow(pool, account_id=ACC, name="ud", script=script)
    assert "bash" in str(exc.value)
    assert (await wf_service.list_workflows(pool, account_id=ACC)) == []


async def test_create_accepts_fully_covered(pool: asyncpg.Pool[Any]) -> None:
    script = "async def main(input):\n    return await tool('bash', {'command': 'ls'})\n"
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="ok", script=script, tools=[ToolSpec(type="bash")]
    )
    assert wf.id
    assert wf.script == script
    assert {t.type for t in wf.tools} == {"bash"}


async def test_create_accepts_unastable_tool_name(pool: asyncpg.Pool[Any]) -> None:
    # A computed tool name is excluded from the required-surface check — accepted even
    # with an empty tool surface.
    script = "async def main(input):\n    name = input['tool']\n    return await tool(name, {})\n"
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="dyn", script=script)
    assert wf.id


# ─── criterion 5: named-agent surface union (needs a live agent) ─────────────


async def test_create_rejects_under_declared_agent_surface(pool: asyncpg.Pool[Any]) -> None:
    # A child agent that declares `read`+`write`; the workflow declares only `read`.
    # The #794 clamp (agent ∩ run) would silently strip `write` from the child -> reject.
    child = await _make_agent(pool, "child", tools=[ToolSpec(type="read"), ToolSpec(type="write")])
    agent_id = child.id
    script = f"async def main(input):\n    return await agent({{'x': 1}}, agent_id={agent_id!r})\n"
    with pytest.raises(ValidationError) as exc:
        await wf_service.create_workflow(
            pool, account_id=ACC, name="ag", script=script, tools=[ToolSpec(type="read")]
        )
    assert "write" in str(exc.value)
    assert exc.value.detail is not None and "write" in exc.value.detail["missing_tools"]


async def test_create_accepts_covered_agent_surface(pool: asyncpg.Pool[Any]) -> None:
    child = await _make_agent(pool, "child2", tools=[ToolSpec(type="read"), ToolSpec(type="write")])
    agent_id = child.id
    script = f"async def main(input):\n    return await agent({{'x': 1}}, agent_id={agent_id!r})\n"
    wf = await wf_service.create_workflow(
        pool,
        account_id=ACC,
        name="ag-ok",
        script=script,
        tools=[ToolSpec(type="read"), ToolSpec(type="write")],
    )
    assert wf.id


async def test_create_accepts_unknown_agent_id(pool: asyncpg.Pool[Any]) -> None:
    # A literal agent_id that does not resolve to a live same-account agent is skipped
    # (it fails loud at run time as agent_not_found — a different failure class).
    script = "async def main(input):\n    return await agent({'x': 1}, agent_id='does-not-exist')\n"
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="ghost", script=script)
    assert wf.id


# ─── agent-author path enforces the same checks (criterion 8) ────────────────


async def test_agent_author_path_enforces_validation(pool: asyncpg.Pool[Any]) -> None:
    agent = await _make_agent(pool, "author", tools=[ToolSpec(type="bash")])
    session_id = await _make_session(pool, agent)
    bad = "async def main(input):\n    return (\n"
    with pytest.raises(ValidationError):
        await wf_service.create_workflow(
            pool,
            account_id=ACC,
            name="auth",
            script=bad,
            tools=[ToolSpec(type="bash")],
            creator_session_id=session_id,
        )


# ─── update enforces the same checks (criteria 1-7 on update) ────────────────


async def test_update_rejects_syntax_error(pool: asyncpg.Pool[Any]) -> None:
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="u1", script=_VALID)
    with pytest.raises(ValidationError) as exc:
        await wf_service.update_workflow(
            pool,
            wf.id,
            account_id=ACC,
            expected_version=wf.version,
            script="async def main(input):\n    return (\n",
        )
    assert "compile" in str(exc.value)
    # The stored definition is unchanged (still at its original version).
    after = await wf_service.get_workflow(pool, wf.id, account_id=ACC)
    assert after.version == wf.version and after.script == _VALID


async def test_update_rejects_under_declared_tool_against_merged_surface(
    pool: asyncpg.Pool[Any],
) -> None:
    # Created with bash declared; an update that introduces a tool('http_request') call
    # without adding it to tools is rejected against the merged surface.
    wf = await wf_service.create_workflow(
        pool, account_id=ACC, name="u2", script=_VALID, tools=[ToolSpec(type="bash")]
    )
    new_script = "async def main(input):\n    return await tool('http_request', {})\n"
    with pytest.raises(ValidationError) as exc:
        await wf_service.update_workflow(
            pool, wf.id, account_id=ACC, expected_version=wf.version, script=new_script
        )
    assert "http_request" in str(exc.value)


async def test_update_without_script_change_is_not_revalidated(pool: asyncpg.Pool[Any]) -> None:
    # An update that does not touch the script preserves the (already-validated) body and
    # is not re-checked — a description-only edit succeeds.
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="u3", script=_VALID)
    updated = await wf_service.update_workflow(
        pool, wf.id, account_id=ACC, expected_version=wf.version, description="touched"
    )
    assert updated.description == "touched" and updated.version == wf.version + 1


async def test_update_accepts_valid_new_script(pool: asyncpg.Pool[Any]) -> None:
    wf = await wf_service.create_workflow(pool, account_id=ACC, name="u4", script=_VALID)
    new_script = "async def main(input):\n    return await tool('bash', {'command': 'ls'})\n"
    updated = await wf_service.update_workflow(
        pool,
        wf.id,
        account_id=ACC,
        expected_version=wf.version,
        script=new_script,
        tools=[ToolSpec(type="bash")],
    )
    assert updated.script == new_script and updated.version == wf.version + 1
