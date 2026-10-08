"""Run visibility (#2468): a model-dispatch run is private to its session.

A workflow-as-model run's input is the bound session's full request, so the
agent run-read tools (``get_run`` / ``list_runs`` / ``list_run_events`` /
``archive_run``) and their run-side twins must not show it to any other session
in the account. Its sub-runs inherit the same visibility. The operator API still
sees everything.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest
from pydantic import ValidationError

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ForbiddenError, NotFoundError
from aios.harness import runtime
from aios.models.agents import AgentCreate, AgentUpdate, ToolSpec
from aios.models.triggers import TriggerCreate
from aios.models.workflows import (
    OperatorAuthority,
    RunAuthority,
    RunReader,
    SessionAuthority,
    WfRun,
)
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.services import workflows as wf_service
from aios.tools import workflow_management as tools
from aios.workflows import run_tools, service
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_SECRET = "the bound session's private conversation"
_names = itertools.count()


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_vis', NULL, TRUE, 'vis')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_vis', 'vis-env', '{}'::jsonb, 'acc_vis')"
            )
        with mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()):
            yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _session(pool: asyncpg.Pool[Any], name: str) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_vis",
        name=name,
        model="test/dummy",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    session = await sessions_service.create_session(
        pool,
        account_id="acc_vis",
        agent_id=agent.id,
        environment_id="env_vis",
        title=None,
        metadata={},
    )
    return session.id


async def _workflow(pool: asyncpg.Pool[Any]) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_vis",
            name=f"recipe-{next(_names)}",
            script="async def main(input):\n    return 1\n",
        )
    return wf.id


async def _dispatch_run(pool: asyncpg.Pool[Any], session_id: str) -> WfRun:
    """The run a WaM park launches: its caller edge marks it ``model_dispatch``."""
    return await service.create_run(
        pool,
        account_id="acc_vis",
        authority=SessionAuthority(session_id, None),
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
        input={"messages": [{"role": "user", "content": _SECRET}]},
        caller={"kind": "session", "id": session_id, "purpose": "model_dispatch"},
    )


async def test_another_session_cannot_read_a_model_dispatch_run(
    pool: asyncpg.Pool[Any],
) -> None:
    owner = await _session(pool, "owner")
    other = await _session(pool, "other")
    run = await _dispatch_run(pool, owner)
    assert run.visibility == "session"

    for handler in (
        tools.get_run_handler,
        tools.list_run_events_handler,
        tools.archive_run_handler,
    ):
        with pytest.raises(NotFoundError):
            await handler(other, {"run_id": run.id})
    listed = await tools.list_runs_handler(other, {"account_wide": True})
    assert run.id not in {r["id"] for r in listed["runs"]}


async def test_the_owning_session_and_the_operator_still_read_it(
    pool: asyncpg.Pool[Any],
) -> None:
    owner = await _session(pool, "owner")
    run = await _dispatch_run(pool, owner)

    got = await tools.get_run_handler(owner, {"run_id": run.id})
    assert got["input"]["messages"][0]["content"] == _SECRET
    listed = await tools.list_runs_handler(owner, {"account_wide": True})
    assert run.id in {r["id"] for r in listed["runs"]}
    events = await tools.list_run_events_handler(owner, {"run_id": run.id})
    assert isinstance(events["events"], list)

    operator_read = await wf_service.get_run(pool, run.id, account_id="acc_vis", reader=None)
    assert operator_read.id == run.id


async def test_sub_runs_inherit_session_visibility(pool: asyncpg.Pool[Any]) -> None:
    owner = await _session(pool, "owner")
    other = await _session(pool, "other")
    parent = await _dispatch_run(pool, owner)
    sub = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=SessionAuthority(owner, parent.id),
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
        caller={"kind": "run", "id": parent.id},
    )
    assert sub.visibility == "session"
    with pytest.raises(NotFoundError):
        await tools.get_run_handler(other, {"run_id": sub.id})


async def test_a_run_cannot_read_another_sessions_model_dispatch_run(
    pool: asyncpg.Pool[Any],
) -> None:
    """The run-side ``get_run`` / ``list_runs`` apply the same rule, reading as the
    run's launching session (an operator run has none, so it sees only account runs)."""
    owner = await _session(pool, "owner")
    private = await _dispatch_run(pool, owner)
    reader = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
    )

    got = await run_tools._read_run_journal(
        run=reader, tool_name="get_run", args={"run_id": private.id}
    )
    assert "error" in got
    listed = await run_tools._read_run_journal(run=reader, tool_name="list_runs", args={})
    assert private.id not in {r["id"] for r in listed["runs"]}
    assert reader.id in {r["id"] for r in listed["runs"]}


async def test_only_the_owners_run_completion_trigger_fires_on_a_private_run(
    pool: asyncpg.Pool[Any],
) -> None:
    """A run_completion fire hands the completed run's output to the trigger owner by
    value, so a session-private run matches only its launching session's triggers."""
    owner = await _session(pool, "owner")
    other = await _session(pool, "other")
    run = await _dispatch_run(pool, owner)
    assert run.workflow_id is not None
    target = await _workflow(pool)
    trigger_ids: dict[str, str] = {}
    for session_id in (owner, other):
        echo = await triggers_service.add_trigger(
            pool,
            session_id,
            TriggerCreate.model_validate(
                {
                    "name": f"watch-{next(_names)}",
                    "source": {"kind": "run_completion", "workflow_id": run.workflow_id},
                    "action": {"kind": "workflow", "workflow_id": target},
                }
            ),
            account_id="acc_vis",
        )
        trigger_ids[session_id] = echo.id

    async with pool.acquire() as conn, conn.transaction():
        fires = await queries.insert_run_completion_fires(
            conn,
            account_id="acc_vis",
            workflow_id=run.workflow_id,
            run_id=run.id,
            status="completed",
            visibility=run.visibility,
            launcher_session_id=run.launcher_session_id,
        )
    assert [f.trigger_id for f in fires] == [trigger_ids[owner]]


async def test_resume_gate_on_another_sessions_private_run_404s(
    pool: asyncpg.Pool[Any],
) -> None:
    """The agent gate resume must not tell a private run apart from a missing one; an
    account-visible run another session launched still reads as forbidden."""
    owner = await _session(pool, "owner")
    other = await _session(pool, "other")
    private = await _dispatch_run(pool, owner)
    shared = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=SessionAuthority(owner, None),
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
    )
    with pytest.raises(NotFoundError):
        await tools.resume_gate_handler(other, {"run_id": private.id, "gate_nonce": "x"})
    with pytest.raises(ForbiddenError):
        await tools.resume_gate_handler(other, {"run_id": shared.id, "gate_nonce": "x"})


# ── #2475: the replay tools, which only an operator run can hold ──────────────

_REPLAY_TOOLS = [ToolSpec(type="sample_requests"), ToolSpec(type="get_request")]


async def _replay_workflow(pool: asyncpg.Pool[Any]) -> str:
    """An operator-authored workflow that declares the replay tools."""
    wf = await wf_service.create_workflow(
        pool,
        account_id="acc_vis",
        name=f"eval-{next(_names)}",
        script="async def main(input):\n    return 1\n",
        tools=_REPLAY_TOOLS,
    )
    return wf.id


def test_an_agent_cannot_declare_a_replay_tool() -> None:
    for spec in (
        {"name": "a", "model": "m", "tools": [{"type": "sample_requests"}]},
        {"name": "a", "model": "m", "tools": [{"type": "get_request"}]},
    ):
        with pytest.raises(ValidationError, match="only be declared by a workflow"):
            AgentCreate.model_validate(spec)
    with pytest.raises(ValidationError, match="only be declared by a workflow"):
        AgentUpdate.model_validate({"version": 1, "tools": [{"type": "get_request"}]})


async def test_an_agent_cannot_author_or_edit_a_replay_workflow(
    pool: asyncpg.Pool[Any],
) -> None:
    """No agent holds a replay tool, so the authoring gate refuses an agent that
    declares one, and an agent edit of a workflow that holds one, even a script-only
    edit."""
    agent_session = await _session(pool, "author")
    with pytest.raises(ForbiddenError):
        await wf_service.create_workflow(
            pool,
            account_id="acc_vis",
            name="agent-eval",
            script="async def main(input):\n    return 1\n",
            tools=_REPLAY_TOOLS,
            creator_session_id=agent_session,
        )
    wf_id = await _replay_workflow(pool)
    with pytest.raises(ForbiddenError):
        await wf_service.update_workflow(
            pool,
            wf_id,
            account_id="acc_vis",
            expected_version=1,
            script="async def main(input):\n    return 2\n",
            actor_session_id=agent_session,
        )


async def test_an_operator_run_holds_the_replay_tools_and_no_agent_sees_it(
    pool: asyncpg.Pool[Any],
) -> None:
    wf_id = await _replay_workflow(pool)
    run = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=wf_id,
        environment_id="env_vis",
    )
    assert {t.type for t in run.tools} == {"sample_requests", "get_request"}
    assert (run.principal, run.visibility, run.launcher_session_id) == (
        "operator",
        "session",
        None,
    )

    reader = await _session(pool, "reader")
    with pytest.raises(NotFoundError):
        await wf_service.get_run(pool, run.id, account_id="acc_vis", reader=RunReader(reader))
    listed = await wf_service.list_runs(pool, account_id="acc_vis", reader=RunReader(reader))
    assert run.id not in {r.id for r in listed}
    # A run-side reader with no launching session sees nothing private either.
    assert not RunReader(None).can_see(run)
    # The operator API still reads it.
    assert (await wf_service.get_run(pool, run.id, account_id="acc_vis", reader=None)).id == run.id


async def test_a_replay_runs_sub_runs_are_hidden_too(pool: asyncpg.Pool[Any]) -> None:
    parent = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=await _replay_workflow(pool),
        environment_id="env_vis",
    )
    sub = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=RunAuthority(parent.id),
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
    )
    assert (sub.visibility, sub.launcher_session_id) == ("session", None)
    reader = await _session(pool, "reader")
    with pytest.raises(NotFoundError):
        await wf_service.get_run(pool, sub.id, account_id="acc_vis", reader=RunReader(reader))


async def test_a_session_launch_drops_the_replay_tools(pool: asyncpg.Pool[Any]) -> None:
    """The clamp meets the workflow with the launching agent's surface, which can't
    hold a replay tool, so the run doesn't either."""
    launcher = await _session(pool, "launcher")
    wf_id = await _replay_workflow(pool)
    run = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=SessionAuthority(launcher, None),
        workflow_id=wf_id,
        environment_id="env_vis",
    )
    assert run.tools == []
    assert (run.principal, run.visibility) == ("session", "account")


def _run_with(run: WfRun, *, principal: str, tools: list[ToolSpec]) -> WfRun:
    return run.model_copy(update={"principal": principal, "tools": tools})


async def test_the_replay_tools_refuse_a_session_run_and_an_undeclared_run(
    pool: asyncpg.Pool[Any],
) -> None:
    run = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=await _replay_workflow(pool),
        environment_id="env_vis",
    )
    # A run that acts for a session can't call them even if it declared them (a run
    # like that can't be created; this is the principal half of the #794 position).
    forged = _run_with(run, principal="session", tools=_REPLAY_TOOLS)
    for tool_name in ("sample_requests", "get_request"):
        refusal = run_tools.gate_run_tool(forged, tool_name)
        assert refusal is not None and "operator run" in refusal["error"]
    undeclared = _run_with(run, principal="operator", tools=[])
    refusal = run_tools.gate_run_tool(undeclared, "sample_requests")
    assert refusal is not None and "declared tools" in refusal["error"]


async def test_a_generic_child_of_a_replay_run_holds_no_replay_tool(
    pool: asyncpg.Pool[Any],
) -> None:
    """A generic ``agent()`` child inherits its run's surface, minus the replay tools:
    no session holds one. Otherwise the child (an eval judge, say) would crash on its
    first step, and could author a workflow that declares them."""
    wf = await wf_service.create_workflow(
        pool,
        account_id="acc_vis",
        name=f"eval-judge-{next(_names)}",
        script="async def main(input):\n    return await agent('judge this', model='test/judge')\n",
        tools=[*_REPLAY_TOOLS, ToolSpec(type="read")],
    )
    run = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=wf.id,
        environment_id="env_vis",
    )
    with (
        mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
        mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
    ):
        await run_workflow_step(run.id)

    async with pool.acquire() as conn:
        child_id = await conn.fetchval("SELECT id FROM sessions WHERE parent_run_id = $1", run.id)
        frozen = await queries.get_session_frozen_surface(conn, child_id, account_id="acc_vis")
    assert frozen is not None
    assert [t.type for t in frozen.tools] == ["read"]
    with pytest.raises(ForbiddenError):
        await wf_service.create_workflow(
            pool,
            account_id="acc_vis",
            name=f"laundered-{next(_names)}",
            script="async def main(input):\n    return 1\n",
            tools=_REPLAY_TOOLS,
            creator_session_id=child_id,
        )


# ── #2475 gap tests: every launch path keeps the replay tools only on an operator run ─


async def test_an_as_agent_sub_run_of_a_replay_run_drops_the_replay_tools(
    pool: asyncpg.Pool[Any],
) -> None:
    """``as_agent`` re-roots a sub-run at an agent's surface within the parent's. No
    agent holds a replay tool, so the sub-run drops them even though both the parent
    and the sub-workflow declare them, and it stays operator-principal and private."""
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_vis",
        name=f"arm-{next(_names)}",
        model="test/dummy",
        system="",
        tools=[ToolSpec(type="read")],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    child = await wf_service.create_workflow(
        pool,
        account_id="acc_vis",
        name=f"eval-arm-{next(_names)}",
        script="async def main(input):\n    return 1\n",
        tools=[*_REPLAY_TOOLS, ToolSpec(type="read")],
    )
    parent_wf = await wf_service.create_workflow(
        pool,
        account_id="acc_vis",
        name=f"eval-driver-{next(_names)}",
        script=(
            "async def main(input):\n"
            "    return await invoke_workflow(input['wf'], {}, as_agent=input['as_agent'])\n"
        ),
        tools=[*_REPLAY_TOOLS, ToolSpec(type="read")],
    )
    parent = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        workflow_id=parent_wf.id,
        environment_id="env_vis",
        input={"wf": child.id, "as_agent": {"agent_id": agent.id, "version": 1}},
    )
    assert {t.type for t in parent.tools} >= {"sample_requests", "get_request"}
    with mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()):
        await run_workflow_step(parent.id)

    async with pool.acquire() as conn:
        sub_id = await conn.fetchval("SELECT id FROM wf_runs WHERE parent_run_id = $1", parent.id)
        assert sub_id is not None
        sub = await wf_queries.get_run_for_step(conn, sub_id)
    assert sub is not None
    assert [t.type for t in sub.tools] == ["read"]
    assert (sub.principal, sub.visibility, sub.launcher_session_id) == (
        "agent",
        "session",
        None,
    )


@pytest.mark.parametrize("replay_tool", ["sample_requests", "get_request"])
async def test_an_inline_call_workflow_from_a_session_cannot_declare_a_replay_tool(
    pool: asyncpg.Pool[Any], replay_tool: str
) -> None:
    """An inline run's surface must not exceed its launching agent's, and no agent
    holds a replay tool: the launch is refused, not silently narrowed."""
    launcher = await _session(pool, "inline-launcher")
    with pytest.raises(ForbiddenError):
        await wf_service.launch_awaited_run(
            pool,
            account_id="acc_vis",
            inline=service.InlineScript(
                script="async def main(input):\n    return 1\n",
                tools=[ToolSpec(type=replay_tool)],
            ),
            environment_id="env_vis",
            caller={"kind": "session", "id": launcher},
            authority=SessionAuthority(launcher, None),
        )
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM wf_runs") == 0


async def test_an_operator_inline_run_declaring_only_get_request_is_private(
    pool: asyncpg.Pool[Any],
) -> None:
    """Either replay tool alone makes the run private (the 0191 arm has a disjunct per
    tool)."""
    run = await service.create_run(
        pool,
        account_id="acc_vis",
        authority=OperatorAuthority(),
        inline=service.InlineScript(
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="get_request")],
        ),
        environment_id="env_vis",
    )
    assert [t.type for t in run.tools] == ["get_request"]
    assert (run.principal, run.visibility, run.launcher_session_id) == (
        "operator",
        "session",
        None,
    )


async def test_the_tasks_api_workflow_arm_keeps_the_replay_tools_and_is_private(
    pool: asyncpg.Pool[Any],
) -> None:
    """``POST /v1/tasks`` with ``target_kind=workflow`` launches under the operator."""
    wf_id = await _replay_workflow(pool)
    handle = await sessions_service.invoke(
        pool,
        account_id="acc_vis",
        target_kind="workflow",
        target=wf_id,
        input={},
        environment_id="env_vis",
    )
    assert handle.servicer_kind == "run"
    run = await wf_service.get_run(pool, handle.servicer_id, account_id="acc_vis", reader=None)
    assert {t.type for t in run.tools} == {"sample_requests", "get_request"}
    assert (run.principal, run.visibility, run.launcher_session_id) == (
        "operator",
        "session",
        None,
    )
    reader = await _session(pool, "tasks-reader")
    with pytest.raises(NotFoundError):
        await wf_service.get_run(pool, run.id, account_id="acc_vis", reader=RunReader(reader))
