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

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ForbiddenError, NotFoundError
from aios.harness import runtime
from aios.models.triggers import TriggerCreate
from aios.models.workflows import WfRun
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.services import workflows as wf_service
from aios.tools import workflow_management as tools
from aios.workflows import run_tools, service

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
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
        input={"messages": [{"role": "user", "content": _SECRET}]},
        launcher_session_id=session_id,
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
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
        launcher_session_id=owner,
        parent_run_id=parent.id,
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
        pool, account_id="acc_vis", workflow_id=await _workflow(pool), environment_id="env_vis"
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
        workflow_id=await _workflow(pool),
        environment_id="env_vis",
        launcher_session_id=owner,
    )
    with pytest.raises(NotFoundError):
        await tools.resume_gate_handler(other, {"run_id": private.id, "gate_nonce": "x"})
    with pytest.raises(ForbiddenError):
        await tools.resume_gate_handler(other, {"run_id": shared.id, "gate_nonce": "x"})
