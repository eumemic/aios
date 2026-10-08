"""Nested run budgets (#2476): a sub-run with no budget of its own is held to its
nearest budgeted ancestor's ceiling, ``invoke_workflow`` is refused once the budget
is spent, and ``invoke_workflow(budget_usd=)`` gives a sub-run its own ceiling, clamped
to what its caller may still spend.

Spend is seeded straight onto a run's ``call_llm`` meter, which the subtree rollup
reads, so the tests control exactly what each budget run has spent.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.workflows import OperatorAuthority, RunAuthority, SessionAuthority, WfRun
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.workflows import run_tools, service
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_ACC = "acc_budget"
_ENV = "env_budget"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    run_tools._INFLIGHT.clear()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'budget')",
                _ACC,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'budget-env', '{}'::jsonb, $2)",
                _ENV,
                _ACC,
            )
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _workflow(pool: asyncpg.Pool[Any], name: str, script: str) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id=_ACC, name=name, script=script)
    return wf.id


async def _root(
    pool: asyncpg.Pool[Any], workflow_id: str, *, input: Any, budget_usd: float | None
) -> str:
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=OperatorAuthority(),
        workflow_id=workflow_id,
        environment_id=_ENV,
        input=input,
        budget_usd=budget_usd,
    )
    return run.id


async def _run(pool: asyncpg.Pool[Any], run_id: str) -> WfRun:
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None
    return run


async def _spend(pool: asyncpg.Pool[Any], run_id: str, usd: float) -> None:
    async with pool.acquire() as conn:
        await wf_queries.add_run_call_llm_cost_microusd(
            conn, run_id, round(usd * 1_000_000), account_id=_ACC, model="m"
        )


async def _sub_run(pool: asyncpg.Pool[Any], parent_id: str) -> str:
    async with pool.acquire() as conn:
        sub_id: str = await conn.fetchval(
            "SELECT id FROM wf_runs WHERE parent_run_id = $1", parent_id
        )
    return sub_id


# A run that hands its input on to a sub-run of ``input['wf']`` and returns its answer.
_HAND_ON = (
    "async def main(input):\n"
    "    kwargs = {} if input.get('budget') is None else {'budget_usd': input['budget']}\n"
    "    return await invoke_workflow(input['wf'], input.get('next'), **kwargs)\n"
)

# A leaf that reads its budget, waits at a gate (so the test can spend meanwhile), then
# tries all three spending calls and reports how each went.
_LEAF = (
    "async def main(input):\n"
    "    before = await budget()\n"
    "    await gate('spend')\n"
    "    llm = await call_llm({'model': 'm', 'messages': [{'role': 'user', 'content': 'x'}]})\n"
    "    try:\n"
    "        await invoke_workflow(input['wf'], None)\n"
    "        sub = 'ran'\n"
    "    except AgentError as e:\n"
    "        sub = e.kind\n"
    "    return {'before': before, 'after': await budget(), 'llm': llm, 'sub': sub}\n"
)


async def _resume_gate(pool: asyncpg.Pool[Any], run_id: str) -> None:
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    gate = next(e for e in events if e.type == "call_started" and e.payload["capability"] == "gate")
    assert gate.call_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate.call_key, result=None)


async def test_a_sub_run_without_a_budget_is_held_to_its_ancestors(
    pool: asyncpg.Pool[Any],
) -> None:
    """Root ($1) → middle (no budget) → leaf (no budget): the leaf inherits the root's
    ceiling through the middle, sees it in ``budget()``, and once the root's subtree has
    spent it, its ``call_llm`` and ``invoke_workflow`` are refused."""
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    leaf_wf = await _workflow(pool, "leaf", _LEAF)
    middle_wf = await _workflow(pool, "middle", _HAND_ON)
    root_wf = await _workflow(pool, "root", _HAND_ON)
    root = await _root(
        pool,
        root_wf,
        input={"wf": middle_wf, "next": {"wf": leaf_wf, "next": {"wf": noop}}},
        budget_usd=1.0,
    )
    await run_workflow_step(root)
    middle = await _sub_run(pool, root)
    await run_workflow_step(middle)
    leaf = await _sub_run(pool, middle)
    for _ in range(3):
        await run_workflow_step(leaf)
    assert (await _run(pool, middle)).budget_run_id == root
    leaf_row = await _run(pool, leaf)
    assert leaf_row.budget_usd is None and leaf_row.budget_run_id == root
    assert leaf_row.status == "suspended"  # at the gate

    await _spend(pool, root, 0.4)
    await _spend(pool, leaf, 0.6)  # the leaf's own spend counts in the root's subtree
    await _resume_gate(pool, leaf)
    with mock.patch("aios.workflows.step.run_llm.launch_call_llm_task") as launch:
        for _ in range(6):
            await run_workflow_step(leaf)
    launch.assert_not_called()
    done = await _run(pool, leaf)
    assert done.status == "completed", done.output
    assert done.output["before"] == {"total_usd": 1.0, "spent_usd": 0.0, "remaining_usd": 1.0}
    assert done.output["after"] == {"total_usd": 1.0, "spent_usd": 1.0, "remaining_usd": 0.0}
    assert "call_llm is refused" in done.output["llm"]["error"]
    assert done.output["sub"] == "budget_exceeded"
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM wf_runs WHERE parent_run_id = $1", leaf) == 0
        )


async def test_a_spent_run_cannot_invoke_a_workflow(pool: asyncpg.Pool[Any]) -> None:
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(
        pool,
        "caller",
        "async def main(input):\n"
        "    try:\n"
        "        return await invoke_workflow(input['wf'], None)\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n",
    )
    root = await _root(pool, caller, input={"wf": noop}, budget_usd=0.5)
    await _spend(pool, root, 0.5)
    for _ in range(3):
        await run_workflow_step(root)
    done = await _run(pool, root)
    assert done.output["kind"] == "budget_exceeded"
    assert "spent $0.50 of $0.50" in done.output["msg"]


@pytest.mark.parametrize(("asked", "granted"), [(10.0, 0.6), (0.25, 0.25)])
async def test_a_sub_runs_own_budget_is_clamped_to_what_its_caller_has_left(
    pool: asyncpg.Pool[Any], asked: float, granted: float
) -> None:
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": noop, "budget": asked}, budget_usd=1.0)
    await _spend(pool, root, 0.4)
    await run_workflow_step(root)
    sub = await _run(pool, await _sub_run(pool, root))
    assert sub.budget_usd == pytest.approx(granted)
    assert sub.budget_run_id == root


async def test_an_unbudgeted_caller_passes_the_asked_budget_through(
    pool: asyncpg.Pool[Any],
) -> None:
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": noop, "budget": 3.0}, budget_usd=None)
    await run_workflow_step(root)
    sub = await _run(pool, await _sub_run(pool, root))
    assert sub.budget_usd == 3.0
    assert sub.budget_run_id is None


async def test_a_non_positive_budget_is_an_author_error(pool: asyncpg.Pool[Any]) -> None:
    """The host refuses a non-positive budget before it reaches the wire (the worker
    re-checks the wire value too)."""
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(
        pool,
        "caller",
        "async def main(input):\n"
        "    try:\n"
        "        return await invoke_workflow(input['wf'], None, budget_usd=0)\n"
        "    except ValueError as e:\n"
        "        return {'raised': str(e)}\n",
    )
    root = await _root(pool, caller, input={"wf": noop}, budget_usd=None)
    for _ in range(2):
        await run_workflow_step(root)
    assert "budget_usd > 0" in (await _run(pool, root)).output["raised"]


@pytest.mark.parametrize("asked", [1e-7, 0.0000005, 1e13, sys.float_info.max])
@pytest.mark.parametrize("caller_budget", [None, 1.0])
async def test_a_budget_out_of_micro_usd_range_is_an_author_error(
    pool: asyncpg.Pool[Any], asked: float, caller_budget: float | None
) -> None:
    """A positive finite budget that rounds to 0 micro-USD, or past bigint, is
    refused as a catchable author error before the sub-run's row is written (#2535)."""
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(
        pool,
        "caller",
        "async def main(input):\n"
        "    try:\n"
        "        return await invoke_workflow(input['wf'], None, budget_usd=input['budget'])\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n",
    )
    root = await _root(pool, caller, input={"wf": noop, "budget": asked}, budget_usd=caller_budget)
    for _ in range(3):
        await run_workflow_step(root)
    done = await _run(pool, root)
    assert done.output["kind"] == "bad_invoke_workflow"
    assert "budget_usd" in done.output["msg"]
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM wf_runs WHERE parent_run_id = $1", root) == 0
        )


async def test_a_budget_of_one_micro_usd_still_spawns(pool: asyncpg.Pool[Any]) -> None:
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": noop, "budget": 0.000001}, budget_usd=None)
    await run_workflow_step(root)
    sub = await _run(pool, await _sub_run(pool, root))
    assert sub.budget_usd == pytest.approx(0.000001)


async def test_the_sweep_wakes_a_parked_sub_run_whose_ancestor_is_spent(
    pool: asyncpg.Pool[Any],
) -> None:
    """A sub-run parked on an ``agent()`` call with no budget of its own is woken once
    its budget run's subtree is spent, so the step can force-resolve the call."""
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": noop}, budget_usd=1.0)
    await run_workflow_step(root)
    sub = await _sub_run(pool, root)
    async with pool.acquire() as conn:
        # Park the sub-run on an open agent() call, as a real spawn would.
        await wf_queries.append_run_event(
            conn,
            account_id=_ACC,
            run_id=sub,
            type="call_started",
            call_key="k-agent",
            payload={"capability": "agent", "child_session_id": "ses_none"},
        )
        await conn.execute("UPDATE wf_runs SET status = 'suspended' WHERE id = $1", sub)
        assert sub not in await wf_queries.list_parked_run_ids_over_budget(conn)
    await _spend(pool, root, 1.0)
    async with pool.acquire() as conn:
        over = await wf_queries.list_parked_run_ids_over_budget(conn)
    assert sub in over
    assert root not in over  # parked on invoke_workflow, not agent()


_REPORT_BUDGET = "async def main(input):\n    return await budget()\n"


async def _child_session(pool: asyncpg.Pool[Any]) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACC,
        name="child",
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
        account_id=_ACC,
        agent_id=agent.id,
        environment_id=_ENV,
        title=None,
        metadata={},
    )
    return session.id


async def _direct_sub_run(pool: asyncpg.Pool[Any], parent_id: str, workflow_id: str) -> str:
    run = await service.create_run(
        pool,
        account_id=_ACC,
        authority=RunAuthority(parent_id, None),
        workflow_id=workflow_id,
        environment_id=_ENV,
    )
    return run.id


async def test_a_run_a_child_session_launches_is_held_to_the_lineage_budget(
    pool: asyncpg.Pool[Any],
) -> None:
    """A session a budgeted run spawned launches a run (a WaM turn, ``call_workflow``):
    the run's lineage parent names the budget it is held to, directly or through an
    unbudgeted sub-run."""
    report = await _workflow(pool, "report", _REPORT_BUDGET)
    root = await _root(pool, report, input=None, budget_usd=1.0)
    middle = await _direct_sub_run(pool, root, report)
    session_id = await _child_session(pool)
    for parent in (root, middle):
        launched = await service.create_run(
            pool,
            account_id=_ACC,
            authority=SessionAuthority(session_id, parent),
            workflow_id=report,
            environment_id=_ENV,
        )
        assert launched.principal == "session"
        assert launched.budget_run_id == root
    await _spend(pool, root, 0.3)
    for _ in range(3):
        await run_workflow_step(launched.id)
    done = await _run(pool, launched.id)
    assert done.output == {"total_usd": 1.0, "spent_usd": 0.3, "remaining_usd": 0.7}


async def test_a_sub_run_created_before_a_lost_call_started_is_reattached_over_budget(
    pool: asyncpg.Pool[Any],
) -> None:
    """A wake creates the sub-run (on its own connection) but rolls back before its
    ``call_started`` is journaled; the budget is then spent. The replay re-attaches the
    sub-run that exists instead of refusing the call and orphaning it."""
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": noop}, budget_usd=0.5)
    real_append = wf_queries.append_run_event

    async def _lose_invoke_call_started(*args: Any, **kwargs: Any) -> Any:
        if (
            kwargs.get("type") == "call_started"
            and (kwargs.get("payload") or {}).get("capability") == "invoke_workflow"
        ):
            raise RuntimeError("lost before commit")
        return await real_append(*args, **kwargs)

    with (
        mock.patch.object(wf_queries, "append_run_event", new=_lose_invoke_call_started),
        pytest.raises(RuntimeError, match="lost before commit"),
    ):
        await run_workflow_step(root)
    sub = await _sub_run(pool, root)
    await _spend(pool, root, 0.5)
    await run_workflow_step(root)
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, root)
    started = [e for e in events if e.type == "call_started"]
    assert [e.payload["child_run_id"] for e in started] == [sub]
    assert not [e for e in events if e.type == "call_result"]  # not refused


async def test_a_sub_run_with_its_own_budget_outlives_its_spent_ancestor(
    pool: asyncpg.Pool[Any],
) -> None:
    """A sub-run given its own budget checks only that budget, so its ancestor's spend
    doesn't stop it: the documented overshoot bound."""
    leaf = await _workflow(
        pool, "leaf", "async def main(input):\n    await gate('g')\n    return await budget()\n"
    )
    caller = await _workflow(pool, "caller", _HAND_ON)
    root = await _root(pool, caller, input={"wf": leaf, "budget": 0.25}, budget_usd=1.0)
    await run_workflow_step(root)
    sub = await _sub_run(pool, root)
    await run_workflow_step(sub)
    await _spend(pool, root, 1.0)
    await _resume_gate(pool, sub)
    for _ in range(3):
        await run_workflow_step(sub)
    done = await _run(pool, sub)
    assert done.output == {"total_usd": 0.25, "spent_usd": 0.0, "remaining_usd": 0.25}


async def test_the_sweep_wakes_every_sub_run_sharing_a_spent_budget_run(
    pool: asyncpg.Pool[Any],
) -> None:
    noop = await _workflow(pool, "noop", "async def main(input):\n    return 1\n")
    root = await _root(pool, noop, input=None, budget_usd=1.0)
    subs = [await _direct_sub_run(pool, root, noop) for _ in range(2)]
    async with pool.acquire() as conn:
        for i, sub in enumerate(subs):
            await wf_queries.append_run_event(
                conn,
                account_id=_ACC,
                run_id=sub,
                type="call_started",
                call_key=f"k-agent-{i}",
                payload={"capability": "agent", "child_session_id": "ses_none"},
            )
            await conn.execute("UPDATE wf_runs SET status = 'suspended' WHERE id = $1", sub)
    await _spend(pool, root, 1.0)
    async with pool.acquire() as conn:
        over = await wf_queries.list_parked_run_ids_over_budget(conn)
    assert set(subs) <= set(over)


async def test_a_sub_run_whose_budget_run_is_pruned_is_held_to_nothing(
    pool: asyncpg.Pool[Any],
) -> None:
    """The archive prune deletes run rows. A descendant still running under a pruned
    budget run reads its budget as spent, not as unbounded."""
    report = await _workflow(pool, "report", _REPORT_BUDGET)
    root = await _root(pool, report, input=None, budget_usd=1.0)
    sub = await _direct_sub_run(pool, root, report)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM wf_runs WHERE id = $1", root)
    for _ in range(3):
        await run_workflow_step(sub)
    done = await _run(pool, sub)
    assert done.output == {"total_usd": 0.0, "spent_usd": 0.0, "remaining_usd": 0.0}
