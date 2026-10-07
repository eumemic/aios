"""Nested run budgets (#2476): a sub-run with no budget of its own is held to its
nearest budgeted ancestor's ceiling, ``invoke_workflow`` is refused once the budget
is spent, and ``invoke_workflow(budget_usd=)`` gives a sub-run its own ceiling, clamped
to what its caller may still spend.

Spend is seeded straight onto a run's ``call_llm`` meter, which the subtree rollup
reads, so the tests control exactly what each budget run has spent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.workflows import OperatorAuthority, WfRun
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
