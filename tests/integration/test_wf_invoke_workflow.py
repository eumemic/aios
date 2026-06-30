"""``invoke_workflow()`` — the run-caller surface (#1129), end to end against a
real Postgres.

A workflow run invoking another workflow as a sub-run is the run dual of
``agent()``: the parent suspends on a journaled ``call_started`` carrying a
DETERMINISTIC ``child_run_id``; the sub-run completes and its terminal record
(``run_completed`` + ``status``) IS the answer; the parent's next step resolves it
through ``derive_run_response`` and fast-forwards it into the ``await`` (§3.6 — no
separate ``request_response`` event). These tests drive the harvest manually
(``run_workflow_step``), reusing ``test_wf_step``'s runtime fixture.
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
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.models.attenuation import surface_of
from aios.models.sessions import Session
from aios.models.workflows import WfRun
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.workflows import run_tools, service
from aios.workflows.child_run_id import child_run_id
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration


@pytest.fixture
async def wf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_wf', NULL, TRUE, 'wf-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_wf', 'wf-env', '{}'::jsonb, 'acc_wf')"
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _insert_workflow(pool: asyncpg.Pool[Any], name: str, script: str) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id="acc_wf", name=name, script=script)
    return wf.id


async def _make_run(pool: asyncpg.Pool[Any], workflow_id: str, *, input: Any = None) -> str:
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=workflow_id,
        environment_id="env_wf",
        input=input,
    )
    return run.id


async def _run(pool: asyncpg.Pool[Any], run_id: str) -> WfRun:
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None
    return run


async def _events(pool: asyncpg.Pool[Any], run_id: str) -> list[tuple[str, str | None]]:
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    return [(e.type, e.call_key) for e in rows]


# A parent that invokes the workflow id passed in its input and returns the result.
_PARENT = (
    "async def main(input):\n"
    "    r = await invoke_workflow(input['wf'], {'n': input['n']})\n"
    "    return {'got': r}\n"
)


async def test_invoke_workflow_happy_path_spawns_and_harvests(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """Parent suspends on ``call_started`` → sub-run completes → parent's next step
    harvests the sub-run's terminal answer (via ``derive_run_response``) and completes."""
    pool = wf_runtime
    child_wf = await _insert_workflow(
        pool, "child", "async def main(input):\n    return input['n'] + 1\n"
    )
    parent_wf = await _insert_workflow(pool, "parent", _PARENT)
    run_id = await _make_run(pool, parent_wf, input={"wf": child_wf, "n": 41})

    # Step 1: parent spawns the sub-run and suspends.
    await run_workflow_step(run_id)
    parent = await _run(pool, run_id)
    assert parent.status == "suspended"
    types = [t for t, _ in await _events(pool, run_id)]
    assert "call_started" in types and "call_result" not in types

    # The deterministic sub-run id is reproducible and the row carries the edge.
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    cs = next(e for e in rows if e.type == "call_started")
    sub_run_id = cs.payload["child_run_id"]
    assert cs.payload["capability"] == "invoke_workflow"
    sub = await _run(pool, sub_run_id)
    assert sub.parent_run_id == run_id
    assert sub.request_id == cs.call_key
    assert sub.caller == {"kind": "run", "id": run_id, "awaited": True}

    # Step 2: drive the sub-run to completion. Its terminal record (run_completed +
    # status) IS the answer — no separate request_response event (§3.6).
    await run_workflow_step(sub_run_id)
    sub = await _run(pool, sub_run_id)
    assert sub.status == "completed" and sub.output == 42
    sub_types = [t for t, _ in await _events(pool, sub_run_id)]
    assert "request_response" not in sub_types and "run_completed" in sub_types

    # Step 3: parent harvests the answer and completes.
    await run_workflow_step(run_id)
    parent = await _run(pool, run_id)
    assert parent.status == "completed"
    assert parent.output == {"got": 42}


async def test_cancelled_sub_run_wakes_its_parent_via_child_done(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A cancelled sub-run writes a durable ``child_done`` into its PARENT's signals (6d).

    ``invoke_workflow`` maps to NULL in the staleness sweep CASE, so a parked parent has NO
    backstop other than this explicit signal — without it (the 6b gap) a cancelled sub-run
    stranded its parent forever. The signal makes the parent sweep-visible (durable wake),
    and the harvested outcome is ``cancelled`` (the await raises, erroring the parent script).
    """
    pool = wf_runtime
    child_wf = await _insert_workflow(pool, "child", "async def main(input):\n    return 1\n")
    parent_wf = await _insert_workflow(pool, "parent", _PARENT)
    run_id = await _make_run(pool, parent_wf, input={"wf": child_wf, "n": 1})
    await run_workflow_step(run_id)  # parent spawns the sub-run and suspends
    async with pool.acquire() as conn:
        cs = next(
            e for e in await wf_queries.list_run_events(conn, run_id) if e.type == "call_started"
        )
    sub_run_id = cs.payload["child_run_id"]

    # Cancel the sub-run, then drive its terminal step (harvests the cancel → cancelled).
    async with pool.acquire() as conn:
        await wf_queries.insert_run_signal(
            conn, run_id=sub_run_id, call_key=wf_queries.CANCEL_SIGNAL_CALL_KEY, kind="cancel"
        )
    await run_workflow_step(sub_run_id)
    sub = await _run(pool, sub_run_id)
    assert sub.status == "cancelled"

    async with pool.acquire() as conn:
        signals = await wf_queries.list_run_signals(conn, run_id)
        assert any(s.kind == "child_done" and s.call_key == sub.request_id for s in signals)
        # The unharvested child_done makes the parent sweep-visible (the durable backstop).
        needing = await wf_queries.list_run_ids_needing_step(
            conn, agent_deadline_seconds=999, tool_stale_seconds=999
        )
        assert run_id in needing

    # The parent, when stepped, harvests ``cancelled`` — the await raises, erroring the script.
    await run_workflow_step(run_id)
    parent = await _run(pool, run_id)
    assert parent.status == "errored"


async def test_invoke_workflow_deterministic_reattach_on_replay(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """Re-driving the spawn step before the sub-run finishes re-attaches the SAME
    sub-run (deterministic id) — never a second sub-run."""
    pool = wf_runtime
    child_wf = await _insert_workflow(
        pool, "child", "async def main(input):\n    return input['n']\n"
    )
    parent_wf = await _insert_workflow(pool, "parent", _PARENT)
    run_id = await _make_run(pool, parent_wf, input={"wf": child_wf, "n": 7})

    await run_workflow_step(run_id)
    call_key = next(k for t, k in await _events(pool, run_id) if t == "call_started")
    await run_workflow_step(run_id)  # re-drive while still suspended

    async with pool.acquire() as conn:
        sub_ids = [
            r["id"]
            for r in await conn.fetch("SELECT id FROM wf_runs WHERE parent_run_id = $1", run_id)
        ]
    # Exactly ONE sub-run, at the deterministic id — the replay re-attached.
    assert sub_ids == [child_run_id(run_id, call_key or "")]


async def test_invoke_workflow_output_schema_violation_fails_loud(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A sub-run whose output violates the request's output_schema fails loud
    (``output_schema_violation``); the parent sees a catchable AgentError."""
    pool = wf_runtime
    # child returns a string; the request demands an integer.
    child_wf = await _insert_workflow(
        pool, "child", "async def main(input):\n    return 'not-an-int'\n"
    )
    parent = (
        "async def main(input):\n"
        "    try:\n"
        "        await invoke_workflow(input['wf'], {}, output_schema={'type': 'integer'})\n"
        "        return {'ok': True}\n"
        "    except Exception as e:\n"
        "        return {'caught': type(e).__name__}\n"
    )
    parent_wf = await _insert_workflow(pool, "parent", parent)
    run_id = await _make_run(pool, parent_wf, input={"wf": child_wf})

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    sub_run_id = next(e for e in rows if e.type == "call_started").payload["child_run_id"]

    await run_workflow_step(sub_run_id)
    sub = await _run(pool, sub_run_id)
    assert sub.status == "errored"
    completed = next(e for e in await _list(pool, sub_run_id) if e.type == "run_completed")
    assert completed.payload["error"]["kind"] == "output_schema_violation"

    await run_workflow_step(run_id)
    parent_run = await _run(pool, run_id)
    assert parent_run.status == "completed"
    assert parent_run.output == {"caught": "AgentError"}


async def test_invoke_workflow_missing_target_is_catchable(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """An unknown ``workflow_id`` rejects as a catchable author error — no sub-run."""
    pool = wf_runtime
    parent = (
        "async def main(input):\n"
        "    try:\n"
        "        await invoke_workflow('wf_does_not_exist', {})\n"
        "        return {'ok': True}\n"
        "    except Exception as e:\n"
        "        return {'caught': type(e).__name__}\n"
    )
    parent_wf = await _insert_workflow(pool, "parent", parent)
    run_id = await _make_run(pool, parent_wf, input={})

    # First step journals the rejection + self-wakes; second step replays + throws.
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)
    parent_run = await _run(pool, run_id)
    assert parent_run.status == "completed"
    assert parent_run.output == {"caught": "AgentError"}
    async with pool.acquire() as conn:
        n = await conn.fetchval("SELECT count(*) FROM wf_runs WHERE parent_run_id = $1", run_id)
    assert n == 0


async def _list(pool: asyncpg.Pool[Any], run_id: str) -> list[Any]:
    async with pool.acquire() as conn:
        return list(await wf_queries.list_run_events(conn, run_id))


# ── #1653: invoke_workflow sub-run principal lineage (privilege-escalation fix) ──
#
# An ``invoke_workflow`` sub-run is created via ``create_run`` WITHOUT the
# originating principal threaded through, so it defaulted to ``launcher_session_id
# = NULL`` and was mis-classified as an OPERATOR run. That gives a self-authoring
# agent two escalations through the run→sub-run seam:
#
#   * the #1636 ``workflow:`` model-binding guard (keyed on
#     ``run.launcher_session_id is None``) is BYPASSED — a sub-run may bind the
#     operator-only ``workflow:`` model for its children;
#   * the #794 launcher surface clamp (``service.py``: gated on
#     ``launcher_session_id is not None``) is SKIPPED — the sub-run runs
#     un-attenuated on the tool/mcp/http axis.
#
# The fix threads the parent run's ``launcher_session_id`` down the
# ``parent_run_id`` lineage so ``is_operator_run`` reflects the ORIGINATING
# principal, not the sub-run's absent launcher. Both exposures close at once.


async def _make_launcher_session(pool: asyncpg.Pool[Any], agent_id: str) -> Session:
    return await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )


async def _narrow_launcher(pool: asyncpg.Pool[Any]) -> str:
    """A launcher SESSION whose agent has an EMPTY tool surface — the originating
    (non-operator) principal. Its surface is the lattice floor, so any tool a
    sub-workflow declares must be clamped away once the sub-run inherits it.

    Returns the session id (what ``launcher_session_id`` must be)."""
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name="narrow-launcher",
        model="test/dummy",
        system="narrow launcher",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    session = await _make_launcher_session(pool, agent.id)
    return session.id


async def _agent_launched_parent(
    pool: asyncpg.Pool[Any], parent_script: str, *, input: Any, launcher_id: str
) -> str:
    """A parent run LAUNCHED BY AN AGENT (non-operator originating principal)."""
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="parent-1653", script=parent_script
        )
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=wf.id,
        environment_id="env_wf",
        input=input,
        launcher_session_id=launcher_id,
    )
    return run.id


# Parent A: invoke sub-workflow B (no args needed beyond its id).
_INVOKE_PARENT = "async def main(input):\n    return await invoke_workflow(input['wf'], {})\n"
# Sub-workflow B: route a child's inference through an operator-only workflow: model.
_BINDS_WORKFLOW_MODEL = (
    "async def main(input):\n    return await agent('go', model='workflow:wf_bound')\n"
)


async def test_agent_originated_invoke_workflow_subrun_cannot_bind_workflow_model(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#1653: A→B→Y chain. An agent launches parent A (``invoke_workflow(B)``); B's
    script binds a ``workflow:`` model for a child. The sub-run B inherits A's
    originating principal, so it is NON-operator and the #1636 guard REJECTS the
    binding (``workflow_model_forbidden``) before any child row exists.

    On master this FAILS: the sub-run defaults to ``launcher_session_id = NULL`` →
    mis-classified operator → the binding sails through and a child is spawned."""
    pool = wf_runtime
    launcher_id = await _narrow_launcher(pool)
    child_wf = await _insert_workflow(pool, "binds-wf-model", _BINDS_WORKFLOW_MODEL)
    parent_run = await _agent_launched_parent(
        pool, _INVOKE_PARENT, input={"wf": child_wf}, launcher_id=launcher_id
    )

    # Step 1: parent A spawns the sub-run B and suspends.
    await run_workflow_step(parent_run)
    async with pool.acquire() as conn:
        cs = next(
            e
            for e in await wf_queries.list_run_events(conn, parent_run)
            if e.type == "call_started"
        )
    sub_run_id = cs.payload["child_run_id"]

    # The sub-run carries the ORIGINATING principal (not a NULL/operator launcher).
    sub = await _run(pool, sub_run_id)
    assert sub.launcher_session_id == launcher_id

    # Step 2: drive sub-run B — its agent(model='workflow:…') binding must be
    # REJECTED as operator-only, and NO child session may be created.
    await run_workflow_step(sub_run_id)
    async with pool.acquire() as conn:
        sub_events = await wf_queries.list_run_events(conn, sub_run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", sub_run_id
        )
    result_evt = next(e for e in sub_events if e.type == "call_result")
    assert result_evt.payload["error"]["kind"] == "workflow_model_forbidden"
    assert children == 0  # refused before write — no escalated child


async def test_agent_originated_invoke_workflow_subrun_surface_is_attenuated(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#1653 (the bonus exposure): an ``invoke_workflow`` sub-run launched off an
    agent-originated parent must be CLAMPED to the launcher's surface (#794), not
    snapshotted verbatim. The launcher holds NO tools, so a sub-workflow declaring
    a tool must have it clamped AWAY on the sub-run.

    On master this FAILS: the NULL launcher skips the clamp → the declared tool
    survives un-attenuated on the sub-run."""
    pool = wf_runtime
    launcher_id = await _narrow_launcher(pool)  # empty tool surface
    # Sub-workflow declares a `read` tool (a surface broader than the launcher's).
    async with pool.acquire() as conn:
        child_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="broad-surface-child",
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="read")],
        )
    parent_run = await _agent_launched_parent(
        pool, _INVOKE_PARENT, input={"wf": child_wf.id}, launcher_id=launcher_id
    )

    await run_workflow_step(parent_run)
    async with pool.acquire() as conn:
        cs = next(
            e
            for e in await wf_queries.list_run_events(conn, parent_run)
            if e.type == "call_started"
        )
    sub = await _run(pool, cs.payload["child_run_id"])

    # The sub-run inherits the originating principal AND its surface is clamped to
    # the launcher's (empty) — the declared `read` tool is attenuated away.
    assert sub.launcher_session_id == launcher_id
    assert surface_of(sub).tools == []


async def test_operator_originated_invoke_workflow_subrun_may_bind_workflow_model(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#1653 NO-OVER-RESTRICTION: a genuine operator/HTTP ``invoke_workflow`` chain
    (edgeless root parent → no launcher all the way down) stays operator-classified,
    so the operator privilege to bind a ``workflow:`` model survives in the sub-run.

    Guards against a fix that over-restricts by blanket-blocking every sub-run."""
    pool = wf_runtime
    child_wf = await _insert_workflow(pool, "binds-wf-model", _BINDS_WORKFLOW_MODEL)
    parent_wf = await _insert_workflow(pool, "op-parent", _INVOKE_PARENT)
    # Edgeless root: no launcher_session_id → operator-owned, like POST /runs.
    parent_run = await _make_run(pool, parent_wf, input={"wf": child_wf})

    await run_workflow_step(parent_run)
    async with pool.acquire() as conn:
        cs = next(
            e
            for e in await wf_queries.list_run_events(conn, parent_run)
            if e.type == "call_started"
        )
    sub_run_id = cs.payload["child_run_id"]
    sub = await _run(pool, sub_run_id)
    assert sub.launcher_session_id is None  # operator lineage preserved

    # The sub-run binds the workflow: model freely (operator privilege) — the child
    # spawns and its model is stamped, with NO rejection journaled.
    with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()):
        await run_workflow_step(sub_run_id)
    async with pool.acquire() as conn:
        sub_events = await wf_queries.list_run_events(conn, sub_run_id)
        started = next(e for e in sub_events if e.type == "call_started")
        child = await db_queries.get_session_bare(
            conn, started.payload["child_session_id"], account_id="acc_wf"
        )
    assert not [e for e in sub_events if e.type == "call_result"]  # no rejection
    assert child.model == "workflow:wf_bound"  # binding admitted + stamped


# A→B→C→Y. A invokes a fixed B forwarding {'wf': C}; B invokes input['wf'] (= C).
_INVOKE_FIXED_FORWARDING_NEXT = (
    "async def main(input):\n    return await invoke_workflow(input['b'], {'wf': input['c']})\n"
)
_INVOKE_FROM_INPUT = "async def main(input):\n    return await invoke_workflow(input['wf'], {})\n"


async def test_agent_originated_nested_invoke_workflow_propagates_principal(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#1653 self-review: the principal must propagate at EVERY depth of a nested
    ``invoke_workflow → invoke_workflow → …`` chain, since each sub-run inherits its
    parent run's launcher verbatim. An agent launches A; A invokes B; B invokes C; C
    binds a ``workflow:`` model. The grandchild sub-run C is STILL non-operator (the
    launcher threads through both hops), so the #1636 guard rejects the binding."""
    pool = wf_runtime
    launcher_id = await _narrow_launcher(pool)
    c_wf = await _insert_workflow(pool, "leaf-binds-wf-model", _BINDS_WORKFLOW_MODEL)
    b_wf = await _insert_workflow(pool, "mid-invoker", _INVOKE_FROM_INPUT)
    # A invokes B (fixed), forwarding {'wf': C}; B then invokes C.
    a_run = await _agent_launched_parent(
        pool,
        _INVOKE_FIXED_FORWARDING_NEXT,
        input={"b": b_wf, "c": c_wf},
        launcher_id=launcher_id,
    )

    # Hop 1: A spawns B (the mid invoker).
    await run_workflow_step(a_run)
    async with pool.acquire() as conn:
        b_cs = next(
            e for e in await wf_queries.list_run_events(conn, a_run) if e.type == "call_started"
        )
    b_run_id = b_cs.payload["child_run_id"]
    b_sub = await _run(pool, b_run_id)
    assert b_sub.launcher_session_id == launcher_id  # hop-1 inherits the originator

    # Hop 2: drive B → it invokes C. C inherits B's launcher (= the originator).
    await run_workflow_step(b_run_id)
    async with pool.acquire() as conn:
        c_cs = next(
            e for e in await wf_queries.list_run_events(conn, b_run_id) if e.type == "call_started"
        )
    c_run_id = c_cs.payload["child_run_id"]
    c_sub = await _run(pool, c_run_id)
    assert c_sub.launcher_session_id == launcher_id  # hop-2 STILL the originator

    # The grandchild C is non-operator → its workflow: binding is rejected.
    await run_workflow_step(c_run_id)
    async with pool.acquire() as conn:
        c_events = await wf_queries.list_run_events(conn, c_run_id)
        grandkids = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", c_run_id
        )
    result_evt = next(e for e in c_events if e.type == "call_result")
    assert result_evt.payload["error"]["kind"] == "workflow_model_forbidden"
    assert grandkids == 0
