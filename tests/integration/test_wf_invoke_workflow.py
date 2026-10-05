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
from aios.db.queries import trace as trace_q
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.agents import HttpMethod, HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.attenuation import surface_of
from aios.models.sessions import Session
from aios.models.workflows import AsAgent, OperatorAuthority, SessionAuthority, WfRun
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
        authority=OperatorAuthority(),
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
            conn,
            agent_deadline_seconds=999,
            agent_cost_ceiling_microusd=0,
            tool_stale_seconds=999,
            bash_default_timeout_seconds=120,
            sandbox_provisioning_slack_seconds=180,
            max_bash_timeout_seconds=3_155_760_000,
            call_llm_stale_seconds=999,
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
# The fix threaded the parent run's ``launcher_session_id`` down the
# ``parent_run_id`` lineage, so the sub-run acted for the ORIGINATING session. #2467
# moved the guard onto the immutable ``principal``, which the sub-run inherits, and
# #2472 clamps every sub-run to its parent run's frozen surface (``RunAuthority``),
# which the parent already took from the originating session at its own launch.


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
        authority=SessionAuthority(launcher_id, None),
        workflow_id=wf.id,
        environment_id="env_wf",
        input=input,
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


async def test_run_of_deleted_session_launches_subrun_within_its_frozen_surface(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472: a sub-run is bound by its parent run's frozen surface, not the launching
    session's live one, so deleting that session neither blocks the sub-run (#2467 had to
    refuse it) nor widens it: the declared ``read`` tool is still clamped away, and the
    sub-run still acts for a session."""
    pool = wf_runtime
    launcher_id = await _narrow_launcher(pool)  # empty tool surface
    async with pool.acquire() as conn:
        child_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="broad-child",
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="read")],
        )
    parent_run = await _agent_launched_parent(
        pool, _INVOKE_PARENT, input={"wf": child_wf.id}, launcher_id=launcher_id
    )
    async with pool.acquire() as conn:
        await db_queries.delete_session(conn, launcher_id, account_id="acc_wf")

    await run_workflow_step(parent_run)

    events = await _list(pool, parent_run)
    assert not [e for e in events if e.type == "call_result"]  # no refusal
    cs = next(e for e in events if e.type == "call_started")
    sub = await _run(pool, cs.payload["child_run_id"])
    assert sub.principal == "session"
    assert sub.launcher_session_id is None
    assert surface_of(sub).tools == []


async def test_operator_chain_subrun_is_clamped_to_parent_surface(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472: an operator run's sub-run is clamped to the parent's frozen surface too.
    Before, an operator chain had no launcher to clamp against, so a parent declaring
    no tools could reach a broader sub-workflow's whole surface."""
    pool = wf_runtime
    async with pool.acquire() as conn:
        child_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="broad-child",
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="read")],
        )
    parent_wf = await _insert_workflow(pool, "narrow-op-parent", _INVOKE_PARENT)
    parent_run = await _make_run(pool, parent_wf, input={"wf": child_wf.id})

    await run_workflow_step(parent_run)

    cs = next(e for e in await _list(pool, parent_run) if e.type == "call_started")
    sub = await _run(pool, cs.payload["child_run_id"])
    assert sub.principal == "operator"
    assert sub.as_agent is None
    assert surface_of(sub).tools == []


_INVOKE_AS_AGENT = (
    "async def main(input):\n"
    "    return await invoke_workflow(input['wf'], {}, as_agent=input['as_agent'])\n"
)


async def _agent_with_tools(pool: asyncpg.Pool[Any], name: str, tools: list[ToolSpec]) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name=name,
        model="test/dummy",
        system=name,
        tools=tools,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    return agent.id


async def _broad_child(pool: asyncpg.Pool[Any]) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="broad-child",
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="read"), ToolSpec(type="write")],
        )
    return wf.id


async def test_as_agent_reroots_an_operator_subrun_at_the_agent_surface(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472: ``as_agent`` clamps the sub-run to that agent version's surface within the
    parent's, and records which agent version it ran as."""
    pool = wf_runtime
    agent_id = await _agent_with_tools(pool, "reader", [ToolSpec(type="read")])
    child_wf = await _broad_child(pool)
    async with pool.acquire() as conn:
        parent_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="op-parent",
            script=_INVOKE_AS_AGENT,
            tools=[ToolSpec(type="read"), ToolSpec(type="write")],
        )
    parent_run = await _make_run(
        pool,
        parent_wf.id,
        input={"wf": child_wf, "as_agent": {"agent_id": agent_id, "version": 1}},
    )

    await run_workflow_step(parent_run)

    events = await _list(pool, parent_run)
    assert not [e for e in events if e.type == "call_result"]
    cs = next(e for e in events if e.type == "call_started")
    sub = await _run(pool, cs.payload["child_run_id"])
    assert sub.as_agent == AsAgent(agent_id=agent_id, version=1)
    assert [t.type for t in surface_of(sub).tools] == ["read"]
    assert sub.principal == "operator"
    async with pool.acquire() as conn:
        facts = await wf_queries.sub_run_facts(conn, parent_run, account_id="acc_wf", max_nodes=10)
    assert [n["as_agent"] for n in facts["nodes"]] == [{"agent_id": agent_id, "version": 1}]


@pytest.mark.parametrize(
    ("as_agent", "kind"),
    [
        ({"agent_id": "agt_missing", "version": 1}, "agent_version_not_found"),
        ({"agent_id": "agt_x", "version": True}, "bad_invoke_workflow"),
        ({"agent_id": "agt_x", "version": 1, "model": "x"}, "bad_invoke_workflow"),
        ("agt_x", "bad_invoke_workflow"),
    ],
)
async def test_as_agent_rejects_a_bad_or_missing_agent_version(
    wf_runtime: asyncpg.Pool[Any], as_agent: Any, kind: str
) -> None:
    pool = wf_runtime
    child_wf = await _broad_child(pool)
    parent_wf = await _insert_workflow(pool, "op-parent", _INVOKE_AS_AGENT)
    parent_run = await _make_run(pool, parent_wf, input={"wf": child_wf, "as_agent": as_agent})

    await run_workflow_step(parent_run)

    events = await _list(pool, parent_run)
    assert [e.payload["error"]["kind"] for e in events if e.type == "call_result"] == [kind]
    assert not [e for e in events if e.type == "call_started"]


async def test_as_agent_is_refused_to_a_run_that_acts_for_a_session(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472: re-rooting a sub-run at an agent's surface is an operator tool. A run that
    acts for a session must not borrow another agent's authority, even a narrower one."""
    pool = wf_runtime
    launcher_id = await _narrow_launcher(pool)
    agent_id = await _agent_with_tools(pool, "reader", [ToolSpec(type="read")])
    child_wf = await _broad_child(pool)
    parent_run = await _agent_launched_parent(
        pool,
        _INVOKE_AS_AGENT,
        input={"wf": child_wf, "as_agent": {"agent_id": agent_id, "version": 1}},
        launcher_id=launcher_id,
    )

    await run_workflow_step(parent_run)

    events = await _list(pool, parent_run)
    refusals = [e.payload["error"]["kind"] for e in events if e.type == "call_result"]
    assert refusals == ["invoke_workflow_refused"]
    async with pool.acquire() as conn:
        sub_runs = await conn.fetchval(
            "SELECT count(*) FROM wf_runs WHERE parent_run_id = $1", parent_run
        )
    assert sub_runs == 0


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
    assert sub.principal == "operator"

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


# ── #2472 D1: invoke_workflow(version=N) pins the sub-run's version ───────────

_INVOKE_PINNED = (
    "async def main(input):\n"
    "    return await invoke_workflow(input['wf'], {}, version=input['version'])\n"
)


async def _two_versions(pool: asyncpg.Pool[Any]) -> str:
    """A workflow whose v1 returns 'v1' and whose current v2 returns 'v2'."""
    wf_id = await _insert_workflow(pool, "versioned", "async def main(input):\n    return 'v1'\n")
    async with pool.acquire() as conn:
        await wf_queries.update_workflow(
            conn,
            wf_id,
            account_id="acc_wf",
            expected_version=1,
            script="async def main(input):\n    return 'v2'\n",
        )
    return wf_id


async def _invoke_once(pool: asyncpg.Pool[Any], wf_id: str, version: Any) -> list[Any]:
    parent_wf = await _insert_workflow(pool, f"pinned-parent-{version}", _INVOKE_PINNED)
    parent = await _make_run(pool, parent_wf, input={"wf": wf_id, "version": version})
    await run_workflow_step(parent)
    async with pool.acquire() as conn:
        return list(await wf_queries.list_run_events(conn, parent))


async def test_a_pinned_invoke_runs_the_pinned_version(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    wf_id = await _two_versions(pool)
    events = await _invoke_once(pool, wf_id, 1)
    started = next(e for e in events if e.type == "call_started")
    sub = await _run(pool, started.payload["child_run_id"])
    assert sub.source_version == 1
    await run_workflow_step(sub.id)
    assert (await _run(pool, sub.id)).output == "v1"


async def test_a_bad_or_unknown_version_is_a_catchable_rejection(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    pool = wf_runtime
    wf_id = await _two_versions(pool)
    cases = (
        (wf_id, 0, "bad_invoke_workflow"),
        # Past int4: a catchable rejection, not a DataError crashing the step.
        (wf_id, 2**31, "bad_invoke_workflow"),
        (wf_id, 7, "workflow_version_not_found"),
        # A pin on a workflow that doesn't exist is still workflow_not_found.
        ("wf_absent", 1, "workflow_not_found"),
    )
    for target, version, kind in cases:
        events = await _invoke_once(pool, target, version)
        kinds = [e.payload["error"]["kind"] for e in events if e.type == "call_result"]
        assert kinds == [kind], (target, version, kinds)


# ── #2472 D3: sub_runs() reads facts about the run's creation subtree ─────────

_READS_FACTS = (
    "async def main(input):\n"
    "    await invoke_workflow(input['wf'], {}, version=1, label='arm')\n"
    "    return await sub_runs()\n"
)


async def test_sub_runs_reports_each_sub_run_with_its_version_label_and_usage(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    pool = wf_runtime
    wf_id = await _two_versions(pool)
    parent_wf = await _insert_workflow(pool, "reads-facts", _READS_FACTS)
    parent = await _make_run(pool, parent_wf, input={"wf": wf_id})

    await run_workflow_step(parent)
    async with pool.acquire() as conn:
        started = next(
            e for e in await wf_queries.list_run_events(conn, parent) if e.type == "call_started"
        )
    sub_id = started.payload["child_run_id"]
    await run_workflow_step(sub_id)
    async with pool.acquire() as conn:
        await wf_queries.add_run_call_llm_cost_microusd(
            conn,
            sub_id,
            1234,
            account_id="acc_wf",
            input_tokens=100,
            output_tokens=20,
            model="openrouter/arm-model",
        )
    # Harvest the sub-run (sub_runs() resolves and journals inline), then the step
    # that fast-forwards through it and completes.
    await run_workflow_step(parent)
    await run_workflow_step(parent)

    facts = (await _run(pool, parent)).output
    assert facts["truncated"] is False
    [node] = facts["nodes"]
    assert (node["kind"], node["id"], node["label"]) == ("run", sub_id, "arm")
    assert (node["workflow_id"], node["workflow_version"]) == (wf_id, 1)
    assert node["status"] == "completed" and node["duration_ms"] is not None
    assert node["parent"] == {"kind": "run", "id": parent}
    assert node["usage"] == [
        {
            "model": "openrouter/arm-model",
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cost_microusd": 1234,
        }
    ]


_FANS_OUT = (
    "async def main(input):\n"
    "    return await parallel([\n"
    "        lambda: invoke_workflow(input['wf'], {}, label='arm'),\n"
    "        lambda: agent('hi', agent_id=input['agent'], label='judge'),\n"
    "    ])\n"
)


async def test_sub_run_facts_reports_live_session_and_run_children_and_truncates(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """Both node kinds, mid-flight: an agent() session and a still-running sub-run, in
    spawn order, each with the label its call gave it; ``max_nodes`` truncates. The
    trace's children-of walk carries the same labels."""
    pool = wf_runtime
    wf_id = await _two_versions(pool)
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name="judge-agent",
        model="test/judge",
        system="judge",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    parent_wf = await _insert_workflow(pool, "fans-out", _FANS_OUT)
    parent = await _make_run(pool, parent_wf, input={"wf": wf_id, "agent": agent.id})
    await run_workflow_step(parent)

    async with pool.acquire() as conn:
        facts = await wf_queries.sub_run_facts(conn, parent, account_id="acc_wf", max_nodes=10)
        cut = await wf_queries.sub_run_facts(conn, parent, account_id="acc_wf", max_nodes=1)
        kids = await trace_q.children_of(
            conn, caller_kind="run", caller_id=parent, account_id="acc_wf"
        )

    assert facts["truncated"] is False
    by_label = {n["label"]: n for n in facts["nodes"]}
    assert set(by_label) == {"arm", "judge"}
    started = [n["started_at"] for n in facts["nodes"]]
    assert started == sorted(started)
    arm, judge = by_label["arm"], by_label["judge"]
    assert arm["kind"] == "run" and arm["duration_ms"] is None
    assert arm["status"] not in {"completed", "errored", "cancelled"}
    assert judge["kind"] == "session"
    assert (judge["agent_id"], judge["agent_version"]) == (agent.id, agent.version)
    assert judge["parent"] == {"kind": "run", "id": parent}
    assert judge["archived"] is False and judge["usage"] == []

    assert cut["truncated"] is True and len(cut["nodes"]) == 1
    assert {k.id: k.label for k in kids} == {arm["id"]: "arm", judge["id"]: "judge"}


# ── #2472 B1: pin the clamp law against the mutations the suites above let through ──
#
# Each test below is RED under one argument-order or bound-source regression in
# ``service.create_run``: the as_agent inner meet swapped, the outer meet swapped, the
# RunAuthority bound re-read from the launching session's LIVE agent, or the clamp skipped
# for a pinned ``version=``.

_GH = "https://api.github.com"


def _narrow_gh() -> HttpServerSpec:
    """The parent's grant: GET-only on /repos/a/**, no query string."""
    return HttpServerSpec(
        name="gh",
        base_url=_GH,
        routes=[HttpRouteSpec(path_pattern="/repos/a/**", methods=["GET"], allow_query=False)],
    )


def _broad_gh() -> HttpServerSpec:
    """A wider grant on the same ``base_url``: every verb, query strings, and /**."""
    return HttpServerSpec(
        name="gh",
        base_url=_GH,
        routes=[
            HttpRouteSpec(path_pattern="/repos/a/**", methods=None, allow_query=True),
            HttpRouteSpec(path_pattern="/**", methods=None, allow_query=True),
        ],
    )


def _routes(run: WfRun) -> list[tuple[str, list[HttpMethod] | None, bool]]:
    return [
        (r.path_pattern, r.methods, r.allow_query)
        for srv in surface_of(run).http_servers
        for r in srv.routes
    ]


async def _spawned_sub_run(pool: asyncpg.Pool[Any], parent_run: str) -> WfRun:
    events = await _list(pool, parent_run)
    assert not [e for e in events if e.type == "call_result"]  # spawned, not refused
    cs = next(e for e in events if e.type == "call_started")
    return await _run(pool, cs.payload["child_run_id"])


async def _broad_http_child(pool: asyncpg.Pool[Any]) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="broad-http-child",
            script="async def main(input):\n    return 1\n",
            http_servers=[_broad_gh()],
        )
    return wf.id


async def _narrow_http_parent(pool: asyncpg.Pool[Any], script: str) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="narrow-http-parent",
            script=script,
            http_servers=[_narrow_gh()],
        )
    return wf.id


async def test_subrun_never_widens_parent_http_routes(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472 (c): an http server that survives the meet keeps the PARENT's routes,
    ``methods`` and ``allow_query``. The child declares every verb, ``allow_query`` and an
    extra ``/**`` route; none of it may reach the sub-run. RED if the outer meet's
    arguments are swapped (``clamp(bound, source_surface)``): the child's routes and
    ``allow_query`` would then be kept verbatim."""
    pool = wf_runtime
    child_wf = await _broad_http_child(pool)
    parent_wf = await _narrow_http_parent(pool, _INVOKE_PARENT)
    parent_run = await _make_run(pool, parent_wf, input={"wf": child_wf})
    assert _routes(await _run(pool, parent_run)) == [("/repos/a/**", ["GET"], False)]

    await run_workflow_step(parent_run)

    sub = await _spawned_sub_run(pool, parent_run)
    assert _routes(sub) == [("/repos/a/**", ["GET"], False)]


async def test_as_agent_subrun_never_widens_parent_http_routes(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472 (d): under ``as_agent`` the parent stays the binding (second) argument of the
    inner meet, so an agent version holding the broad http server grants the sub-run no
    route, verb or ``allow_query`` the parent lacks. RED if the inner meet's arguments are
    swapped (``clamp(bound, surface_of(av))``) or the outer one is."""
    pool = wf_runtime
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name="broad-http-agent",
        model="test/dummy",
        system="broad http agent",
        tools=[],
        http_servers=[_broad_gh()],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    child_wf = await _broad_http_child(pool)
    parent_wf = await _narrow_http_parent(pool, _INVOKE_AS_AGENT)
    parent_run = await _make_run(
        pool,
        parent_wf,
        input={"wf": child_wf, "as_agent": {"agent_id": agent.id, "version": agent.version}},
    )

    await run_workflow_step(parent_run)

    sub = await _spawned_sub_run(pool, parent_run)
    assert sub.as_agent == AsAgent(agent_id=agent.id, version=agent.version)
    assert _routes(sub) == [("/repos/a/**", ["GET"], False)]


async def test_launcher_widened_after_parent_launch_does_not_widen_subrun(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472 (a), the headline property: a sub-run is bound by its parent's FROZEN surface,
    not the launching session's current one. The launcher holds no tools when the parent
    launches and is widened to ``read`` before the parent spawns its sub-run; the sub-run
    must still get no tools. RED if the RunAuthority bound is re-read from the launcher
    session's live agent (the pre-#2472 law)."""
    pool = wf_runtime
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name="widened-launcher",
        model="test/dummy",
        system="widened launcher",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    session = await _make_launcher_session(pool, agent.id)
    async with pool.acquire() as conn:
        child_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="reads-child",
            script="async def main(input):\n    return 1\n",
            tools=[ToolSpec(type="read")],
        )
        # The parent declares ``read`` too, so only the launcher's surface AT LAUNCH
        # keeps it out of the parent's frozen surface.
        parent_wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="reads-parent",
            script=_INVOKE_PARENT,
            tools=[ToolSpec(type="read")],
        )
    parent = await service.create_run(
        pool,
        account_id="acc_wf",
        authority=SessionAuthority(session.id, None),
        workflow_id=parent_wf.id,
        environment_id="env_wf",
        input={"wf": child_wf.id},
    )
    assert surface_of(parent).tools == []

    widened = await agents_service.update_agent(
        pool,
        agent.id,
        account_id="acc_wf",
        expected_version=agent.version,
        tools=[ToolSpec(type="read")],
    )
    assert [t.type for t in widened.tools] == ["read"]

    await run_workflow_step(parent.id)

    sub = await _spawned_sub_run(pool, parent.id)
    assert sub.launcher_session_id == session.id
    assert surface_of(sub).tools == []


async def test_pinned_invoke_is_clamped_to_parent_surface(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """#2472 (b): a version-pinned ``invoke_workflow`` sub-run is clamped exactly like an
    unpinned one. The operator parent declares no tools; v1 of the child declares
    ``read``. RED if a pinned ``version=`` skips the clamp."""
    pool = wf_runtime
    async with pool.acquire() as conn:
        child = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="pinned-reads-child",
            script="async def main(input):\n    return 'v1'\n",
            tools=[ToolSpec(type="read")],
        )
        await wf_queries.update_workflow(
            conn,
            child.id,
            account_id="acc_wf",
            expected_version=1,
            script="async def main(input):\n    return 'v2'\n",
        )
    parent_wf = await _insert_workflow(pool, "pinned-no-tools-parent", _INVOKE_PINNED)
    parent_run = await _make_run(pool, parent_wf, input={"wf": child.id, "version": 1})

    await run_workflow_step(parent_run)

    sub = await _spawned_sub_run(pool, parent_run)
    assert sub.source_version == 1
    assert surface_of(sub).tools == []
