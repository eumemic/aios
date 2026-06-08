"""B1.4 + B1.5 — run_workflow_step end to end against a real Postgres.

The headline is the gate round-trip: a run suspends at a gate, an external resume
delivers a value, and the next wake replays-with-memo past the gate to completion
— with the journal staying a clean [run_started, call_started, call_result,
run_completed] and every step idempotent.

``defer_run_wake`` (the procrastinate enqueue) is patched out; the step is driven
directly, which is exactly the surface under test.
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
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.workflows import service
from aios.workflows.child_id import child_session_id
from aios.workflows.determinism import CallKeyer
from aios.workflows.host_launcher import HostOutcome
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_GATE_SCRIPT = (
    "async def main(input):\n    r = await gate({'q': 'ok?'})\n    return {'answer': r}\n"
)


@pytest.fixture
async def wf_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool installed on ``runtime.pool`` (so the step's ``require_pool`` sees
    it) + a seeded root tenant; ``defer_run_wake`` patched out."""
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
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _events(pool: asyncpg.Pool[Any], run_id: str) -> list[tuple[int, str, str | None]]:
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    return [(e.seq, e.type, e.call_key) for e in rows]


async def _make_run(pool: asyncpg.Pool[Any], script: str, *, input: Any = None) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id="acc_wf", name="w", script=script)
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf", input=input
    )
    return run.id


@pytest.fixture
async def wf_agent_id(wf_runtime: asyncpg.Pool[Any]) -> str:
    """A minimal agent for spawning children (model is never called in these tests)."""
    agent = await agents_service.create_agent(
        wf_runtime,
        account_id="acc_wf",
        name="child-agent",
        model="test/dummy",
        system="test child agent",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    return agent.id


async def _spawn_child(pool: asyncpg.Pool[Any], agent_id: str, ordinal: str) -> tuple[str, str]:
    """Make a run + idempotently spawn one child for ``ordinal``; return (run_id, cid)."""
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    cid = child_session_id(run_id, ordinal)
    await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=agent_id,
        environment_id="env_wf",
        agent_version=1,
        parent_run_id=run_id,
        request_id=ordinal,
        input="hi",
    )
    return run_id, cid


# ─── B2.B — child create (idempotent) + session read-model ───────────────────


async def test_create_child_session_idempotent(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    cid = child_session_id(run_id, "sha:x#0")

    created = await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        parent_run_id=run_id,
        request_id="sha:x#0",
        input={"q": "hi"},
    )
    assert created is True
    # A replay (same id) is a rowcount-0 no-op — no double row, no double request.
    again = await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        parent_run_id=run_id,
        request_id="sha:x#0",
        input={"q": "hi"},
    )
    assert again is False
    async with pool.acquire() as conn:
        user_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'message' AND data->>'role' = 'user'",
            cid,
            "acc_wf",
        )
        # The request is delivered exactly once, carrying its correlation metadata.
        req = await conn.fetchrow(
            "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'message' AND role = 'user' ORDER BY seq ASC LIMIT 1",
            cid,
            "acc_wf",
        )
    assert user_msgs == 1
    from aios.db.queries import parse_jsonb

    request = parse_jsonb(req["data"])["metadata"]["request"]
    assert request["request_id"] == "sha:x#0" and request["caller"] == {"kind": "run", "id": run_id}


async def test_child_session_origin_and_parent_round_trip(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:y#0")
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert child.origin == "background"
    assert child.parent_run_id == run_id
    assert child.agent_version == 1  # pinned, not None


async def test_pure_script_completes_in_one_wake(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(
        pool, "async def main(input):\n    return input['x'] * 2", input={"x": 21}
    )
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == 42
    assert [(t, k) for _s, t, k in await _events(pool, run_id)] == [
        ("run_started", None),
        ("run_completed", None),
    ]


async def test_gate_suspend_resume_replay_roundtrip(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)

    # Wake 1: drives to the gate and parks.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"
    events = await _events(pool, run_id)
    assert [(t, k is not None) for _s, t, k in events] == [
        ("run_started", False),
        ("call_started", True),
    ]
    gate_key = events[1][2]
    assert gate_key is not None

    # Resume: a durable signal is recorded (the journal is untouched until harvest).
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")
    assert [t for _s, t, _k in await _events(pool, run_id)] == ["run_started", "call_started"]

    # Wake 2: harvest the signal → fast-forward past the gate → complete.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"answer": "yes"}
    assert [t for _s, t, _k in await _events(pool, run_id)] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]


async def test_terminal_run_and_double_resume_are_noops(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)
    gate_key = (await _events(pool, run_id))[1][2]
    assert gate_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")
    await run_workflow_step(run_id)  # → completed

    before = await _events(pool, run_id)
    await run_workflow_step(run_id)  # wake on a completed run: no-op
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="OTHER")  # idempotent
    await run_workflow_step(run_id)
    assert await _events(pool, run_id) == before  # journal unchanged


# ─── B2.C — the agent spawn arm + crash matrix ───────────────────────────────


async def test_agent_spawn_creates_child_and_suspends(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, {{'task': 'go'}})\n"
    run_id = await _make_run(pool, script)
    with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as child_wake:
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "suspended"
    started = [e for e in events if e.type == "call_started"]
    assert len(started) == 1
    assert started[0].payload["capability"] == "agent"
    assert started[0].payload["child_agent_version"] == 1  # pinned
    child_id = started[0].payload["child_session_id"]

    # The child row exists (background, linked to the run) with the input delivered.
    child = await sessions_service.get_session_basic(pool, child_id, account_id="acc_wf")
    assert child.origin == "background" and child.parent_run_id == run_id
    async with pool.acquire() as conn:
        user_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND data->>'role' = 'user'",
            child_id,
        )
    assert user_msgs == 1
    child_wake.assert_awaited_once()  # prompt wake of the child


async def test_agent_spawn_idempotent_on_replay(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """C1/C2/C6: a re-step (crash replay) must NOT double-spawn or re-deliver input."""
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, 'hi')\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # wake 1: spawn
    await run_workflow_step(run_id)  # wake 2: replay — re-emits the same frontier

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    started = [e for e in events if e.type == "call_started"]
    assert len(started) == 1  # exactly one call_started — no double-spawn
    assert children == 1  # exactly one child row
    child_id = started[0].payload["child_session_id"]
    async with pool.acquire() as conn:
        user_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND data->>'role' = 'user'",
            child_id,
        )
    assert user_msgs == 1  # input delivered exactly once across replays


async def test_agent_not_found_errors_the_run(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(
        pool, "async def main(input):\n    return await agent('agent_nope', 'x')\n"
    )
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "agent_not_found"


# ─── B2.D — return()/error() completion tools + injection gate ────────────────


async def test_completion_tools_injected_only_for_children(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.harness.step_context import compute_step_prelude

    pool = wf_runtime
    # compute_step_prelude asks the tool provider for connection tools; stub it.
    prev_tp = runtime.tool_provider
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[])
    runtime.tool_provider = tp
    try:
        await _check_completion_injection(pool, wf_agent_id, compute_step_prelude)
    finally:
        runtime.tool_provider = prev_tp


async def _check_completion_injection(
    pool: asyncpg.Pool[Any], wf_agent_id: str, compute_step_prelude: Any
) -> None:
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d1#0")
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    child_agent = await agents_service.load_for_session(pool, child, account_id="acc_wf")
    child_prelude = await compute_step_prelude(
        pool,
        cid,
        account_id="acc_wf",
        session=child,
        agent=child_agent,
        channels=[],
        memory_store_echoes=[],
    )
    child_tools = {t["function"]["name"] for t in child_prelude.tools}
    assert {"return", "error"} <= child_tools

    fg = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )
    fg_session = await sessions_service.get_session_basic(pool, fg.id, account_id="acc_wf")
    fg_agent = await agents_service.load_for_session(pool, fg_session, account_id="acc_wf")
    fg_prelude = await compute_step_prelude(
        pool,
        fg.id,
        account_id="acc_wf",
        session=fg_session,
        agent=fg_agent,
        channels=[],
        memory_store_echoes=[],
    )
    fg_tools = {t["function"]["name"] for t in fg_prelude.tools}
    assert "return" not in fg_tools and "error" not in fg_tools


async def test_return_writes_response_and_wakes_caller_without_archiving(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """return() writes the request's response + wakes the caller, but does NOT
    archive the child — responding is not terminating. Reclamation is run-end
    (off the correctness path), so right after responding the child is unarchived."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d2#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        res = await workflow_completion.return_handler(cid, {"value": {"answer": 42}})
    assert res == {"status": "returned"}
    wake.assert_awaited_once_with(run_id)

    async with pool.acquire() as conn:
        response = await db_queries.read_workflow_child_done(conn, cid, account_id="acc_wf")
    assert response is not None
    assert response["is_error"] is False and response["result"] == {"answer": 42}
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert child.archived_at is None  # NOT archived by the handler — run-end reclaim does that


async def test_error_marker_carries_message(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion

    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d2e#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        res = await workflow_completion.error_handler(cid, {"message": "nope"})
    assert res == {"status": "errored"}
    async with pool.acquire() as conn:
        marker = await db_queries.read_workflow_child_done(conn, cid, account_id="acc_wf")
    assert marker is not None and marker["is_error"] is True
    assert marker["error"] == {"message": "nope"}


async def test_return_from_non_child_fails_closed(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion
    from aios.tools.registry import ToolResult

    pool = wf_runtime
    fg = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        res = await workflow_completion.return_handler(fg.id, {"value": 1})
    assert isinstance(res, ToolResult) and res.is_error
    wake.assert_not_awaited()  # never signal a NULL parent run
    async with pool.acquire() as conn:
        marker = await db_queries.read_workflow_child_done(conn, fg.id, account_id="acc_wf")
    assert marker is None


async def test_request_is_answered_exactly_once(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that responds twice (return then return, or return racing error)
    yields exactly ONE response — first-writer-wins — and wakes the caller once."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:once#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        r1 = await workflow_completion.return_handler(cid, {"value": "first"})
        r2 = await workflow_completion.error_handler(cid, {"message": "too late"})
    assert r1 == {"status": "returned"} and r2 == {"status": "errored"}
    wake.assert_awaited_once_with(run_id)  # only the first response woke the caller

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'",
            cid,
            "acc_wf",
        )
    assert len(rows) == 1  # exactly one response
    response = db_queries.parse_jsonb(rows[0]["data"])
    assert response["is_error"] is False and response["result"] == "first"  # the first one won
    assert response["request_id"] == "sha:once#0"  # correlated to the request


async def test_return_real_dispatch_appends_result_and_does_not_archive(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The blind spot the bug survived in: drive ``return`` through the REAL tool
    path (``_execute_tool_async``/``_tool_lifecycle``), not the handler directly.
    Marker-only completion must append exactly one ``tool_result`` + both spans +
    the marker, wake the parent, and leave the child UN-archived — so a concurrent
    sibling tool in the same batch could still append (the multi-tool race that
    sank the terminal-tool design). No exception is the headline: the original bug
    crashed here on every completion."""
    import json as _json

    from aios.harness import tool_dispatch
    from aios.harness.task_registry import TaskRegistry

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:rd#0")
    call = {
        "id": "call_ret",
        "function": {"name": "return", "arguments": _json.dumps({"value": "ok"})},
    }

    prev_reg = runtime.task_registry
    runtime.task_registry = TaskRegistry()
    try:
        with (
            mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake,
            mock.patch("aios.harness.sweep.defer_wake", new=AsyncMock()),
        ):
            await tool_dispatch._execute_tool_async(pool, cid, call, account_id="acc_wf")
    finally:
        runtime.task_registry = prev_reg

    wake.assert_awaited_once_with(run_id)
    async with pool.acquire() as conn:
        result = await db_queries.find_tool_result_event(conn, cid, "call_ret", account_id="acc_wf")
        marker = await db_queries.read_workflow_child_done(conn, cid, account_id="acc_wf")
        events = await db_queries.read_events(conn, cid, account_id="acc_wf", limit=1000)
    # invariant #4: exactly one tool_result for the call; both spans balanced.
    assert result is not None and not bool(result.data.get("is_error"))
    spans = [e.data.get("event") for e in events if e.kind == "span"]
    assert spans.count("tool_execute_start") == 1 and spans.count("tool_execute_end") == 1
    assert marker is not None and marker["result"] == "ok"
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert child.archived_at is None  # un-archived → a sibling tool's appends won't crash


async def test_reattach_skips_defer_wake_and_self_wakes_on_marker(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """``defer_wake(child)`` fires only on first spawn; a re-attach skips it (the
    child is already sweep-wakeable, and may even be terminal — a wake span would
    crash). If a re-attached child already has its marker (C1'/C4), the spawn arm
    requests a self-wake so the parent harvests it now."""
    from aios.workflows.host_launcher import EmittedCapability
    from aios.workflows.step import _open_agent_capability

    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    spec = {"agent_id": wf_agent_id, "input": "go", "output_schema": None}
    call_key = CallKeyer().next("agent", spec)
    cap = EmittedCapability(capability_id="agent", call_key=call_key, spec=spec)
    cid = child_session_id(run_id, call_key)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        # First spawn: created → defer_wake fires once.
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w1:
            r1 = await _open_agent_capability(conn, pool, run, cap)
        assert not r1.rejected and not r1.needs_rewake
        w1.assert_awaited_once()
        # Re-attach, no marker yet → no defer_wake, no self-wake.
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w2:
            r2 = await _open_agent_capability(conn, pool, run, cap)
        assert not r2.rejected and not r2.needs_rewake
        w2.assert_not_awaited()

    # Child completes before the parent journals call_started (C1'/C4).
    from aios.tools import workflow_completion

    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(cid, {"value": "r"})

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w3:
            r3 = await _open_agent_capability(conn, pool, run, cap)
        assert not r3.rejected and r3.needs_rewake  # marker present → self-wake to harvest
        w3.assert_not_awaited()


# ─── B2.E — harvest the child marker (spawn -> return -> complete) ────────────


async def _child_id_of(pool: asyncpg.Pool[Any], run_id: str) -> str:
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    return next(e.payload["child_session_id"] for e in events if e.type == "call_started")


async def test_agent_round_trip_harvest_completes_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)

    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(child_id, {"value": "child-result"})

    await run_workflow_step(run_id)  # harvest marker -> call_result -> replay -> complete
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "completed" and run.output == "done"
    assert [e.type for e in events] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is False and cr.payload["result"] == "child-result"


async def test_uncaught_agent_error_bubbles_and_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R3 — an agent error the script does NOT catch is thrown at the ``await`` as
    an ``AgentError``, propagates out of ``main``, and errors the run (the bubble).
    The harvest still journals the child's error in the ``call_result``; the run's
    terminal output is the raised ``AgentError`` repr."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    child_id = await _child_id_of(pool, run_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.error_handler(child_id, {"message": "boom"})

    await run_workflow_step(run_id)  # harvest -> throw AgentError -> uncaught -> RAISED
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is True and cr.payload["error"] == {"message": "boom"}
    rc = next(e for e in events if e.type == "run_completed")
    assert rc.payload["is_error"] is True
    assert "AgentError: boom" in rc.payload["output"]  # the raised AgentError bubbled out
    assert run is not None and run.status == "errored"


async def test_workflow_try_except_agent_error_continues(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R3 headline — a workflow can ``try/except AgentError`` a failing agent and
    carry on to a clean completion (Tom's "try/except a failing agent")."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    child_id = await _child_id_of(pool, run_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.error_handler(child_id, {"message": "nope"})

    await run_workflow_step(run_id)  # harvest -> throw AgentError -> caught -> return
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    # An explicit error() carries no kind; the script caught it and completed.
    assert run is not None and run.status == "completed"
    assert run.output == {"caught": True, "kind": None}


async def test_workflow_uses_agent_return_value(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R3 ``{ok}`` unwrap through the full step: the value the child ``return``s is
    fast-forwarded into the ``await`` for the script to use."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = f"async def main(input):\n    r = await agent({wf_agent_id!r}, 'go')\n    return {{'got': r}}\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    child_id = await _child_id_of(pool, run_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(child_id, {"value": "the-answer"})

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed" and run.output == {"got": "the-answer"}


async def test_model_failure_writes_error_response_and_run_resolves(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R2 — the totality hole: a child whose model errors past its retry budget can
    no longer answer its request. The harness erroring path responds on its behalf
    with a monotonic ``child_errored`` response, so the invoking run harvests it and
    resolves — instead of hanging forever on a dead child."""
    from aios.harness import loop

    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)

    # Drive the child to its terminal-error landing pad: seed a full rescheduling
    # streak so the next failure spends the retry budget, then run the erroring path.
    for _ in range(len(loop._RETRY_BACKOFF_SECONDS)):
        await loop._append_lifecycle(
            pool, child_id, "turn_ended", "rescheduling", "rescheduling", account_id="acc_wf"
        )
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        delay = await loop._apply_retry_or_failure(pool, child_id, account_id="acc_wf")
    assert delay is None  # terminal — retry budget spent
    wake.assert_awaited_once_with(run_id)  # the caller was woken to harvest

    async with pool.acquire() as conn:
        response = await db_queries.read_workflow_child_done(conn, child_id, account_id="acc_wf")
    assert response is not None
    assert response["is_error"] is True and response["error"] == {"kind": "child_errored"}

    await run_workflow_step(run_id)  # harvest the error response -> resolve (no hang)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is True and cr.payload["error"] == {"kind": "child_errored"}
    # The run RESOLVES instead of hanging — the harvested child_errored becomes an
    # AgentError at the await; uncaught here, it bubbles and terminally errors the
    # run (R3). The totality hole is closed: a dead child no longer hangs the run.
    rc = next(e for e in events if e.type == "run_completed")
    assert rc.payload["is_error"] is True and "AgentError" in rc.payload["output"]
    assert run is not None and run.status == "errored"


async def test_model_failure_does_not_clobber_a_prior_response(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R2 first-writer-wins: a child that already ``return``ed keeps that response
    even if its model later errors out. The erroring path's ``child_errored``
    response no-ops and does NOT re-wake the caller (it was woken by the return)."""
    from aios.harness import loop
    from aios.tools import workflow_completion

    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:prec#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(cid, {"value": "real"})

    for _ in range(len(loop._RETRY_BACKOFF_SECONDS)):
        await loop._append_lifecycle(
            pool, cid, "turn_ended", "rescheduling", "rescheduling", account_id="acc_wf"
        )
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        await loop._apply_retry_or_failure(pool, cid, account_id="acc_wf")
    wake.assert_not_awaited()  # duplicate response is a no-op — caller already woken

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'lifecycle' AND data->>'event' = 'workflow_child_done'",
            cid,
            "acc_wf",
        )
    assert len(rows) == 1  # still exactly one response
    response = db_queries.parse_jsonb(rows[0]["data"])
    assert response["is_error"] is False and response["result"] == "real"  # the return() won


async def test_agent_stays_suspended_until_marker(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    await run_workflow_step(run_id)  # marker absent -> stays suspended, no call_result
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "suspended"
    assert not any(e.type == "call_result" for e in events)


# ─── crash-safety + divergence (B1.6 + B1.7) ─────────────────────────────────


async def test_resumes_from_journaled_call_result_after_crash(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """C5: a crash after the gate's call_result is journaled but before
    run_completed → the next wake fast-forwards to completion, no re-run."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)  # → suspended, call_started{gate}
    gate_key = (await _events(pool, run_id))[1][2]
    assert gate_key is not None

    # Simulate the harvest that committed just before the crash (status stays
    # 'suspended' — the run_completed + status flip never happened).
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="call_result",
            call_key=gate_key,
            payload={"result": "yes", "is_error": False},
        )

    await run_workflow_step(run_id)  # fast-forward past the gate → complete
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"answer": "yes"}
    assert [t for _s, t, _k in await _events(pool, run_id)] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]


async def test_divergent_replay_is_caught(wf_runtime: asyncpg.Pool[Any]) -> None:
    """The replay-prefix assertion: an open capability the script never re-emits
    (a nondeterministic prior wake) errors the run instead of orphaning it."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    # Inject a journal whose open capability has a call_key the script can't emit.
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn, account_id="acc_wf", run_id=run_id, type="run_started", payload={"input": None}
        )
        await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="call_started",
            call_key="sha:deadbeef#0",
            payload={"capability": "gate", "gate_nonce": "x"},
        )

    await run_workflow_step(run_id)  # host emits the REAL gate key, not sha:deadbeef#0
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "nondeterministic_replay"


async def test_host_crash_on_suspended_gate_is_not_divergence(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A host crash/timeout (emitted=[]) on a run suspended at a gate must report
    the REAL infra cause, not a fabricated ``nondeterministic_replay`` — the
    divergence check runs only on a real replay, after the raised branch."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)  # wake 1: opens the gate, suspends
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"

    timeout = HostOutcome(
        kind="raised", error_kind="script_host_timeout", error_repr="deadline", emitted=[]
    )
    with mock.patch("aios.workflows.step.run_script_host", new=AsyncMock(return_value=timeout)):
        await run_workflow_step(run_id)  # wake 2: host killed before re-emitting the gate

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "script_host_timeout"


async def test_spawn_failure_is_transient_not_terminal(wf_runtime: asyncpg.Pool[Any]) -> None:
    """``script_host_spawn_failed`` (EAGAIN/ENOMEM at fork) is a worker-infra fault,
    not the script's: the step raises (so the sweep retries) and never terminally
    errors the run."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    spawn_failed = HostOutcome(
        kind="raised", error_kind="script_host_spawn_failed", error_repr="EAGAIN", emitted=[]
    )
    with (
        mock.patch("aios.workflows.step.run_script_host", new=AsyncMock(return_value=spawn_failed)),
        pytest.raises(RuntimeError),
    ):
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status != "errored"  # retriable, not terminal
    assert not any(e.type == "run_completed" for e in events)


async def test_early_gate_signal_triggers_self_rewake(wf_runtime: asyncpg.Pool[Any]) -> None:
    """A resume delivered BEFORE the gate's call_started is journaled (its call_key
    is derivable) is harvested on a prompt self-wake, not only by the ~30s sweep."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    # Pre-deliver the resume for the not-yet-opened gate (deterministic key #0).
    gate_key = CallKeyer().next("gate", {"q": "ok?"})
    async with pool.acquire() as conn:
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=gate_key, kind="gate_resume", result="early"
        )

    with mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()) as rewake:
        await run_workflow_step(run_id)  # opens the gate, sees the signal, self-wakes
        assert rewake.await_count == 1  # a prompt self-wake, not a 30s sweep wait
        await run_workflow_step(run_id)  # the self-wake's step harvests + completes

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"answer": "early"}
