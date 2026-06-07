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
        input={"q": "hi"},
    )
    assert created is True
    # A replay (same id) is a rowcount-0 no-op — no double row, no double input.
    again = await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        parent_run_id=run_id,
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
    assert user_msgs == 1  # input delivered exactly once


async def test_child_session_origin_and_parent_round_trip(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    cid = child_session_id(run_id, "sha:y#0")
    await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        parent_run_id=run_id,
        input="hi",
    )
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


async def _spawn_child(pool: asyncpg.Pool[Any], agent_id: str, ordinal: str) -> tuple[str, str]:
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
        input="hi",
    )
    return run_id, cid


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


async def test_return_writes_marker_archives_and_wakes_parent(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d2#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        res = await workflow_completion.return_handler(cid, {"value": {"answer": 42}})
    assert res == {"status": "returned"}
    wake.assert_awaited_once_with(run_id)

    async with pool.acquire() as conn:
        marker = await db_queries.read_workflow_child_done(conn, cid, account_id="acc_wf")
    assert marker is not None
    assert marker["is_error"] is False and marker["result"] == {"answer": 42}
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert child.archived_at is not None  # self-archived — genuinely terminal


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
