"""B1.4 + B1.5 — run_workflow_step end to end against a real Postgres.

The headline is the gate round-trip: a run suspends at a gate, an external resume
delivers a value, and the next wake replays-with-memo past the gate to completion
— with the journal staying a clean [run_started, call_started, call_result,
run_completed] and every step idempotent.

``defer_run_wake`` (the procrastinate enqueue) is patched out; the step is driven
directly, which is exactly the surface under test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import workflows as wf_service
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
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
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


async def _spawn_child(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    ordinal: str,
    *,
    output_schema: dict[str, Any] | None = None,
) -> tuple[str, str]:
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
        output_schema=output_schema,
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


# ─── cancel — user-requested termination via the cancel signal ───────────────


async def test_cancel_suspended_run_finalizes_cancelled(wf_runtime: asyncpg.Pool[Any]) -> None:
    """A suspended run is cancelled on its next wake: the cancel API records a signal
    (journal untouched, run still suspended), then the wake harvests it under the lock
    and finalizes ``cancelled`` with a non-error ``run_completed`` bookend (so a live
    ``/stream`` closes on the event). A further wake / re-cancel is an idempotent
    no-op."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)  # → suspended at the gate
    async with pool.acquire() as conn:
        suspended = await wf_queries.get_run_for_step(conn, run_id)
    assert suspended is not None and suspended.status == "suspended"

    # Request cancel: signal recorded, run still suspended (the flip lands on the wake).
    run = await wf_service.cancel_run(pool, run_id=run_id, account_id="acc_wf")
    assert run.status == "suspended"
    assert [t for _s, t, _k in await _events(pool, run_id)] == ["run_started", "call_started"]

    # Wake: harvest the cancel → cancelled.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "cancelled" and run.output is None
    rc = events[-1]
    assert rc.type == "run_completed"
    assert rc.payload["cancelled"] is True and rc.payload["is_error"] is False

    # Idempotent: a wake on a cancelled run is a no-op; a re-cancel returns it unchanged.
    before = await _events(pool, run_id)
    await run_workflow_step(run_id)
    again = await wf_service.cancel_run(pool, run_id=run_id, account_id="acc_wf")
    assert again.status == "cancelled"
    assert await _events(pool, run_id) == before  # journal frozen


async def test_cancel_pending_run_finalizes_before_it_starts(wf_runtime: asyncpg.Pool[Any]) -> None:
    """A run cancelled before its first wake never runs the script: its journal is a
    lone terminal ``run_completed`` and the status is ``cancelled``."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)  # pending — no events yet
    run = await wf_service.cancel_run(pool, run_id=run_id, account_id="acc_wf")
    assert run.status == "pending"

    await run_workflow_step(run_id)  # harvest cancel before run_started
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "cancelled"
    assert [t for _s, t, _k in await _events(pool, run_id)] == ["run_completed"]


async def test_cancel_is_noop_on_an_already_terminal_run(wf_runtime: asyncpg.Pool[Any]) -> None:
    """Cancelling a run that already completed is a pure no-op: it returns the run
    unchanged, records no ``cancel`` signal, and a subsequent wake stays a no-op."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    await run_workflow_step(run_id)  # → completed
    before = await _events(pool, run_id)

    run = await wf_service.cancel_run(pool, run_id=run_id, account_id="acc_wf")
    assert run.status == "completed"  # returned unchanged
    async with pool.acquire() as conn:
        signals = await wf_queries.list_run_signals(conn, run_id)
    assert all(s.kind != "cancel" for s in signals)  # no cancel marker written
    await run_workflow_step(run_id)
    assert await _events(pool, run_id) == before  # journal unchanged


# ─── parent_run_id filters — a run's child runs + child agent-sessions ────────


async def test_children_listable_by_parent_run_id(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A run's children are listable by ``parent_run_id`` on both surfaces: child
    agent-sessions via ``list_sessions`` and nested child runs via ``list_wf_runs``,
    each scoped to the parent and excluding an unrelated run's children."""
    pool = wf_runtime
    parent_id, child_sid = await _spawn_child(pool, wf_agent_id, "sha:pf#0")
    # An unrelated foreground session (parent_run_id is None) that must be excluded.
    other = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )

    async with pool.acquire() as conn:
        scoped = await db_queries.list_sessions(
            conn, account_id="acc_wf", parent_run_id=parent_id, limit=50
        )
        parent_run = await wf_queries.get_run_for_step(conn, parent_id)
        assert parent_run is not None
        # A nested child run under the parent (the parent run itself is parentless).
        child_run = await wf_queries.insert_wf_run(
            conn,
            account_id="acc_wf",
            workflow_id=parent_run.workflow_id,
            environment_id="env_wf",
            script="async def main(i):\n    return 1",
            script_sha="sha",
            parent_run_id=parent_id,
        )
        child_runs = await wf_queries.list_wf_runs(
            conn, account_id="acc_wf", parent_run_id=parent_id, limit=50
        )

    scoped_ids = [s.id for s in scoped]
    assert scoped_ids == [child_sid]  # only the parent's child session
    assert other.id not in scoped_ids  # the foreground session is excluded
    assert [r.id for r in child_runs] == [child_run.id]  # only the nested child run


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
    archive the child — responding is not terminating. Archive-on-quiescence reclaims
    the child only at its next idle end-of-step, so right after responding (still mid-turn,
    a tool_result just landed) the child is unarchived."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d2#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        res = await workflow_completion.return_handler(
            cid, {"request_id": "sha:d2#0", "value": {"answer": 42}}
        )
    assert res == {"status": "returned"}
    wake.assert_awaited_once_with(run_id)

    async with pool.acquire() as conn:
        response = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:d2#0"
        )
    assert response is not None
    assert response["is_error"] is False and response["result"] == {"answer": 42}
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    # NOT archived by the handler — archive-on-quiescence reclaims at the next idle step.
    assert child.archived_at is None


async def test_error_marker_carries_message(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion

    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:d2e#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        res = await workflow_completion.error_handler(
            cid, {"request_id": "sha:d2e#0", "message": "nope"}
        )
    assert res == {"status": "errored"}
    async with pool.acquire() as conn:
        marker = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:d2e#0"
        )
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
        res = await workflow_completion.return_handler(fg.id, {"request_id": "x", "value": 1})
    assert isinstance(res, ToolResult) and res.is_error
    wake.assert_not_awaited()  # never signal a NULL parent run
    async with pool.acquire() as conn:
        marker = await db_queries.read_request_response(
            conn, fg.id, account_id="acc_wf", request_id="x"
        )
    assert marker is None


async def test_request_is_answered_exactly_once(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A request is answered exactly once. The first response wins and wakes the
    caller; a second answer to the now-closed request is rejected as
    ``unknown_request`` (it's no longer open). The genuinely-concurrent race is
    still caught one layer down by ``write_response_if_absent``."""
    from aios.tools import workflow_completion
    from aios.tools.registry import ToolResult

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:once#0")
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        r1 = await workflow_completion.return_handler(
            cid, {"request_id": "sha:once#0", "value": "first"}
        )
        r2 = await workflow_completion.error_handler(
            cid, {"request_id": "sha:once#0", "message": "too late"}
        )
    assert r1 == {"status": "returned"}
    assert isinstance(r2, ToolResult) and r2.is_error  # request already answered → not open
    wake.assert_awaited_once_with(run_id)  # only the first response woke the caller

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_response'",
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
        "function": {
            "name": "return",
            "arguments": _json.dumps({"request_id": "sha:rd#0", "value": "ok"}),
        },
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
        marker = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:rd#0"
        )
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
        await workflow_completion.return_handler(cid, {"request_id": call_key, "value": "r"})

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


async def _open_request_id(pool: asyncpg.Pool[Any], session_id: str) -> str:
    """The child's single open request id (= the agent() call_key it answers)."""
    async with pool.acquire() as conn:
        ids = await db_queries.get_open_request_ids(conn, session_id, account_id="acc_wf")
    return ids[0]


async def test_agent_round_trip_harvest_completes_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = f"async def main(input):\n    await agent({wf_agent_id!r}, 'go')\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    rid = await _open_request_id(pool, child_id)

    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(
            child_id, {"request_id": rid, "value": "child-result"}
        )

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
    rid = await _open_request_id(pool, child_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.error_handler(child_id, {"request_id": rid, "message": "boom"})

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
    rid = await _open_request_id(pool, child_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.error_handler(child_id, {"request_id": rid, "message": "nope"})

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
    rid = await _open_request_id(pool, child_id)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(
            child_id, {"request_id": rid, "value": "the-answer"}
        )

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
    rid = await _open_request_id(pool, child_id)  # capture before the erroring path answers it

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
        response = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=rid
        )
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
        await workflow_completion.return_handler(cid, {"request_id": "sha:prec#0", "value": "real"})

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
            "AND kind = 'lifecycle' AND data->>'event' = 'request_response'",
            cid,
            "acc_wf",
        )
    assert len(rows) == 1  # still exactly one response
    response = db_queries.parse_jsonb(rows[0]["data"])
    assert response["is_error"] is False and response["result"] == "real"  # the return() won


# ─── R4 — the session-quiescence totality guard (nudge → no_return) ───────────


async def _idle_assistant_turn(pool: asyncpg.Pool[Any], session_id: str) -> dict[str, Any]:
    """An assistant message that reacts to everything so far and calls no tools —
    i.e. a pure-text turn that would leave the session idle (the guard's trigger)."""
    child = await sessions_service.get_session_basic(pool, session_id, account_id="acc_wf")
    return {"role": "assistant", "content": "thinking…", "reacting_to": child.last_event_seq}


async def test_idle_with_open_request_is_nudged(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that ends a tool-call-free turn while still owing a request is nudged
    in the same transaction as the idling assistant event — so it never goes idle
    with a debt — and the nudge (a user message) keeps it active for another turn."""
    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:n#0")

    assistant = await _idle_assistant_turn(pool, cid)
    nudged, caller = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, assistant, account_id="acc_wf", parent_run_id=_run_id
    )
    assert nudged and caller is None

    async with pool.acquire() as conn:
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        nudges = await db_queries.count_request_nudges(
            conn, cid, account_id="acc_wf", request_id="sha:n#0"
        )
        status = await db_queries.derive_session_status(conn, cid, account_id="acc_wf")
    assert open_ids == ["sha:n#0"]  # still open — nudging is not answering
    assert nudges == 1
    assert status == "active"  # the nudge user message keeps it alive, no idle window


async def test_quiescence_guard_is_noop_for_non_child(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The guard is a strict no-op for an ordinary (non-child) session: it cannot
    owe a request, so the assistant is appended, the session goes idle, no nudge —
    and the parent_run_id gate skips the open-request scan entirely."""
    pool = wf_runtime
    fg = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )
    assistant = await _idle_assistant_turn(pool, fg.id)
    nudged, caller = await sessions_service.append_assistant_and_guard_quiescence(
        pool, fg.id, assistant, account_id="acc_wf", parent_run_id=None
    )
    assert not nudged and caller is None
    async with pool.acquire() as conn:
        status = await db_queries.derive_session_status(conn, fg.id, account_id="acc_wf")
        nudge_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'message' AND role = 'user'",
            fg.id,
            "acc_wf",
        )
    assert status == "idle"  # an ordinary session is free to rest
    assert nudge_msgs == 0  # no nudge injected


async def test_open_request_is_auto_errored_after_nudge_budget(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """After REQUEST_NUDGE_BUDGET nudges, a still-unanswered request is auto-errored
    with a monotonic no_return response — so totality holds even for a model that
    ignores every nudge — and the session is then free to go idle."""
    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:nr#0")

    for _ in range(sessions_service.REQUEST_NUDGE_BUDGET):
        nudged, _caller = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
            parent_run_id=run_id,
        )
        assert nudged  # still under budget — keep nudging

    # Budget spent: the next pure-text turn auto-errors the request instead.
    nudged, caller = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, await _idle_assistant_turn(pool, cid), account_id="acc_wf", parent_run_id=run_id
    )
    assert not nudged
    assert caller == run_id  # auto-errored → the caller run is woken to harvest the no_return

    async with pool.acquire() as conn:
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:nr#0"
        )
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        status = await db_queries.derive_session_status(conn, cid, account_id="acc_wf")
    assert resp is not None and resp["is_error"] is True and resp["error"] == {"kind": "no_return"}
    assert open_ids == []  # answered (by the backstop) → no longer open
    assert status == "idle"  # nothing owed → free to rest


async def test_non_answering_child_resolves_run_via_agent_no_return_error(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """R4 headline — a child that never answers does not hang the run. After the
    nudge budget the request is auto-errored (no_return); the run harvests it and
    the script's ``agent()`` raises ``AgentNoReturnError``, which the workflow can
    catch and continue past."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentNoReturnError:\n"
        "        return 'gave-up'\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)

    # The child ignores every nudge: BUDGET nudges, then the auto-error turn.
    for _ in range(sessions_service.REQUEST_NUDGE_BUDGET + 1):
        await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            child_id,
            await _idle_assistant_turn(pool, child_id),
            account_id="acc_wf",
            parent_run_id=run_id,
        )

    await run_workflow_step(run_id)  # harvest no_return -> AgentNoReturnError -> caught -> return
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is True and cr.payload["error"] == {"kind": "no_return"}
    assert run is not None and run.status == "completed" and run.output == "gave-up"


# ─── R5 — archived/deleted-child totality gap + the archived-session guard ────


async def test_archived_session_wake_is_a_noop(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A wake for an archived (or deleted) session is an idempotent no-op — the
    run_session_step entry guard returns before any append, so a stray wake (a
    sweep racing an archive, a future reclaim) never crashes on append_event's
    archived guard. Mirrors run_workflow_step's terminal early-return."""
    from aios.harness import runtime
    from aios.harness.loop import run_session_step

    pool = wf_runtime
    sess = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )
    async with pool.acquire() as conn:
        await db_queries.archive_session(conn, sess.id, account_id="acc_wf")
        before = await conn.fetchval("SELECT last_event_seq FROM sessions WHERE id = $1", sess.id)

    prev_reg = runtime.task_registry
    runtime.task_registry = None  # the guard must return before require_task_registry
    try:
        await run_session_step(sess.id)  # must NOT raise
    finally:
        runtime.task_registry = prev_reg

    async with pool.acquire() as conn:
        after = await conn.fetchval("SELECT last_event_seq FROM sessions WHERE id = $1", sess.id)
    assert after == before  # nothing appended — a clean no-op


# ─── archive-on-quiescence: the general reclaim hook + its workflow-child use ──


async def test_reclaim_session_if_idle_archives_only_when_idle(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """``reclaim_session_if_idle`` is the atomic conditional behind archive-on-quiescence:
    it archives an idle session, no-ops on an active one (a stimulus racing the idle
    transition flips ``_SESSION_ACTIVE_EXPR`` and wins), and no-ops once archived."""
    pool = wf_runtime

    # Active session (an unreacted user stimulus) → reclaim is a no-op.
    active = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
        archive_when_idle=True,
    )
    await sessions_service.append_user_message(pool, active.id, "hi", account_id="acc_wf")
    async with pool.acquire() as conn:
        assert (
            await db_queries.derive_session_status(conn, active.id, account_id="acc_wf") == "active"
        )
        assert (
            await db_queries.reclaim_session_if_idle(conn, active.id, account_id="acc_wf") is False
        )
    assert (
        await sessions_service.get_session_basic(pool, active.id, account_id="acc_wf")
    ).archived_at is None

    # Idle session → archived; a second call is an idempotent no-op.
    idle = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
        archive_when_idle=True,
    )
    async with pool.acquire() as conn:
        assert await db_queries.derive_session_status(conn, idle.id, account_id="acc_wf") == "idle"
        assert await db_queries.reclaim_session_if_idle(conn, idle.id, account_id="acc_wf") is True
        assert await db_queries.reclaim_session_if_idle(conn, idle.id, account_id="acc_wf") is False
    assert (
        await sessions_service.get_session_basic(pool, idle.id, account_id="acc_wf")
    ).archived_at is not None


async def test_child_reclaimed_on_quiescence_and_parent_still_harvests(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The workflow-child use of the general feature: a child is born ephemeral
    (``insert_child_session`` sets ``archive_when_idle`` TRUE), answers its request,
    and on its first genuine idle is reclaimed — yet the parent's harvest still returns
    the answer, because ``derive_response`` reads the response *event* (which survives
    archival), not the live row."""
    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:reclaim#0")

    # Born ephemeral — the workflow-child wiring, no per-call plumbing.
    born = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert born.archive_when_idle is True

    # It answers its request, then ends a tool-free turn → idle owing nothing. The guard
    # does not nudge an already-answered child, so it is free to rest.
    async with pool.acquire() as conn:
        await db_queries.write_response_if_absent(
            conn,
            cid,
            account_id="acc_wf",
            request_id="sha:reclaim#0",
            is_error=False,
            result={"answer": 42},
            error=None,
        )
    nudged, caller = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, await _idle_assistant_turn(pool, cid), account_id="acc_wf", parent_run_id=run_id
    )
    assert not nudged and caller is None

    # End-of-step reclaim (what loop._run_step does for an archive_when_idle session).
    assert await sessions_service.reclaim_session_if_idle(pool, cid, account_id="acc_wf") is True
    child = await sessions_service.get_session_basic(pool, cid, account_id="acc_wf")
    assert child.archived_at is not None

    # The parent harvest still resolves to the answer — archival did not lose it.
    async with pool.acquire() as conn:
        resp = await db_queries.derive_response(
            conn, cid, account_id="acc_wf", request_id="sha:reclaim#0"
        )
    assert resp is not None and resp["is_error"] is False and resp["result"] == {"answer": 42}


async def test_archive_when_idle_persists_across_launch_surfaces(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The launch flag round-trips on every surface: SessionCreate carries it onto the
    row (default False), and a session_template stores + updates it — the value the
    per-chat resolver copies into ``create_session``."""
    from aios.services import session_templates as st_service

    pool = wf_runtime
    fg = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
        archive_when_idle=True,
    )
    assert fg.archive_when_idle is True
    fg_default = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )
    assert fg_default.archive_when_idle is False

    tmpl = await st_service.create_session_template(
        pool,
        account_id="acc_wf",
        name="ephemeral-tmpl",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=None,
        vault_ids=[],
        memory_store_ids=[],
        metadata={},
        archive_when_idle=True,
    )
    assert tmpl.archive_when_idle is True
    assert (
        await st_service.get_session_template(pool, tmpl.id, account_id="acc_wf")
    ).archive_when_idle is True
    updated = await st_service.update_session_template(
        pool, tmpl.id, account_id="acc_wf", archive_when_idle=False
    )
    assert updated.archive_when_idle is False


async def test_operator_archived_child_resolves_run_as_child_gone(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """An operator archiving a *running* child before it answers must not hang the
    run: derive_response sees the child is no longer live and resolves the call as a
    catchable AgentError(child_gone)."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    async with pool.acquire() as conn:  # operator archives the child mid-flight
        await db_queries.archive_session(conn, child_id, account_id="acc_wf")

    await run_workflow_step(run_id)  # harvest -> child_gone -> AgentError -> caught
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"caught": True, "kind": "child_gone"}


async def test_operator_deleted_child_resolves_run_as_child_gone(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Same totality guarantee when the child is hard-deleted (its events are gone):
    the run resolves from the child's absence, not a written response."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return e.kind\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    async with pool.acquire() as conn:
        await db_queries.delete_session(conn, child_id, account_id="acc_wf")

    await run_workflow_step(run_id)  # harvest -> child_gone -> AgentError -> caught
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed" and run.output == "child_gone"


# ─── G — parallel() spawns a child fan-out and collects results ───────────────


async def _children_of(pool: asyncpg.Pool[Any], run_id: str) -> list[str]:
    """All agent children a run has spawned, in call_started (emission) order."""
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    return [e.payload["child_session_id"] for e in events if e.type == "call_started"]


async def test_parallel_spawns_fanout_and_collects_results_in_order(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """parallel() over two agent() calls spawns BOTH children in one step (a single
    frontier), then a later step harvests both and the run completes with the
    results in thunk order."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent({wf_agent_id!r}, 'a'),"
        f" lambda: agent({wf_agent_id!r}, 'b')])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # drive parallel -> spawn both children + suspend
    children = await _children_of(pool, run_id)
    assert len(children) == 2  # fanned out at once

    for cid, value in zip(children, ["ra", "rb"], strict=True):
        rid = await _open_request_id(pool, cid)
        with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
            await workflow_completion.return_handler(cid, {"request_id": rid, "value": value})

    await run_workflow_step(run_id)  # harvest both -> replay past parallel -> complete
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == ["ra", "rb"]  # branch order preserved


async def test_parallel_barrier_yields_none_for_an_errored_child(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """An errored branch (uncaught) contributes None to the results list; the other
    branch's value is unaffected and the run completes."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent({wf_agent_id!r}, 'a'),"
        f" lambda: agent({wf_agent_id!r}, 'b')])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    children = await _children_of(pool, run_id)
    rid0 = await _open_request_id(pool, children[0])
    rid1 = await _open_request_id(pool, children[1])
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(children[0], {"request_id": rid0, "value": "ok"})
        await workflow_completion.error_handler(
            children[1], {"request_id": rid1, "message": "nope"}
        )

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == ["ok", None]  # the errored branch → None, barrier still returns


# ─── H — runaway caps + the per-call wall-clock deadline ──────────────────────


async def test_lifetime_agent_cap_errors_atomically_before_spawning(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A fan-out that would push the run past its lifetime agent-call cap errors
    BEFORE spawning any of that step's children — no partial fan-out of orphans."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 2})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent({wf_agent_id!r}, str(i)) for i in range(3)])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # 3 > cap 2 → error, nothing spawned

    assert await _children_of(pool, run_id) == []  # no call_started journaled
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        # the invariant at its real surface: no child session rows exist for the run
        orphans = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert orphans == 0  # atomic: no partial fan-out of orphan children
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "too_many_agents"
    assert "2-agent call cap" in events[-1].payload["output"]


async def test_agent_cap_at_boundary_is_allowed(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Exactly at the cap is allowed: a fan-out of N with cap N spawns all N children
    and suspends (the boundary is strict ``>``, mirroring H2's at-cap host test)."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 3})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent({wf_agent_id!r}, str(i)) for i in range(3)])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # 3 == cap 3 → allowed

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"
    assert len(await _children_of(pool, run_id)) == 3  # all spawned


async def test_lowering_cap_mid_flight_does_not_kill_a_harvest_only_step(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """H1 gates NEW spawns only: a harvest-only re-suspend (no new agents) must never
    error, even after the cap is lowered below the already-spawned count — otherwise a
    config reduction would retroactively kill in-flight runs and orphan their
    children."""
    from aios.config import get_settings
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent({wf_agent_id!r}, str(i)) for i in range(2)])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn 2 under the default (high) cap
    children = await _children_of(pool, run_id)
    assert len(children) == 2

    # Now lower the cap below the 2 already-spawned, and resolve only one child so the
    # next step is harvest-only (the other stays inflight → zero new agent caps).
    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 1})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)
    rid0 = await _open_request_id(pool, children[0])
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(children[0], {"request_id": rid0, "value": "r0"})

    await run_workflow_step(run_id)  # harvest child0; child1 inflight; new_agent_caps == []
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"  # NOT errored — no new spawns


async def test_agent_call_times_out_when_child_never_responds(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that never responds past its wall-clock deadline resolves the agent()
    call as a catchable AgentError(timeout) — totality even for a child that never
    goes idle (so the quiescence nudge never fires). The child is NOT terminated:
    the timeout writes a response, it doesn't end the target (responding ≠ ending)."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        return await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'timed_out': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    rid = await _open_request_id(pool, child_id)

    # Age the call_started past the (default 1h) deadline — deterministic, no sleep.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_run_events SET created_at = created_at - interval '2 hours' "
            "WHERE run_id = $1 AND type = 'call_started'",
            run_id,
        )

    await run_workflow_step(run_id)  # harvest: past deadline → timeout response → resolve

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        resp = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=rid
        )
        child = await db_queries.get_session_bare(conn, child_id, account_id="acc_wf")
    assert run is not None and run.status == "completed"
    assert run.output == {"timed_out": "timeout"}
    assert resp is not None and resp["is_error"] is True and resp["error"] == {"kind": "timeout"}
    assert child.archived_at is None  # left running — responding ≠ terminating


async def test_past_deadline_child_that_already_responded_keeps_its_real_response(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that answered before the deadline harvest keeps its real value: the
    harvest's first derive_response is non-None, so the timeout branch is never
    entered and no clobbering timeout is written (the deadline never overrides an
    existing response)."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        return await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'timed_out': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    child_id = await _child_id_of(pool, run_id)
    rid = await _open_request_id(pool, child_id)

    # The child answers for real, THEN we also age the call past the deadline.
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(child_id, {"request_id": rid, "value": "real"})
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_run_events SET created_at = created_at - interval '2 hours' "
            "WHERE run_id = $1 AND type = 'call_started'",
            run_id,
        )

    await run_workflow_step(run_id)  # derive_response already non-None → no timeout written
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        resp = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=rid
        )
    assert run is not None and run.status == "completed" and run.output == "real"
    assert resp is not None and resp["is_error"] is False and resp["result"] == "real"


async def test_past_deadline_gone_child_resolves_as_child_gone_not_timeout(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that is BOTH archived AND past its deadline resolves as child_gone, not
    timeout: derive_response folds in liveness, so a gone child returns non-None
    before the deadline branch — and even a child archived in the derive→write window
    surfaces as child_gone (write_response_if_absent's NotFoundError is absorbed, no
    crash)."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        return await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    child_id = await _child_id_of(pool, run_id)

    # Archive the child (it never answered) AND age the call past the deadline.
    async with pool.acquire() as conn:
        await db_queries.archive_session(conn, child_id, account_id="acc_wf")
        await conn.execute(
            "UPDATE wf_run_events SET created_at = created_at - interval '2 hours' "
            "WHERE run_id = $1 AND type = 'call_started'",
            run_id,
        )

    await run_workflow_step(run_id)  # gone wins over the deadline — no timeout, no crash
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"kind": "child_gone"}


async def test_child_archived_in_timeout_write_window_resolves_as_child_gone(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The narrow TOCTOU the H3 timeout guards against: a child live at the harvest's
    derive (None) but archived before the timeout write. write_response_if_absent
    raises NotFoundError against the gone row; the helper absorbs it and the re-derive
    resolves the call as child_gone — never a step crash."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        return await agent({wf_agent_id!r}, 'go')\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn the child
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_run_events SET created_at = created_at - interval '2 hours' "
            "WHERE run_id = $1 AND type = 'call_started'",
            run_id,
        )

    # Inject the race: the first derive observes the child live (returns None) but
    # archives it in the same breath, so the subsequent timeout write hits a gone row.
    real_derive = db_queries.derive_response
    calls = {"n": 0}

    async def racing_derive(conn: Any, sid: str, *, account_id: str, request_id: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            await db_queries.archive_session(conn, sid, account_id=account_id)
            return None  # "live at the snapshot" — but now archived under us
        return await real_derive(conn, sid, account_id=account_id, request_id=request_id)

    monkeypatch.setattr(db_queries, "derive_response", racing_derive)
    await run_workflow_step(run_id)  # must NOT crash on the gone-row write

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"kind": "child_gone"}  # resolved gracefully, no crash


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


# ─── Block 3 surface: the services.workflows facade (create + by-nonce resume) ─


async def test_service_create_and_resume_by_nonce_roundtrip(wf_runtime: asyncpg.Pool[Any]) -> None:
    """The public service path: create a workflow + run, drive to a gate, resume by
    the gate's NONCE (not call_key), and complete."""
    pool = wf_runtime
    wf = await wf_service.create_workflow(
        pool, account_id="acc_wf", name="gate-demo", script=_GATE_SCRIPT
    )
    run = await wf_service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
    )

    await run_workflow_step(run.id)  # → suspends at the gate
    assert (await wf_service.get_run(pool, run.id, account_id="acc_wf")).status == "suspended"
    events = await wf_service.list_run_events(pool, run.id, account_id="acc_wf")
    call_started = next(e for e in events if e.type == "call_started")
    nonce = call_started.payload["gate_nonce"]
    assert isinstance(nonce, str) and nonce

    await wf_service.resume_gate_by_nonce(
        pool, run_id=run.id, account_id="acc_wf", gate_nonce=nonce, result="yes"
    )
    await run_workflow_step(run.id)  # harvest the signal → replay past the gate → complete
    done = await wf_service.get_run(pool, run.id, account_id="acc_wf")
    assert done.status == "completed" and done.output == {"answer": "yes"}

    # The gate is now resolved — re-resuming with the same (valid) nonce 404s
    # ("no OPEN gate matches") rather than writing an orphaned signal nothing harvests.
    with pytest.raises(NotFoundError):
        await wf_service.resume_gate_by_nonce(
            pool, run_id=run.id, account_id="acc_wf", gate_nonce=nonce, result="again"
        )


async def test_service_resume_by_nonce_rejects_bad_nonce_and_cross_tenant(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    pool = wf_runtime
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_intruder', 'acc_wf', FALSE, 'intruder')"
        )
    wf = await wf_service.create_workflow(
        pool, account_id="acc_wf", name="gate-demo", script=_GATE_SCRIPT
    )
    run = await wf_service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
    )
    await run_workflow_step(run.id)  # → suspended at the gate
    events = await wf_service.list_run_events(pool, run.id, account_id="acc_wf")
    nonce = next(e for e in events if e.type == "call_started").payload["gate_nonce"]

    # Wrong nonce → 404 (no gate matches).
    with pytest.raises(NotFoundError):
        await wf_service.resume_gate_by_nonce(
            pool, run_id=run.id, account_id="acc_wf", gate_nonce="not-a-real-nonce", result="x"
        )
    # Cross-tenant resume → 404 on the scope check, BEFORE the (correct) nonce is read.
    with pytest.raises(NotFoundError):
        await wf_service.resume_gate_by_nonce(
            pool, run_id=run.id, account_id="acc_intruder", gate_nonce=nonce, result="x"
        )
    with pytest.raises(NotFoundError):
        await wf_service.get_run(pool, run.id, account_id="acc_intruder")
    # The run is untouched — still suspended, resumable by its owner.
    assert (await wf_service.get_run(pool, run.id, account_id="acc_wf")).status == "suspended"


# ─── B3 slice 2 — agent() structured output (output_schema) ───────────────────

_OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


async def test_get_request_output_schema_is_per_request(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The schema is keyed by request_id: a child owing several requests with
    different schemas resolves each independently (unknown id / schema-less → None).
    The workflow path is 1:1 today, but the query is per-request by design."""
    pool = wf_runtime
    schema_b = {"type": "number"}
    run_id, cid = await _spawn_child(pool, wf_agent_id, "req:a", output_schema=_OBJ_SCHEMA)
    async with pool.acquire() as conn:
        for rid, schema in (("req:b", schema_b), ("req:none", None)):
            request: dict[str, Any] = {"request_id": rid, "caller": {"kind": "run", "id": run_id}}
            if schema is not None:
                request["output_schema"] = schema
            await db_queries.append_event(
                conn,
                account_id="acc_wf",
                session_id=cid,
                kind="message",
                data={"role": "user", "content": "x", "metadata": {"request": request}},
            )
        get = db_queries.get_request_output_schema
        assert await get(conn, cid, request_id="req:a") == _OBJ_SCHEMA
        assert await get(conn, cid, request_id="req:b") == schema_b
        assert await get(conn, cid, request_id="req:none") is None
        assert await get(conn, cid, request_id="req:nope") is None


async def test_return_enforces_output_schema(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A non-conforming value bounces back as a tool error with NO response written
    (the request stays open for the child to retry); a conforming value is answered."""
    from aios.tools import workflow_completion
    from aios.tools.registry import ToolResult

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "se:1", output_schema=_OBJ_SCHEMA)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        bad = await workflow_completion.return_handler(
            cid,
            {"request_id": "se:1", "value": {"answer": 42}},  # answer must be a string
        )
    assert isinstance(bad, ToolResult) and bad.is_error and "schema" in bad.content.lower()
    wake.assert_not_awaited()  # nothing answered → caller not woken
    async with pool.acquire() as conn:
        assert (
            await db_queries.read_request_response(
                conn, cid, account_id="acc_wf", request_id="se:1"
            )
            is None  # request stays open for a retry
        )

    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake:
        good = await workflow_completion.return_handler(
            cid, {"request_id": "se:1", "value": {"answer": "yes"}}
        )
    assert good == {"status": "returned"}
    wake.assert_awaited_once_with(run_id)
    async with pool.acquire() as conn:
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="se:1"
        )
    assert resp is not None and resp["result"] == {"answer": "yes"}


async def test_malformed_output_schema_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A malformed output_schema fails the run cleanly (author-facing bad_agent_call)
    and never spawns a child."""
    pool = wf_runtime
    bad_schema = {"type": "not-a-real-type"}
    script = (
        f"async def main(input):\n"
        f"    return await agent({wf_agent_id!r}, 'hi', output_schema={bad_schema!r})\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "errored"
    completed = [e for e in events if e.type == "run_completed"]
    assert completed and completed[0].payload["error"]["kind"] == "bad_agent_call"
    assert children == 0  # rejected before any spawn


async def test_non_object_output_schema_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A non-object output_schema (a bare boolean — valid JSON Schema, but ``false``
    rejects every value and ``true`` disables enforcement) is a degenerate author input:
    fail the run cleanly as bad_agent_call rather than spawn a child that can never
    return (or one with no enforcement)."""
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, 'hi', output_schema=False)\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "errored"
    completed = [e for e in events if e.type == "run_completed"]
    assert completed and completed[0].payload["error"]["kind"] == "bad_agent_call"
    assert "object schema" in completed[0].payload["output"]
    assert children == 0  # rejected before any spawn


async def test_agent_output_schema_end_to_end(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Full slice: the spawn carries the schema (request metadata + call_started
    audit), the child must return a conforming value (a bad one is rejected with no
    response), and the run harvests the validated value as the agent() result."""
    from aios.tools import workflow_completion
    from aios.tools.registry import ToolResult

    pool = wf_runtime
    script = (
        f"async def main(input):\n"
        f"    return await agent({wf_agent_id!r}, 'task', output_schema={_OBJ_SCHEMA!r})\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "suspended"
    started = next(e for e in events if e.type == "call_started")
    request_id = started.call_key
    child_id = started.payload["child_session_id"]
    assert started.payload["output_schema"] == _OBJ_SCHEMA  # journaled for audit
    assert request_id is not None

    # The request message carries the schema (for both the model and the validator).
    async with pool.acquire() as conn:
        seen = await db_queries.get_request_output_schema(conn, child_id, request_id=request_id)
    assert seen == _OBJ_SCHEMA

    # The child must conform: a bad value is rejected, a good one answers.
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        rejected = await workflow_completion.return_handler(
            child_id, {"request_id": request_id, "value": {"answer": 7}}
        )
        assert isinstance(rejected, ToolResult) and rejected.is_error
        ok = await workflow_completion.return_handler(
            child_id, {"request_id": request_id, "value": {"answer": "done"}}
        )
    assert ok == {"status": "returned"}

    await run_workflow_step(run_id)  # harvest → complete
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"answer": "done"}


async def test_float_bearing_output_schema_spawns_and_enforces(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A schema with a decimal numeric constraint (a legitimate, common JSON Schema)
    must NOT crash the run at call_key time: output_schema is canonicalized as a string
    so the determinism float-ban (which guards input DATA) doesn't reject it. The floats
    survive the round-trip and enforcement honors them."""
    from aios.tools import workflow_completion
    from aios.tools.registry import ToolResult

    pool = wf_runtime
    schema = {"type": "number", "minimum": 1.5, "maximum": 9.5}
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, 'pick', output_schema={schema!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "suspended"  # spawned, not crashed on the float
    started = next(e for e in events if e.type == "call_started")
    child_id = started.payload["child_session_id"]
    assert started.payload["output_schema"] == schema  # floats intact in the audit stamp
    assert started.call_key is not None

    async with pool.acquire() as conn:
        seen = await db_queries.get_request_output_schema(
            conn, child_id, request_id=started.call_key
        )
    assert seen == schema  # floats survive the request round-trip

    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        bad = await workflow_completion.return_handler(
            child_id,
            {"request_id": started.call_key, "value": 0.5},  # below minimum
        )
        assert isinstance(bad, ToolResult) and bad.is_error
        ok = await workflow_completion.return_handler(
            child_id, {"request_id": started.call_key, "value": 5.0}
        )
    assert ok == {"status": "returned"}


async def test_valid_local_ref_output_schema_spawns(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A self-contained schema using ``$ref`` into ``$defs`` is valid and must spawn —
    the unresolvable-ref gate rejects only refs that DON'T resolve, not all refs."""
    pool = wf_runtime
    schema = {
        "$defs": {"Name": {"type": "string"}},
        "type": "object",
        "properties": {"name": {"$ref": "#/$defs/Name"}},
        "required": ["name"],
    }
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, 'x', output_schema={schema!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "suspended"  # valid local $ref → spawned
    assert children == 1


async def test_dangling_ref_output_schema_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A schema whose ``$ref`` doesn't resolve passes check_schema but would raise at
    the child's validation time (bricking it until the deadline). Reject it
    author-facing at the spawn gate, before any child spawns."""
    pool = wf_runtime
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}, "required": ["a"]}
    script = f"async def main(input):\n    return await agent({wf_agent_id!r}, 'x', output_schema={schema!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "errored"
    completed = [e for e in events if e.type == "run_completed"]
    assert completed and completed[0].payload["error"]["kind"] == "bad_agent_call"
    assert "reference" in completed[0].payload["output"].lower()
    assert children == 0  # rejected before any spawn


async def test_defer_wake_priority_reflects_real_origin(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Foreground protection end-to-end: against real sessions, defer_wake demotes a
    workflow background child's wake below a user-facing foreground session's, so a
    fan-out can't starve a user's message. Validates the real origin lookup → priority
    (the unit tests mock the lookup); the in-memory connector captures the priority."""
    from procrastinate.testing import InMemoryConnector

    from aios.harness.procrastinate_app import app
    from aios.services.wake import _BACKGROUND_PRIORITY, _FOREGROUND_PRIORITY, defer_wake

    pool = wf_runtime
    fg = await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )  # origin='foreground', no parent run
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "prio:1")  # origin='background'

    with app.replace_connector(InMemoryConnector()) as patched:
        await defer_wake(pool, fg.id, account_id="acc_wf", cause="message")
        await defer_wake(pool, cid, account_id="acc_wf", cause="message")
        priorities = {
            j["args"]["session_id"]: j["priority"] for j in patched.connector.jobs.values()
        }
    assert priorities[fg.id] == _FOREGROUND_PRIORITY  # real foreground → default
    assert priorities[cid] == _BACKGROUND_PRIORITY  # real background child → demoted


# ─── await_run — the await-a-completion primitive, runs backing ──────────────


async def test_await_run_returns_when_already_completed(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A terminal run is returned immediately (the first post-subscribe read sees it):
    ``done`` + the script's ``output``, no error."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    await run_workflow_step(run_id)  # pure script → completed in one wake
    resp = await wf_service.await_run(
        pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5
    )
    assert resp.done is True and resp.run_status == "completed"
    assert resp.output == 1 and resp.is_error is False and resp.error is None


async def test_await_run_surfaces_error_kind(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """An errored run returns ``is_error`` + the ``error.kind`` lifted from the
    ``run_completed`` payload (``author_exception`` for an uncaught script raise)."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    raise ValueError('boom')")
    await run_workflow_step(run_id)  # raise → errored
    resp = await wf_service.await_run(
        pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5
    )
    assert resp.done is True and resp.run_status == "errored" and resp.is_error is True
    assert resp.error == {"kind": "author_exception"}


async def test_await_run_times_out_on_non_terminal_run(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A never-stepped (``pending``) run that doesn't finish within the budget returns
    ``done=False`` with its live status — the re-poll contract, no archive/mutation."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")  # created, not stepped
    resp = await wf_service.await_run(
        pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=0.1
    )
    assert resp.done is False and resp.run_status == "pending"
    assert resp.output is None and resp.is_error is False and resp.error is None


async def test_await_run_wakes_on_completion_during_wait(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """Completion-during-subscribe: the await blocks on a pending run; a concurrent step
    drives it terminal; the run_completed notify wakes the await → it returns the record
    (not a timeout). The LISTEN-before-read ordering is what closes the race."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 7")

    async def _complete_soon() -> None:
        await asyncio.sleep(0.1)  # let await_run subscribe + first-read (pending) first
        await run_workflow_step(run_id)

    resp, _ = await asyncio.gather(
        wf_service.await_run(
            pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=10
        ),
        _complete_soon(),
    )
    assert resp.done is True and resp.run_status == "completed" and resp.output == 7


async def test_await_run_cross_tenant_404(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """The account scope is checked up front (before subscribing): a foreign account 404s."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    with pytest.raises(NotFoundError):
        await wf_service.await_run(
            pool, migrated_db_url, run_id, account_id="acc_other", timeout_seconds=1
        )
