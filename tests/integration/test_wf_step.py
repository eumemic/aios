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

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.workflows import service
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
        with mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()):
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
    run = await service.create_run(pool, account_id="acc_wf", workflow_id=wf.id, input=input)
    return run.id


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


async def test_agent_capability_errors_in_block1(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return await agent('a1')")
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "errored"
    completed = (await _events(pool, run_id))[-1]
    assert completed[1] == "run_completed"


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
