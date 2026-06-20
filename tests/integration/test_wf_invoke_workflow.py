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

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.workflows import WfRun
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
