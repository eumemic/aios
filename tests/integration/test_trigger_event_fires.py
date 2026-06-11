"""Triggers slice 2 (#819) — the reactive fire path against a real Postgres.

Drives ``run_workflow_step`` (real script host) and ``run_trigger_step``
directly, with every procrastinate defer patched out — the same surface
``test_wf_step.py`` exercises. Owns the blocking-obligation tests:

- THE coalescing regression (§1.2): two completions of one watched workflow →
  exactly two carrier rows, two distinct-keyed dispatches, two launched runs.
- parent_run_id threading on BOTH fire origins + the self-fire depth-cycle
  termination at WORKFLOW_RUN_MAX_DEPTH.
- consecutive_failures atomicity under concurrent event fires; auto-disable at
  exactly the threshold with ONE surfaced message.
- the jsonb round-trip read-acceptance regression (numeric-expanded template
  must not poison the scheduler's claim batch).
- origin-derived lifecycle (mid-flight source conversion), event-skip
  finalization, duplicate-claim no-op, the sweep readers, pin drift, and the
  one-shot workflow lifecycle + skip tombstone.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.trigger_runner import run_trigger_step
from aios.models.triggers import TriggerCreate
from aios.services import triggers as trig_service
from aios.services import workflows as wf_service
from aios.workflows import run_tools, service
from aios.workflows.step import run_workflow_step
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_RETURN_ONE = "async def main(input):\n    return {'ok': 1}\n"
_ECHO_INPUT = "async def main(input):\n    return input\n"

ACC = "acc_trig"


@pytest.fixture
async def trig_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """Pool on ``runtime.pool`` + a seeded root tenant; every defer patched out.

    ``defer_trigger_fire`` is patched where the completion hook calls it
    (``workflows.step``) — the tests harvest pending carrier rows from the DB
    instead, which is also what proves the dispatch intents are durable.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                f"VALUES ('{ACC}', NULL, TRUE, 'trig-root')"
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_trigger_fire", new=AsyncMock()),
            mock.patch("aios.harness.trigger_runner.defer_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _make_workflow(pool: asyncpg.Pool[Any], script: str = _RETURN_ONE) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id=ACC, name=f"w-{secrets.token_hex(4)}", script=script
        )
    return wf.id


async def _add_trigger(pool: asyncpg.Pool[Any], session_id: str, body: dict[str, Any]) -> str:
    echo = await trig_service.add_trigger(
        pool, session_id, TriggerCreate.model_validate(body), account_id=ACC
    )
    return echo.id


async def _complete_run_of(pool: asyncpg.Pool[Any], workflow_id: str, env_id: str) -> str:
    """Create + drive one run of ``workflow_id`` to completion; return its id."""
    run = await service.create_run(
        pool, account_id=ACC, workflow_id=workflow_id, environment_id=env_id
    )
    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        final = await wf_queries.get_wf_run(conn, run.id, account_id=ACC)
    assert final.status == "completed", final.status
    return run.id


async def _pending_refs(pool: asyncpg.Pool[Any]) -> list[queries.TriggerFireRef]:
    async with pool.acquire() as conn:
        return await queries.list_pending_trigger_run_refs(conn, older_than_seconds=0.0)


async def _carrier_rows(pool: asyncpg.Pool[Any], trigger_id: str) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT * FROM trigger_runs WHERE trigger_id = $1 ORDER BY created_at",
                trigger_id,
            )
        )


async def _seed_fake_fire(
    pool: asyncpg.Pool[Any], workflow_id: str, run_id: str
) -> list[queries.TriggerFireRef]:
    """Insert carrier rows for a synthetic completion (the run id need not
    exist — the fire then fails at the account-scoped watched-run read, which
    is exactly what the failure-counter tests need)."""
    async with pool.acquire() as conn, conn.transaction():
        return await queries.insert_run_completion_fires(
            conn, account_id=ACC, workflow_id=workflow_id, run_id=run_id, status="completed"
        )


async def _consecutive_failures(pool: asyncpg.Pool[Any], trigger_id: str) -> int | None:
    async with pool.acquire() as conn:
        result: int | None = await conn.fetchval(
            "SELECT consecutive_failures FROM triggers WHERE id = $1", trigger_id
        )
    return result


# ─── the coalescing regression + parent threading (obligations 4 + 1) ────────


async def test_two_completions_two_fires_two_runs(trig_runtime: asyncpg.Pool[Any]) -> None:
    """THE §1.2 obligation: two distinct completions of one watched workflow
    produce exactly two carrier rows and two launched runs — never one."""
    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="coal")
    watched = await _make_workflow(pool)
    target = await _make_workflow(pool, _ECHO_INPUT)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "reactor",
            "source": {"kind": "run_completion", "workflow_id": watched},
            "action": {"kind": "workflow", "workflow_id": target},
        },
    )

    run_a = await _complete_run_of(pool, watched, env.id)
    run_b = await _complete_run_of(pool, watched, env.id)

    refs = await _pending_refs(pool)
    assert len(refs) == 2
    assert len({r.trigger_run_id for r in refs}) == 2  # distinct dispatch keys

    rows = await _carrier_rows(pool, tid)
    assert [r["status"] for r in rows] == ["pending", "pending"]
    # The event writer's exact shape (resolution #3's writer test).
    for row, rid in zip(rows, [run_a, run_b], strict=True):
        event = queries.parse_jsonb(row["event"])
        assert event == {"run_id": rid, "workflow_id": watched, "status": "completed"}

    for ref in refs:
        await run_trigger_step(ref.trigger_id, trigger_run_id=ref.trigger_run_id)

    async with pool.acquire() as conn:
        launched = await conn.fetch(
            "SELECT id, parent_run_id, input FROM wf_runs WHERE workflow_id = $1 "
            "ORDER BY created_at",
            target,
        )
    assert len(launched) == 2
    # Obligation 1, half 1: the completing run is the lineage parent.
    assert [r["parent_run_id"] for r in launched] == [run_a, run_b]
    # The envelope reached the run: output rode by value.
    first_input = queries.parse_jsonb(launched[0]["input"])
    assert first_input["trigger"]["run"] == {
        "id": run_a,
        "workflow_id": watched,
        "status": "completed",
        "output": {"ok": 1},
        "error": None,
    }
    assert first_input["input"] is None

    rows = await _carrier_rows(pool, tid)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert [r["result_id"] for r in rows] == [r2["id"] for r2 in launched]
    assert await _consecutive_failures(pool, tid) == 0


async def test_self_fire_cycle_terminates_at_depth_cap(
    trig_runtime: asyncpg.Pool[Any],
) -> None:
    """Obligation 1, the loop bound: a trigger watching the workflow its own
    action launches terminates at WORKFLOW_RUN_MAX_DEPTH — the depth-11 fire
    errors BEFORE any run row exists, so no completion ever re-arms the chain."""
    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="loop")
    w = await _make_workflow(pool)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "ouroboros",
            "source": {"kind": "run_completion", "workflow_id": w},
            "action": {"kind": "workflow", "workflow_id": w},
        },
    )

    await _complete_run_of(pool, w, env.id)  # the root (depth 1)
    for _ in range(40):  # safety bound far above the expected ~20 iterations
        refs = await _pending_refs(pool)
        for ref in refs:
            await run_trigger_step(ref.trigger_id, trigger_run_id=ref.trigger_run_id)
        async with pool.acquire() as conn:
            active = [
                r["id"] for r in await conn.fetch("SELECT id FROM wf_runs WHERE status = 'pending'")
            ]
        for rid in active:
            await run_workflow_step(rid)
        if not refs and not active:
            break
    else:
        pytest.fail("self-fire loop did not quiesce")

    async with pool.acquire() as conn:
        total_runs = await conn.fetchval("SELECT count(*) FROM wf_runs WHERE workflow_id = $1", w)
    assert total_runs == service.WORKFLOW_RUN_MAX_DEPTH  # 10 — none past the cap

    rows = await _carrier_rows(pool, tid)
    assert len(rows) == 10  # one fire per completion
    assert [r["status"] for r in rows] == ["ok"] * 9 + ["error"]
    assert "WorkflowRunDepthExceededError" in rows[-1]["error_summary"]


async def test_timer_fire_threads_owner_lineage(trig_runtime: asyncpg.Pool[Any]) -> None:
    """Obligation 1, half 2: a timer fire on a workflow-child owner inherits
    the owner session's own parent run (closing the depth-laundering bypass —
    a past-fire_at one-shot is create_run with a 0s delay)."""
    from aios.models.attenuation import Surface
    from aios.services import sessions as sessions_service

    pool = trig_runtime
    agent, env, _ = await seed_agent_env_session(pool, account_id=ACC, prefix="lineage")
    parent_wf = await _make_workflow(pool)
    target = await _make_workflow(pool)
    parent_run = await service.create_run(
        pool, account_id=ACC, workflow_id=parent_wf, environment_id=env.id
    )
    child_id = "sess_" + "0" * 22 + "TRIG"
    await sessions_service.create_child_session(
        pool,
        session_id=child_id,
        account_id=ACC,
        agent_id=agent.id,
        environment_id=env.id,
        agent_version=agent.version,
        parent_run_id=parent_run.id,
        surface=Surface([], [], []),
        vault_ids=[],
        request_id="req#0",
        input="hi",
    )
    tid = await _add_trigger(
        pool,
        child_id,
        {
            "name": "kickoff",
            "source": {"kind": "one_shot", "fire_at": "2026-01-01T00:00:00Z"},
            "action": {"kind": "workflow", "workflow_id": target},
        },
    )

    await run_trigger_step(tid)  # tick-origin fire (no trigger_run_id)

    async with pool.acquire() as conn:
        launched = await conn.fetchrow(
            "SELECT parent_run_id FROM wf_runs WHERE workflow_id = $1", target
        )
    assert launched is not None
    assert launched["parent_run_id"] == parent_run.id


# ─── origin-derived lifecycle + skips + duplicate claim ──────────────────────


async def test_mid_flight_source_conversion_does_not_delete(
    trig_runtime: asyncpg.Pool[Any],
) -> None:
    """An event fire of a trigger converted to one_shot mid-flight must take
    the record arm, never the one-shot DELETE — the lifecycle derives from the
    fire's origin, not the reloaded row."""
    from aios.models.triggers import TriggerUpdate

    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="convert")
    watched = await _make_workflow(pool)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "morph",
            "source": {"kind": "run_completion", "workflow_id": watched},
            "action": {"kind": "wake_owner", "content": "a run completed"},
        },
    )
    await _complete_run_of(pool, watched, env.id)
    (ref,) = await _pending_refs(pool)

    # Convert the source to a FUTURE one-shot while the fire is in flight.
    await trig_service.update_trigger(
        pool,
        session.id,
        "morph",
        TriggerUpdate.model_validate(
            {"source": {"kind": "one_shot", "fire_at": "2030-01-01T00:00:00Z"}}
        ),
        account_id=ACC,
    )

    await run_trigger_step(ref.trigger_id, trigger_run_id=ref.trigger_run_id)

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT source, enabled FROM triggers WHERE id = $1", tid)
    assert row is not None, "event fire deleted the converted trigger"
    assert row["source"] == "one_shot" and row["enabled"]
    rows = await _carrier_rows(pool, tid)
    assert [r["status"] for r in rows] == ["ok"]


async def test_event_skips_finalize_carrier(trig_runtime: asyncpg.Pool[Any]) -> None:
    """Archived / disabled / deleted-trigger event fires all finalize the
    carrier as ``skipped`` — an unfinished claim row would be sweep-re-deferred
    forever — and the pending sweep no longer sees them."""
    from aios.models.triggers import TriggerUpdate

    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="skips")
    watched = await _make_workflow(pool)
    specs = {
        name: await _add_trigger(
            pool,
            session.id,
            {
                "name": name,
                "source": {"kind": "run_completion", "workflow_id": watched},
                "action": {"kind": "wake_owner", "content": "ping"},
            },
        )
        for name in ("t-archived", "t-disabled", "t-deleted")
    }
    await _complete_run_of(pool, watched, env.id)
    refs = {r.trigger_id: r for r in await _pending_refs(pool)}
    assert len(refs) == 3

    # Mutate AFTER the carriers exist, BEFORE the fires execute.
    await trig_service.update_trigger(
        pool,
        session.id,
        "t-disabled",
        TriggerUpdate.model_validate({"enabled": False}),
        account_id=ACC,
    )
    await trig_service.remove_trigger(pool, session.id, "t-deleted", account_id=ACC)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archived_at = now() WHERE id = $1", session.id)

    # NOTE: archiving the session skips ALL THREE remaining fires' sessions —
    # order matters: fire the disabled and deleted ones under the archive too;
    # every branch must still finalize. The deleted trigger exercises the
    # NotFoundError branch; the disabled one is reached as archived-first
    # (archived wins the skip order) — so re-check the disabled reason on a
    # dedicated trigger below.
    for tid, ref in refs.items():
        await run_trigger_step(tid, trigger_run_id=ref.trigger_run_id)

    for name, tid in specs.items():
        rows = await _carrier_rows(pool, tid)
        assert [r["status"] for r in rows] == ["skipped"], name
        assert rows[0]["error_summary"] in (
            "owner session archived",
            "trigger disabled",
            "trigger deleted",
        )
    assert await _pending_refs(pool) == []

    # Dedicated disabled-reason check on a live session.
    _, env2, session2 = await seed_agent_env_session(pool, account_id=ACC, prefix="skips2")
    watched2 = await _make_workflow(pool)
    tid2 = await _add_trigger(
        pool,
        session2.id,
        {
            "name": "t-disabled-2",
            "source": {"kind": "run_completion", "workflow_id": watched2},
            "action": {"kind": "wake_owner", "content": "ping"},
        },
    )
    await _complete_run_of(pool, watched2, env2.id)
    (ref2,) = await _pending_refs(pool)
    await trig_service.update_trigger(
        pool,
        session2.id,
        "t-disabled-2",
        TriggerUpdate.model_validate({"enabled": False}),
        account_id=ACC,
    )
    await run_trigger_step(ref2.trigger_id, trigger_run_id=ref2.trigger_run_id)
    rows = await _carrier_rows(pool, tid2)
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error_summary"] == "trigger disabled"
    # 'skipped' never advances the failure counter.
    assert await _consecutive_failures(pool, tid2) == 0


async def test_duplicate_claim_is_noop(trig_runtime: asyncpg.Pool[Any]) -> None:
    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="dup")
    watched = await _make_workflow(pool)
    target = await _make_workflow(pool)
    await _add_trigger(
        pool,
        session.id,
        {
            "name": "dup",
            "source": {"kind": "run_completion", "workflow_id": watched},
            "action": {"kind": "workflow", "workflow_id": target},
        },
    )
    await _complete_run_of(pool, watched, env.id)
    (ref,) = await _pending_refs(pool)

    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        first = await queries.claim_trigger_run(conn, ref.trigger_run_id, started_at=now)
        second = await queries.claim_trigger_run(conn, ref.trigger_run_id, started_at=now)
    assert first is not None and first["workflow_id"] == watched
    assert second is None

    # A duplicate job against the already-claimed carrier exits without firing.
    await run_trigger_step(ref.trigger_id, trigger_run_id=ref.trigger_run_id)
    async with pool.acquire() as conn:
        launched = await conn.fetchval(
            "SELECT count(*) FROM wf_runs WHERE workflow_id = $1", target
        )
    assert launched == 0


# ─── failure counter atomicity + auto-disable (obligation 3) ─────────────────


async def test_concurrent_error_fires_counter_atomic(
    trig_runtime: asyncpg.Pool[Any],
) -> None:
    """Two unserialized failing event fires increment the counter to EXACTLY 2
    (the SQL CASE; a Python read-modify-write would lose one), the threshold
    disables exactly once with ONE surfaced message, and a concurrent straggler
    past the threshold stays silent (the ``==`` gate)."""
    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="counter")
    watched = await _make_workflow(pool)
    target = await _make_workflow(pool)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "failing",
            "source": {"kind": "run_completion", "workflow_id": watched},
            # The fake completing run below doesn't exist → the account-scoped
            # watched-run read fails NotFound → a deterministic error fire.
            "action": {"kind": "workflow", "workflow_id": target},
        },
    )

    refs = await _seed_fake_fire(pool, watched, "wfr_ghost_a")
    refs += await _seed_fake_fire(pool, watched, "wfr_ghost_b")
    assert len(refs) == 2
    await asyncio.gather(
        *(run_trigger_step(r.trigger_id, trigger_run_id=r.trigger_run_id) for r in refs)
    )
    assert await _consecutive_failures(pool, tid) == 2

    # Two more sequential failures → 4.
    for rid in ("wfr_ghost_c", "wfr_ghost_d"):
        (ref,) = await _seed_fake_fire(pool, watched, rid)
        await run_trigger_step(ref.trigger_id, trigger_run_id=ref.trigger_run_id)
    assert await _consecutive_failures(pool, tid) == 4

    # Two CONCURRENT fires crossing the threshold: counters land on 5 and 6 —
    # exactly one of them surfaces the auto-disable message.
    refs = await _seed_fake_fire(pool, watched, "wfr_ghost_e")
    refs += await _seed_fake_fire(pool, watched, "wfr_ghost_f")
    await asyncio.gather(
        *(run_trigger_step(r.trigger_id, trigger_run_id=r.trigger_run_id) for r in refs)
    )

    async with pool.acquire() as conn:
        enabled = await conn.fetchval("SELECT enabled FROM triggers WHERE id = $1", tid)
        surfaced = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND data::text LIKE '%auto-disabled%'",
            session.id,
        )
    assert enabled is False
    assert surfaced == 1
    failures = await _consecutive_failures(pool, tid)
    assert failures is not None and failures >= 5

    rows = await _carrier_rows(pool, tid)
    assert all(r["status"] in ("error", "skipped") for r in rows)
    assert any("NotFoundError" in (r["error_summary"] or "") for r in rows)


async def test_pin_drift_fire_errors(trig_runtime: asyncpg.Pool[Any]) -> None:
    """An integer pin is a drift assertion: editing the workflow makes the
    next fire record an error naming the drift instead of running the
    unreviewed script."""
    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="pin")
    target = await _make_workflow(pool)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "pinned",
            "source": {"kind": "one_shot", "fire_at": "2026-01-01T00:00:00Z"},
            "action": {"kind": "workflow", "workflow_id": target, "workflow_version": 1},
        },
    )
    await wf_service.update_workflow(
        pool,
        target,
        account_id=ACC,
        expected_version=1,
        script="async def main(input):\n    return 2\n",
    )

    await run_trigger_step(tid)

    rows = await _carrier_rows(pool, tid)
    assert [r["status"] for r in rows] == ["error"]
    assert "workflow version drift" in rows[0]["error_summary"]
    async with pool.acquire() as conn:
        launched = await conn.fetchval(
            "SELECT count(*) FROM wf_runs WHERE workflow_id = $1", target
        )
        surfaced = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 "
            "AND data::text LIKE '%failed to launch its workflow run%'",
            session.id,
        )
    assert launched == 0
    assert surfaced == 1  # the NEW one-shot workflow failure marker


# ─── one-shot lifecycle + tombstones + cron audit ────────────────────────────


async def test_one_shot_workflow_lifecycle(trig_runtime: asyncpg.Pool[Any]) -> None:
    """One-shot workflow fire: row deleted pre-fire; the terminal audit row
    (the only persistent record) carries the created run id; root lineage."""
    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="oneshot")
    target = await _make_workflow(pool)
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "kickoff",
            "source": {"kind": "one_shot", "fire_at": "2026-01-01T00:00:00Z"},
            "action": {"kind": "workflow", "workflow_id": target, "input_template": {"x": 1}},
        },
    )

    await run_trigger_step(tid)

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM triggers WHERE id = $1", tid) == 0
        run = await conn.fetchrow(
            "SELECT id, parent_run_id, launcher_session_id, input FROM wf_runs "
            "WHERE workflow_id = $1",
            target,
        )
    assert run is not None
    assert run["parent_run_id"] is None  # normal session: root run
    assert run["launcher_session_id"] == session.id  # owner authority anchor
    composed = queries.parse_jsonb(run["input"])
    assert composed["trigger"]["source"] == "one_shot"
    assert "run" not in composed["trigger"]
    assert composed["input"] == {"x": 1}

    rows = await _carrier_rows(pool, tid)
    assert [(r["trigger_context"], r["status"]) for r in rows] == [("one_shot", "ok")]
    assert rows[0]["result_id"] == run["id"]


async def test_one_shot_skip_writes_tombstone(trig_runtime: asyncpg.Pool[Any]) -> None:
    """Resolution #7: a one-shot skipped between claim and execute (archive
    race) deletes the row AND leaves a ``skipped`` tombstone — previously the
    skip left zero record anywhere."""
    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="tomb")
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "doomed",
            "source": {"kind": "one_shot", "fire_at": "2026-01-01T00:00:00Z"},
            "action": {"kind": "wake_owner", "content": "never delivered"},
        },
    )
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archived_at = now() WHERE id = $1", session.id)

    await run_trigger_step(tid)

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM triggers WHERE id = $1", tid) == 0
    rows = await _carrier_rows(pool, tid)
    assert [(r["trigger_context"], r["status"]) for r in rows] == [("one_shot", "skipped")]
    assert rows[0]["error_summary"] == "owner session archived"


async def test_cron_fire_writes_audit_row_and_skip_does_not(
    trig_runtime: asyncpg.Pool[Any],
) -> None:
    from aios.models.triggers import TriggerUpdate

    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="cron")
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "beat",
            "source": {"kind": "cron", "schedule": "*/5 * * * *"},
            "action": {"kind": "wake_owner", "content": "tick"},
        },
    )

    await run_trigger_step(tid)  # tick-origin fire

    async with pool.acquire() as conn:
        last_fire_at = await conn.fetchval("SELECT last_fire_at FROM triggers WHERE id = $1", tid)
    rows = await _carrier_rows(pool, tid)
    assert [(r["trigger_context"], r["status"]) for r in rows] == [("cron", "ok")]
    # One fire timestamp: the audit's execution stamp == the echo's last_fire_at.
    assert rows[0]["started_at"] == last_fire_at

    # A cron SKIP records last_fire_status but writes NO audit row (slice-1
    # semantics preserved — the action never executed).
    await trig_service.update_trigger(
        pool,
        session.id,
        "beat",
        TriggerUpdate.model_validate({"enabled": False}),
        account_id=ACC,
    )
    await run_trigger_step(tid)
    async with pool.acquire() as conn:
        status = await conn.fetchval("SELECT last_fire_status FROM triggers WHERE id = $1", tid)
    assert status == "skipped"
    assert len(await _carrier_rows(pool, tid)) == 1  # unchanged


# ─── sweep readers + retention + the jsonb round-trip regression ─────────────


async def test_sweep_readers_and_prune(trig_runtime: asyncpg.Pool[Any]) -> None:
    pool = trig_runtime
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="sweep")
    watched = await _make_workflow(pool)
    await _add_trigger(
        pool,
        session.id,
        {
            "name": "swept",
            "source": {"kind": "run_completion", "workflow_id": watched},
            "action": {"kind": "wake_owner", "content": "ping"},
        },
    )
    await _complete_run_of(pool, watched, env.id)
    (ref,) = await _pending_refs(pool)

    async with pool.acquire() as conn:
        # A young pending row is NOT re-deferred (the live defer gets 60s).
        assert await queries.list_pending_trigger_run_refs(conn, older_than_seconds=60.0) == []
        await conn.execute(
            "UPDATE trigger_runs SET created_at = created_at - interval '2 minutes' WHERE id = $1",
            ref.trigger_run_id,
        )
        aged = await queries.list_pending_trigger_run_refs(conn, older_than_seconds=60.0)
        assert [r.trigger_run_id for r in aged] == [ref.trigger_run_id]

        # Claim it and age it past the stale threshold → counted, never retried.
        assert (
            await queries.claim_trigger_run(conn, ref.trigger_run_id, started_at=datetime.now(UTC))
            is not None
        )
        assert await queries.count_stuck_running_trigger_runs(conn, older_than_seconds=7200.0) == 0
        await conn.execute(
            "UPDATE trigger_runs SET created_at = created_at - interval '3 hours' WHERE id = $1",
            ref.trigger_run_id,
        )
        assert await queries.count_stuck_running_trigger_runs(conn, older_than_seconds=7200.0) == 1

        # Retention prune is age-keyed on created_at.
        assert await queries.prune_trigger_runs(conn, retention_days=30) == 0
        await conn.execute(
            "UPDATE trigger_runs SET created_at = now() - interval '31 days' WHERE id = $1",
            ref.trigger_run_id,
        )
        assert await queries.prune_trigger_runs(conn, retention_days=30) == 1


async def test_numeric_expanded_template_survives_read_and_claim(
    trig_runtime: asyncpg.Pool[Any],
) -> None:
    """The §2.2 read-acceptance twin of the rare-cron test: a legally-written
    template of 1e+308 floats expands ~50x through jsonb numeric normalization;
    the read adapters and — critically — the scheduler's claim transaction must
    still accept the row (one poisoned row would halt EVERY trigger)."""
    pool = trig_runtime
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="jsonb")
    target = await _make_workflow(pool)
    template = [1e308] * 300  # ~2.4 KB on the wire; ~93 KB back from jsonb
    tid = await _add_trigger(
        pool,
        session.id,
        {
            "name": "expander",
            "source": {"kind": "cron", "schedule": "*/5 * * * *"},
            "action": {
                "kind": "workflow",
                "workflow_id": target,
                "input_template": template,
            },
        },
    )

    async with pool.acquire() as conn:
        # Read path revalidates the expanded row…
        row = await queries.unscoped_get_trigger_row(conn, tid)
        assert row.action.kind == "workflow"
        # …and the claim transaction survives it (the blast-radius guard).
        await conn.execute("UPDATE triggers SET next_fire = now() - interval '1 minute'")
    async with pool.acquire() as conn, conn.transaction():
        claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=datetime.now(UTC))
    assert [t.id for t in claimed] == [tid]
