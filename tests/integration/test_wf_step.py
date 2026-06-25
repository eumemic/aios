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
import os
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import httpx
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ForbiddenError, NotFoundError
from aios.harness import runtime
from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.attenuation import Surface
from aios.models.sessions import Session
from aios.models.vaults import VaultCredentialCreate
from aios.models.workflows import WfRunStatus
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services import tasks as tasks_service
from aios.services import vaults as vaults_service
from aios.services import workflows as wf_service
from aios.workflows import run_tools, service
from aios.workflows.child_id import child_session_id
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH, CallKeyer
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
        run_tools._INFLIGHT.clear()  # no leaked tasks from a prior (possibly failed) test
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
            # The fire-and-forget tool task's own wake — patch it too, else it enqueues a
            # real wake_workflow job against the test DB (the harvest is driven manually).
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            # The service-level archive/delete now eagerly fail open child requests and
            # defer a run wake after commit — patch it so they don't enqueue a real job.
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _events(pool: asyncpg.Pool[Any], run_id: str) -> list[tuple[int, str, str | None]]:
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    return [(e.seq, e.type, e.call_key) for e in rows]


async def _needing(pool: asyncpg.Pool[Any]) -> set[str]:
    """Run ids the needs-step sweep filter currently matches (fixed horizons)."""
    async with pool.acquire() as conn:
        return set(
            await wf_queries.list_run_ids_needing_step(
                conn, agent_deadline_seconds=3600, tool_stale_seconds=60
            )
        )


async def _make_run(
    pool: asyncpg.Pool[Any], script: str, *, input: Any = None, name: str = "w"
) -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id="acc_wf", name=name, script=script)
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf", input=input
    )
    return run.id


async def _make_launcher_session(pool: asyncpg.Pool[Any], agent_id: str) -> Session:
    return await sessions_service.create_session(
        pool,
        account_id="acc_wf",
        agent_id=agent_id,
        environment_id="env_wf",
        title=None,
        metadata={},
    )


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
        model=None,
        parent_run_id=run_id,
        surface=Surface([], [], []),
        vault_ids=[],
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
        model=None,
        parent_run_id=run_id,
        surface=Surface([], [], []),
        vault_ids=[],
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
        model=None,
        parent_run_id=run_id,
        surface=Surface([], [], []),
        vault_ids=[],
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

    # The display blob carries the correlation id; the trusted caller is on the edge.
    request = parse_jsonb(req["data"])["metadata"]["request"]
    assert request["request_id"] == "sha:x#0"
    async with pool.acquire() as conn:
        caller = await db_queries.get_request_caller(conn, cid, request_id="sha:x#0")
    assert caller == {"kind": "run", "id": run_id}


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


async def test_launcher_receives_gate_opened_and_resume_gate_happy_path(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    launcher = await _make_launcher_session(pool, wf_agent_id)
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="w-gate", script=_GATE_SCRIPT
        )
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=wf.id,
        environment_id="env_wf",
        launcher_session_id=launcher.id,
    )

    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        gate_event = next(
            e for e in await wf_queries.list_run_events(conn, run.id) if e.type == "call_started"
        )
        delivered = await db_queries.read_request_response(
            conn, launcher.id, account_id="acc_wf", request_id=gate_event.call_key or ""
        )
    assert delivered is not None
    assert delivered["is_error"] is False
    assert delivered["result"] == {
        "event": "gate_opened",
        "run_id": run.id,
        "gate_nonce": gate_event.payload["gate_nonce"],
    }

    returned = await wf_service.resume_gate_by_nonce(
        pool,
        run_id=run.id,
        account_id="acc_wf",
        gate_nonce=gate_event.payload["gate_nonce"],
        result="yes",
        resumer_session_id=launcher.id,
    )
    assert returned.id == run.id and returned.status == "suspended"
    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        completed = await wf_queries.get_run_for_step(conn, run.id)
    assert (
        completed is not None
        and completed.status == "completed"
        and completed.output == {"answer": "yes"}
    )


async def test_gate_opened_delivery_dedupes_by_call_key_on_replay(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    launcher = await _make_launcher_session(pool, wf_agent_id)
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="w-gate", script=_GATE_SCRIPT
        )
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=wf.id,
        environment_id="env_wf",
        launcher_session_id=launcher.id,
    )

    await run_workflow_step(run.id)
    await run_workflow_step(run.id)  # re-drive while still suspended at the same gate

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_response'",
            launcher.id,
            "acc_wf",
        )
    assert len(rows) == 1
    assert db_queries.parse_jsonb(rows[0]["data"])["result"]["event"] == "gate_opened"


async def test_non_launcher_cannot_resume_gate(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    launcher = await _make_launcher_session(pool, wf_agent_id)
    other = await _make_launcher_session(pool, wf_agent_id)
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="w-gate", script=_GATE_SCRIPT
        )
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=wf.id,
        environment_id="env_wf",
        launcher_session_id=launcher.id,
    )

    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        gate_event = next(
            e for e in await wf_queries.list_run_events(conn, run.id) if e.type == "call_started"
        )
    with pytest.raises(ForbiddenError):
        await wf_service.resume_gate_by_nonce(
            pool,
            run_id=run.id,
            account_id="acc_wf",
            gate_nonce=gate_event.payload["gate_nonce"],
            result="nope",
            resumer_session_id=other.id,
        )


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


# ─── annotations — log()/phase() journaling (B-783) ──────────────────────────


async def _typed(pool: asyncpg.Pool[Any], run_id: str) -> list[tuple[str, str | None, str | None]]:
    """The journal as ``(type, kind, text)`` — ``kind``/``text`` are an annotation's
    payload, ``None`` for every other event type."""
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    return [
        (e.type, e.payload.get("kind"), e.payload.get("text"))
        if e.type == "annotation"
        else (e.type, None, None)
        for e in rows
    ]


_ANN_SCRIPT = (
    "async def main(input):\n"
    "    phase('start')\n"
    "    log('before', 'gate')\n"
    "    r = await gate({'q': 'ok?'})\n"
    "    log('after gate')\n"
    "    return r\n"
)


async def test_annotations_journaled_in_execution_order(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, _ANN_SCRIPT)

    # Wake 1: phase() + log() emit before the gate frontier.
    await run_workflow_step(run_id)
    assert [(t, k is not None) for _s, t, k in await _events(pool, run_id)] == [
        ("run_started", False),
        ("annotation", True),
        ("annotation", True),
        ("call_started", True),
    ]
    gate_key = next(k for _s, t, k in await _events(pool, run_id) if t == "call_started")
    assert gate_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")

    # Wake 2: replay re-emits the first two annotations (deduped by the memo), harvests
    # the gate, then emits the post-gate log (new) — so the journal stays in execution
    # order with NO duplicate annotation rows.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert await _typed(pool, run_id) == [
        ("run_started", None, None),
        ("annotation", "phase", "start"),
        ("annotation", "log", "before gate"),  # log(*args) space-joined
        ("call_started", None, None),
        ("call_result", None, None),
        ("annotation", "log", "after gate"),
        ("run_completed", None, None),
    ]


async def test_replay_does_not_duplicate_annotations(wf_runtime: asyncpg.Pool[Any]) -> None:
    # Re-driving the suspended run WITHOUT resolving the gate would quiet-wake (no
    # re-emit); instead force a real replay by resolving the gate, and assert the two
    # pre-gate annotations each appear exactly once though wake 2 re-emitted them.
    pool = wf_runtime
    run_id = await _make_run(pool, _ANN_SCRIPT)
    await run_workflow_step(run_id)
    gate_key = next(k for _s, t, k in await _events(pool, run_id) if t == "call_started")
    assert gate_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")
    await run_workflow_step(run_id)
    annotations = [(k, txt) for typ, k, txt in await _typed(pool, run_id) if typ == "annotation"]
    assert annotations == [("phase", "start"), ("log", "before gate"), ("log", "after gate")]


async def test_annotation_write_once_first_text_wins(wf_runtime: asyncpg.Pool[Any]) -> None:
    # The memo UNIQUE (run_id, call_key, type) makes annotation text write-once per
    # call_key: a replay that re-emits the SAME key with edited text is a no-op, and the
    # original row stands (correct under replay — a deterministic script can't change a
    # position's text, but a hand-edited script source must never rewrite history).
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    async with pool.acquire() as conn:
        first = await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="annotation",
            call_key="sha:ann#0",
            payload={"kind": "log", "text": "first"},
        )
        second = await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="annotation",
            call_key="sha:ann#0",
            payload={"kind": "log", "text": "EDITED"},
        )
    assert first is not None
    assert second is None  # ON CONFLICT (run_id, call_key, type) → no-op, no seq consumed
    annotations = [(k, txt) for typ, k, txt in await _typed(pool, run_id) if typ == "annotation"]
    assert annotations == [("log", "first")]


async def test_annotation_before_crash_is_captured(wf_runtime: asyncpg.Pool[Any]) -> None:
    # The debuggability payoff: a log() just before an author exception is journaled
    # AHEAD of the run_completed(errored) bookend, so an operator can see how far a
    # crashing run got.
    pool = wf_runtime
    run_id = await _make_run(
        pool,
        "async def main(input):\n    log('about to fail')\n    raise ValueError('boom')\n",
    )
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "errored"
    assert await _typed(pool, run_id) == [
        ("run_started", None, None),
        ("annotation", "log", "about to fail"),
        ("run_completed", None, None),
    ]


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
        run_or_none = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run_or_none is not None
    run = run_or_none
    assert run.status == "cancelled" and run.output is None
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
        run_or_none = await wf_queries.get_run_for_step(conn, run_id)
    assert run_or_none is not None
    assert run_or_none.status == "cancelled"
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
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            parent_run_id=parent_id,
            depth=9,
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
    script = f"async def main(input):\n    return await agent({{'task': 'go'}}, agent_id={wf_agent_id!r})\n"
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


async def test_agent_spawn_stamps_decremented_depth_on_edge(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The run->session ``agent()`` edge carries ``run.depth - 1`` (#1124): a root
    run seeds at the full budget, so its child's ``request_opened`` edge is stamped
    one below. The trusted depth rides ONLY the edge, never message metadata."""
    from aios.workflows.service import INVOKE_MAX_DEPTH

    pool = wf_runtime
    script = f"async def main(input):\n    return await agent({{'task': 'go'}}, agent_id={wf_agent_id!r})\n"
    run_id = await _make_run(pool, script)  # edgeless root -> depth == INVOKE_MAX_DEPTH
    with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()):
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None and run.depth == INVOKE_MAX_DEPTH
        events = await wf_queries.list_run_events(conn, run_id)
        child_id = next(e.payload["child_session_id"] for e in events if e.type == "call_started")
        edge_depth = await conn.fetchval(
            "SELECT (data->>'depth')::int FROM events WHERE session_id = $1 "
            "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
            child_id,
        )
    assert edge_depth == INVOKE_MAX_DEPTH - 1  # decremented one hop down


async def test_agent_spawn_refused_when_run_depth_exhausted(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A run at depth 0 has no trusted budget left: ``_open_agent_capability``
    refuses BEFORE writing the child (a journaled, catchable rejection), so no
    over-budget child session/edge ever exists (#1124)."""
    from aios.workflows.host_launcher import EmittedCapability
    from aios.workflows.step import _open_agent_capability

    pool = wf_runtime
    script = f"async def main(input):\n    return await agent('go', agent_id={wf_agent_id!r})\n"
    run_id = await _make_run(pool, script)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE wf_runs SET depth = 0 WHERE id = $1", run_id)

    spec = {"agent_id": wf_agent_id, "input": "go", "output_schema": None}
    call_key = CallKeyer().next("agent", spec)
    cap = EmittedCapability(capability_id="agent", call_key=call_key, spec=spec)
    cid = child_session_id(run_id, call_key)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None and run.depth == 0
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w:
            result = await _open_agent_capability(
                conn, pool, run, cap, agent_spawns=0, max_agent_calls=1000
            )
        assert result.rejected  # catchable rejection journaled, not a crash
        w.assert_not_awaited()  # no child -> no wake
        # Refuse-BEFORE-write: neither the child row nor its edge was created.
        assert await conn.fetchval("SELECT count(*) FROM sessions WHERE id = $1", cid) == 0
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM events WHERE session_id = $1 "
                "AND kind = 'lifecycle' AND data->>'event' = 'request_opened'",
                cid,
            )
            == 0
        )


async def _make_agent_with_litellm_extra(
    pool: asyncpg.Pool[Any], litellm_extra: dict[str, Any]
) -> str:
    """An agent whose model identity (litellm_extra, api_base) is under test (#823)."""
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_wf",
        name="endpoint-agent",
        model="test/dummy",
        system="test child agent",
        tools=[],
        description=None,
        metadata={},
        litellm_extra=litellm_extra,
        window_min=1000,
        window_max=100000,
    )
    return agent.id


def _allow_api_bases(monkeypatch: Any, *bases: str) -> None:
    """Bind the operator trusted-endpoint allowlist for the model-identity clamp."""
    from aios.config import get_settings

    allowed = get_settings().model_copy(update={"trusted_inference_api_bases": list(bases)})
    monkeypatch.setattr("aios.services.attenuation.get_settings", lambda: allowed)


async def test_named_agent_spawn_freezes_clamped_litellm_extra(
    wf_runtime: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """#823: a named-agent child's model identity (litellm_extra, api_base foremost) is
    frozen onto its session row at spawn, mirroring the #794 surface snapshot — so a
    later ``update_agent`` can't shift where the child's mind runs on replay."""
    pool = wf_runtime
    extra = {"api_base": "https://trusted.example", "temperature": 0.1}
    _allow_api_bases(monkeypatch, "https://trusted.example")
    agent_id = await _make_agent_with_litellm_extra(pool, extra)
    script = f"async def main(input):\n    return await agent('go', agent_id={agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
        started = next(e for e in events if e.type == "call_started")
        cid = started.payload["child_session_id"]
        frozen = await db_queries.get_session_frozen_litellm_extra(conn, cid, account_id="acc_wf")
    assert frozen == extra  # the clamped model identity is frozen verbatim


async def test_frozen_litellm_extra_is_replay_sound_against_agent_update(
    wf_runtime: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """The frozen model identity is read by ``load_for_session`` — NOT the live agent —
    so re-pointing the agent's ``api_base`` after spawn does not move the child's
    inference endpoint. Mirrors the surface snapshot's replay-soundness."""
    pool = wf_runtime
    extra = {"api_base": "https://trusted.example"}
    _allow_api_bases(monkeypatch, "https://trusted.example", "https://elsewhere.example")
    agent_id = await _make_agent_with_litellm_extra(pool, extra)
    script = f"async def main(input):\n    return await agent('go', agent_id={agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
        cid = next(e for e in events if e.type == "call_started").payload["child_session_id"]

    # Operator re-points the agent's endpoint AFTER the child was spawned.
    current = await agents_service.get_agent(pool, agent_id, account_id="acc_wf")
    await agents_service.update_agent(
        pool,
        agent_id,
        account_id="acc_wf",
        expected_version=current.version,
        litellm_extra={"api_base": "https://elsewhere.example"},
    )

    # load_for_session resolves the FROZEN identity, not the shifted live one.
    async with pool.acquire() as conn:
        child = await db_queries.get_session_bare(conn, cid, account_id="acc_wf")
        effective = await agents_service.load_for_session(
            pool, child, account_id="acc_wf", conn=conn
        )
    assert effective.litellm_extra == {"api_base": "https://trusted.example"}


async def test_named_agent_spawn_refused_when_api_base_untrusted(
    wf_runtime: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """#823: the model-identity clamp FAILS CLOSED at the spawn edge — an agent routing
    to an endpoint that is neither the launcher's (a run has none) nor on the operator
    allowlist is refused with a catchable ``untrusted_api_base`` rejection, BEFORE any
    child row exists. The attacker never receives the child's context."""
    from aios.workflows.host_launcher import EmittedCapability
    from aios.workflows.step import _open_agent_capability

    pool = wf_runtime
    _allow_api_bases(monkeypatch)  # empty allowlist → no redirect is trusted
    agent_id = await _make_agent_with_litellm_extra(pool, {"api_base": "https://hostile.example"})
    script = f"async def main(input):\n    return await agent('go', agent_id={agent_id!r})\n"
    run_id = await _make_run(pool, script)

    spec = {"agent_id": agent_id, "input": "go", "output_schema": None}
    call_key = CallKeyer().next("agent", spec)
    cap = EmittedCapability(capability_id="agent", call_key=call_key, spec=spec)
    cid = child_session_id(run_id, call_key)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w:
            result = await _open_agent_capability(
                conn, pool, run, cap, agent_spawns=0, max_agent_calls=1000
            )
        assert result.rejected  # catchable rejection journaled, not a crash
        w.assert_not_awaited()  # no child -> no wake
        # Fail-closed BEFORE write: neither the child row nor its request edge exists.
        assert await conn.fetchval("SELECT count(*) FROM sessions WHERE id = $1", cid) == 0
        events = await wf_queries.list_run_events(conn, run_id)
    result_evt = next(e for e in events if e.type == "call_result")
    assert result_evt.payload["error"]["kind"] == "untrusted_api_base"


async def test_named_agent_spawn_admits_allowlisted_api_base(
    wf_runtime: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """The allowlist arm: an operator-trusted endpoint is admitted (the child spawns)."""
    pool = wf_runtime
    _allow_api_bases(monkeypatch, "https://trusted.example")
    agent_id = await _make_agent_with_litellm_extra(pool, {"api_base": "https://trusted.example"})
    script = f"async def main(input):\n    return await agent('go', agent_id={agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert children == 1  # admitted → exactly one child spawned


async def test_named_agent_spawn_admits_when_no_api_base(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The default case: an agent with no ``api_base`` runs on the default operator
    endpoint (== the launcher's None) and is admitted with an EMPTY allowlist — the
    clamp is a no-op for the trusted-catalog model that holds today."""
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent('go', agent_id={wf_agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
        cid = next(e for e in events if e.type == "call_started").payload["child_session_id"]
        frozen = await db_queries.get_session_frozen_litellm_extra(conn, cid, account_id="acc_wf")
    assert frozen == {}  # no redirect frozen → default endpoint


async def test_generic_agent_spawn_creates_agentless_child_with_run_surface(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    pool = wf_runtime
    script = "async def main(input):\n    return await agent({'task': 'go'}, model='test/generic', label='generic')\n"
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="generic-w",
            script=script,
            tools=[ToolSpec(type="web_search")],
        )
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
    )
    await run_workflow_step(run.id)

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run.id)
        started = next(e for e in events if e.type == "call_started")
        child = await db_queries.get_session_bare(
            conn, started.payload["child_session_id"], account_id="acc_wf"
        )
        frozen = await db_queries.get_session_frozen_surface(conn, child.id, account_id="acc_wf")
    assert started.payload["label"] == "generic"
    assert started.payload["child_agent_version"] is None
    assert child.agent_id is None
    assert child.agent_version is None
    assert child.model == "test/generic"
    assert frozen == Surface(tools=[ToolSpec(type="web_search")], mcp_servers=[], http_servers=[])


async def test_agent_spawn_idempotent_on_replay(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """C1/C2/C6: a re-step (crash replay) must NOT double-spawn or re-deliver input."""
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent('hi', agent_id={wf_agent_id!r})\n"
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


async def test_agent_not_found_is_catchable(wf_runtime: asyncpg.Pool[Any]) -> None:
    """A missing agent surfaces as a catchable ``AgentError`` at the ``await``, not an
    uncatchable terminal run error. Wake 1 journals a ``call_result`` error (and
    self-wakes); wake 2 replays and throws the AgentError where the script catches it
    and completes. No child is ever spawned for the bad call."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        "        await agent('x', agent_id='agent_nope')\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # wake 1: journal the catchable error + self-wake
    await run_workflow_step(run_id)  # wake 2: replay -> throw AgentError -> caught -> return

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output == {
        "caught": True,
        "kind": "agent_not_found",
        "msg": "agent 'agent_nope' not found",
    }
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is True
    assert cr.payload["error"] == {
        "kind": "agent_not_found",
        "message": "agent 'agent_nope' not found",
    }
    assert children == 0  # the rejected call never spawned a child


async def test_bad_agent_call_invalid_schema_is_catchable(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A structurally-invalid ``output_schema`` surfaces as a catchable
    ``AgentError(kind='bad_agent_call')`` at the await, preserving the descriptive
    schema message. Two wakes; no child is spawned."""
    pool = wf_runtime
    bad_schema = {"type": "not-a-real-type"}
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('hi', output_schema={bad_schema!r}, agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output["caught"] is True
    assert run.output["kind"] == "bad_agent_call"
    assert run.output["msg"].startswith("agent() output_schema is not a valid JSON Schema")
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["is_error"] is True
    assert cr.payload["error"]["kind"] == "bad_agent_call"
    assert cr.payload["error"]["message"].startswith(
        "agent() output_schema is not a valid JSON Schema"
    )
    assert children == 0


async def test_bad_agent_call_non_string_agent_id_is_catchable(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A non-string ``agent_id`` (here an int) reaches the parent's
    ``not isinstance(agent_id, str)`` branch and surfaces as a catchable
    ``AgentError(kind='bad_agent_call')`` with the exact descriptive message. No child."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        "        await agent('hi', agent_id=123)\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output == {
        "caught": True,
        "kind": "bad_agent_call",
        "msg": "agent() requires agent_id to be a string or None, got 123",
    }
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["error"] == {
        "kind": "bad_agent_call",
        "message": "agent() requires agent_id to be a string or None, got 123",
    }
    assert children == 0


async def test_one_bad_branch_in_fanout_does_not_terminate_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """HEADLINE: a spawn error on ONE branch of a fan-out journals that branch's
    catchable error and must NOT abort spawning the others. After wake 1 the run is
    NOT terminated — the two good children are spawned and exactly one error is
    journaled; it owes one drive ('running'), and a self-wake re-suspends it on the two
    good children. After they answer, the barrier returns ``[None, r_a, r_b]``."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    return await parallel([\n"
        "        lambda: agent('x', agent_id='agent_nope'),\n"
        f"        lambda: agent('a', agent_id={wf_agent_id!r}),\n"
        f"        lambda: agent('b', agent_id={wf_agent_id!r}),\n"
        "    ])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # wake 1: spawn the 2 good children, journal 1 error

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    # NOT terminated by the bad branch — it owes a drive (the catchable error replay).
    assert run is not None and run.status == "running"
    started = [e for e in events if e.type == "call_started"]
    assert len(started) == 2  # exactly the two good children spawned
    errors = [e for e in events if e.type == "call_result" and e.payload.get("is_error")]
    assert len(errors) == 1  # exactly one journaled spawn error
    assert errors[0].payload["error"]["kind"] == "agent_not_found"
    children = await _children_of(pool, run_id)
    assert len(children) == 2

    await run_workflow_step(run_id)  # owed drive: replay throws the caught error → re-suspend
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "suspended"  # settled on the two good children
    assert len([e for e in events if e.type == "call_started"]) == 2  # no new spawns on replay
    assert len([e for e in events if e.type == "call_result" and e.payload.get("is_error")]) == 1

    # Answer the two good children, then drive again: barrier returns with the bad
    # branch's None in its (first) slot.
    for cid, value in zip(children, ["r_a", "r_b"], strict=True):
        rid = await _open_request_id(pool, cid)
        with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
            await workflow_completion.return_handler(cid, {"request_id": rid, "value": value})

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == [None, "r_a", "r_b"]


async def test_spawn_error_call_result_has_no_paired_call_started(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """The journaled spawn error is a bare ``call_result`` for the bad call_key with NO
    paired ``call_started`` — and the replay-prefix check (which asserts every inflight
    ``call_started`` is re-emitted) never trips ``nondeterministic_replay``, because the
    bad call has no open ``call_started`` to orphan."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        "        await agent('x', agent_id='agent_nope')\n"
        "    except AgentError:\n"
        "        return 'caught'\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # wake 1: journal the error, self-wake

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    error_keys = {
        e.call_key for e in events if e.type == "call_result" and e.payload.get("is_error")
    }
    started_keys = {e.call_key for e in events if e.type == "call_started"}
    assert len(error_keys) == 1
    assert error_keys.isdisjoint(started_keys)  # the bad call_result has no call_started
    # The run did not error (least of all with nondeterministic_replay).
    completed = [e for e in events if e.type == "run_completed"]
    assert all(
        e.payload.get("error", {}).get("kind") != "nondeterministic_replay" for e in completed
    )


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
    # #1413 background-child path: get_open_obligations now runs UNCONDITIONALLY
    # (the background-child fast-path short-circuit was removed), so the child's
    # open `run` obligation is fetched onto the prelude -- the data the always-on
    # obligations tail block renders. The fast-path removal did NOT regress the
    # return/error gate (it stayed bool(obligations), asserted above).
    assert child_prelude.obligations, "background child's run obligation must be computed"
    assert child_prelude.obligations[0].caller_kind == "run"
    assert child_prelude.obligations_block_upper_bound_local > 0

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
    # An ordinary foreground session owes nothing -> no obligations, no reserved
    # tail budget (the unconditional query returns []).
    assert fg_prelude.obligations == []
    assert fg_prelude.obligations_block_upper_bound_local == 0


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
    wake.assert_awaited_once_with(run_id, batch=True)

    async with pool.acquire() as conn:
        response = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:d2#0"
        )
        signals = await wf_queries.list_run_signals(conn, run_id)
    assert response is not None
    assert response["is_error"] is False and response["result"] == {"answer": 42}
    # The child_done side-marker committed with the response (#780): a lost caller
    # wake stays SQL-visible to the needs-step sweep.
    assert [(s.call_key, s.kind) for s in signals] == [("sha:d2#0", "child_done")]
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
    wake.assert_awaited_once_with(run_id, batch=True)  # only the first response woke the caller

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
    from aios.harness.inflight_tool_registry import InflightToolRegistry

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:rd#0")
    call = {
        "id": "call_ret",
        "function": {
            "name": "return",
            "arguments": _json.dumps({"request_id": "sha:rd#0", "value": "ok"}),
        },
    }

    prev_reg = runtime.inflight_tool_registry
    runtime.inflight_tool_registry = InflightToolRegistry()
    try:
        with (
            mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()) as wake,
            mock.patch("aios.harness.sweep.defer_wake", new=AsyncMock()),
        ):
            await tool_dispatch._execute_tool_async(pool, cid, call, account_id="acc_wf")
    finally:
        runtime.inflight_tool_registry = prev_reg

    wake.assert_awaited_once_with(run_id, batch=True)
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

    # A cap high enough that the lifetime quota never trips — this test exercises the
    # re-attach / self-wake arm, not H1.
    _SPAWN_KW = {"agent_spawns": 0, "max_agent_calls": 1000}

    pool = wf_runtime
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
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
            r1 = await _open_agent_capability(conn, pool, run, cap, **_SPAWN_KW)
        assert not r1.rejected and not r1.needs_rewake
        w1.assert_awaited_once()
        # Re-attach, no marker yet → no defer_wake, no self-wake.
        with mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()) as w2:
            r2 = await _open_agent_capability(conn, pool, run, cap, **_SPAWN_KW)
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
            r3 = await _open_agent_capability(conn, pool, run, cap, **_SPAWN_KW)
        assert not r3.rejected and r3.needs_rewake  # marker present → self-wake to harvest
        w3.assert_not_awaited()


# ─── B2.E — harvest the child marker (spawn -> return -> complete) ────────────


async def _child_id_of(pool: asyncpg.Pool[Any], run_id: str) -> str:
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    return str(next(e.payload["child_session_id"] for e in events if e.type == "call_started"))


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
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
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
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
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
        f"        await agent('go', agent_id={wf_agent_id!r})\n"
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
    script = f"async def main(input):\n    r = await agent('go', agent_id={wf_agent_id!r})\n    return {{'got': r}}\n"
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
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
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
    wake.assert_awaited_once_with(run_id, batch=True)  # the caller was woken to harvest

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


_TOOLCALL_SEQ = 0


async def _activity_turn(pool: asyncpg.Pool[Any], session_id: str) -> None:
    """Simulate one ACTIVITY (tool-call) turn and resolve it — the #1412 reset
    anchor for ``count_request_nudges``.

    Goes through the real guard (which short-circuits on ``tool_calls`` without
    nudging) to append the assistant tool-call event, then appends the matching
    tool-role result so ``open_tool_call_count`` returns to 0 and the session can
    idle again on the next pure-text turn. After this, the consecutive-inaction
    nudge count for any request resets to 0."""
    global _TOOLCALL_SEQ
    _TOOLCALL_SEQ += 1
    call_id = f"call_act_{_TOOLCALL_SEQ}"
    child = await sessions_service.get_session_basic(pool, session_id, account_id="acc_wf")
    assistant = {
        "role": "assistant",
        "content": "",
        "reacting_to": child.last_event_seq,
        "tool_calls": [{"id": call_id, "function": {"name": "noop", "arguments": "{}"}}],
    }
    # The guard appends the assistant event and short-circuits (tool_calls present
    # -> never idle, no nudge). The tool result then balances the open count.
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, session_id, assistant, account_id="acc_wf"
    )
    assert not result.nudged  # an activity turn is never nudged
    async with pool.acquire() as conn:
        await db_queries.append_event(
            conn,
            account_id="acc_wf",
            session_id=session_id,
            kind="message",
            data={"role": "tool", "tool_call_id": call_id, "name": "noop", "content": "ok"},
        )


async def test_idle_with_open_request_is_nudged(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A child that ends a tool-call-free turn while still owing a request is nudged
    in the same transaction as the idling assistant event — so it never goes idle
    with a debt — and the nudge (a user message) keeps it active for another turn."""
    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:n#0")

    assistant = await _idle_assistant_turn(pool, cid)
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, assistant, account_id="acc_wf"
    )
    assert result.nudged and result.autoerror_caller_run_id is None

    async with pool.acquire() as conn:
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        nudges = await db_queries.count_request_nudges(
            conn, cid, account_id="acc_wf", request_id="sha:n#0"
        )
        status = await db_queries.derive_session_status(conn, cid, account_id="acc_wf")
    assert open_ids == ["sha:n#0"]  # still open — nudging is not answering
    assert nudges == 1
    assert status == "active"  # the nudge user message keeps it alive, no idle window


async def test_nudge_surfaces_owed_obligation_with_contract(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1522 / folds #1514: the quiescence-attempt surfacing (the nudge written when
    a session tries to stop while owing) draws from the SHARED owed-read-model
    renderer and shows the owed obligation WITH its output_schema contract — "here
    is what you owe and in what format". The cheap guard yes/no still rides on
    ``get_open_request_ids``; only this rendered content carries the schema."""
    pool = wf_runtime
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:contract#0", output_schema=schema)

    assistant = await _idle_assistant_turn(pool, cid)
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, assistant, account_id="acc_wf"
    )
    assert result.nudged

    # Read back the nudge user message and assert it surfaces the contract.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'message' "
            "AND data->'metadata' ? 'nudged_request_ids' ORDER BY seq DESC LIMIT 1",
            cid,
        )
    assert row is not None
    content = db_queries.parse_jsonb(row["data"])["content"]
    assert "sha:contract#0" in content  # the owed request_id
    assert "output_schema" in content  # the #1514 contract-bearing surfacing
    assert '"ok"' in content  # the actual schema body, rendered (bounded)


async def test_tell_spawned_child_reaches_idle_with_zero_nudges(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1197: a `Tell(NewSession)` fire-and-forget spawn writes an UNAWAITED edge,
    so the child owes no response — the quiescence guard skips the nudge/no_return
    loop entirely and it reaches idle with ZERO nudges (the acceptance criterion).

    Contrast with `test_idle_with_open_request_is_nudged`: same idle assistant
    turn, but the awaited bit is the only difference."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    cid = child_session_id(run_id, "sha:tell#0")
    created = await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        model=None,
        parent_run_id=run_id,
        surface=Surface([], [], []),
        vault_ids=[],
        input="fire-and-forget",
        awaited=False,  # the Tell arm
    )
    assert created is True

    assistant = await _idle_assistant_turn(pool, cid)
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, assistant, account_id="acc_wf"
    )
    assert not result.nudged and result.autoerror_caller_run_id is None  # no nudge, no auto-error

    async with pool.acquire() as conn:
        # The unawaited edge is excluded from the open set — no obligation.
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        status = await db_queries.derive_session_status(conn, cid, account_id="acc_wf")
        nudge_msgs = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'message' AND role = 'user' "
            "AND data->'metadata'->'nudged_request_ids' IS NOT NULL",
            cid,
            "acc_wf",
        )
    assert open_ids == []  # no awaited request open
    assert status == "idle"  # free to rest — fire-and-forget owes nothing
    assert nudge_msgs == 0  # ZERO nudges


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
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, fg.id, assistant, account_id="acc_wf"
    )
    assert not result.nudged and result.autoerror_caller_run_id is None
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
        _qresult = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
        )
        nudged = _qresult.nudged
        assert nudged  # still under budget — keep nudging

    # Budget spent: the next pure-text turn auto-errors the request instead.
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, await _idle_assistant_turn(pool, cid), account_id="acc_wf"
    )
    assert not result.nudged
    # auto-errored → the caller run is woken to harvest the no_return
    assert result.autoerror_caller_run_id == run_id

    async with pool.acquire() as conn:
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:nr#0"
        )
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        status = await db_queries.derive_session_status(conn, cid, account_id="acc_wf")
        signals = await wf_queries.list_run_signals(conn, run_id)
    assert resp is not None and resp["is_error"] is True and resp["error"] == {"kind": "no_return"}
    assert open_ids == []  # answered (by the backstop) → no longer open
    assert status == "idle"  # nothing owed → free to rest
    # The backstop is the SECOND response writer — it too leaves the child_done
    # marker, so a lost post-commit caller wake stays sweep-visible (#780).
    assert [(s.call_key, s.kind) for s in signals] == [("sha:nr#0", "child_done")]


async def test_activity_turn_resets_consecutive_nudge_count(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1412: ``count_request_nudges`` counts nudges SINCE the last tool-call turn,
    not over the request's lifetime. An activity (tool-call) turn is the reset
    anchor — after it, the consecutive count drops back to 0 even though earlier
    nudges still exist in the log."""
    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:reset#0")

    # Two idle turns while owing the request -> two nudges accumulate.
    for _ in range(2):
        r = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
        )
        assert r.nudged
    async with pool.acquire() as conn:
        assert (
            await db_queries.count_request_nudges(
                conn, cid, account_id="acc_wf", request_id="sha:reset#0"
            )
            == 2
        )

    # An activity turn resets the consecutive count to 0 (the lifetime count is 2).
    await _activity_turn(pool, cid)
    async with pool.acquire() as conn:
        assert (
            await db_queries.count_request_nudges(
                conn, cid, account_id="acc_wf", request_id="sha:reset#0"
            )
            == 0
        )
        # The earlier nudge user messages were NOT deleted — only excluded by seq.
        lifetime = await conn.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'message' AND role = 'user' "
            "AND data->'metadata'->'nudged_request_ids' @> to_jsonb($3::text)",
            cid,
            "acc_wf",
            "sha:reset#0",
        )
    assert lifetime == 2  # the prior nudges remain in the log, just past the anchor


async def test_interleaved_activity_never_trips_nudge_budget(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1412: a working agent that interleaves tool-call turns with idle turns
    NEVER hits ``no_return`` — each activity turn resets the consecutive count, so
    the budget is never reached even after far more than BUDGET total idle turns."""
    pool = wf_runtime
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:work#0")

    for _ in range(sessions_service.REQUEST_NUDGE_BUDGET * 3):
        r = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
        )
        assert r.nudged  # always under budget -> nudged, never auto-errored
        assert r.autoerror_caller_run_id is None
        await _activity_turn(pool, cid)  # reset before the next idle turn

    async with pool.acquire() as conn:
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:work#0"
        )
    assert open_ids == ["sha:work#0"]  # still open — never abandoned
    assert resp is None  # no no_return backstop was ever written


async def test_no_return_after_exactly_n_consecutive_idle_turns(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1412: a session idle exactly N=BUDGET consecutive turns while owing a
    request is ``no_return``'d on turn N+1 — and the count survives an EARLIER
    activity turn (the consecutive run starts after the last tool-call turn)."""
    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:stuck#0")

    # An early activity turn: proves a stale reset anchor doesn't grant extra slack.
    await _activity_turn(pool, cid)

    for _ in range(sessions_service.REQUEST_NUDGE_BUDGET):
        r = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
        )
        assert r.nudged and r.autoerror_caller_run_id is None

    # Turn N+1: budget spent on the consecutive run -> auto-error.
    r = await sessions_service.append_assistant_and_guard_quiescence(
        pool,
        cid,
        await _idle_assistant_turn(pool, cid),
        account_id="acc_wf",
    )
    assert not r.nudged and r.autoerror_caller_run_id == run_id
    async with pool.acquire() as conn:
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:stuck#0"
        )
    assert resp is not None and resp["is_error"] and resp["error"] == {"kind": "no_return"}


async def test_per_request_count_is_independent_stuck_sibling_still_no_returns(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """#1412 (multi-request reset = per-request): the consecutive count is filtered
    by ``nudged_request_ids``, so it is PER-REQUEST. Answering one obligation (which
    stops it being nudged) does not spare a still-stuck SIBLING — the sibling keeps
    its own consecutive count and still hits ``no_return`` — and a nudge naming only
    the answered request never bled into the sibling's budget."""
    pool = wf_runtime
    # One run owing TWO awaited requests from the same child session.
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    cid = child_session_id(run_id, "sha:a#0")
    await sessions_service.create_child_session(
        pool,
        session_id=cid,
        account_id="acc_wf",
        agent_id=wf_agent_id,
        environment_id="env_wf",
        agent_version=1,
        model=None,
        parent_run_id=run_id,
        surface=Surface([], [], []),
        vault_ids=[],
        request_id="sha:a#0",
        input="hi",
    )
    async with pool.acquire() as conn:
        # A second open awaited request edge + its delivered user message.
        await db_queries.append_event(
            conn,
            account_id="acc_wf",
            session_id=cid,
            kind="lifecycle",
            data={
                "event": "request_opened",
                "request_id": "sha:b#0",
                "awaited": True,
                "caller": {"kind": "run", "id": run_id},
            },
        )
        await db_queries.append_event(
            conn,
            account_id="acc_wf",
            session_id=cid,
            kind="message",
            data={
                "role": "user",
                "content": "second",
                "metadata": {
                    "request": {"request_id": "sha:b#0", "caller": {"kind": "run", "id": run_id}}
                },
            },
        )
        open_ids = await db_queries.get_open_request_ids(conn, cid, account_id="acc_wf")
    assert set(open_ids) == {"sha:a#0", "sha:b#0"}  # two obligations open

    # One idle turn nudges BOTH open requests together.
    r = await sessions_service.append_assistant_and_guard_quiescence(
        pool,
        cid,
        await _idle_assistant_turn(pool, cid),
        account_id="acc_wf",
    )
    assert r.nudged
    async with pool.acquire() as conn:
        # Answer A. From now on only B is open, so only B is nudged.
        wrote = await db_queries.write_response_if_absent(
            conn,
            cid,
            account_id="acc_wf",
            request_id="sha:a#0",
            is_error=False,
            result="done-A",
            error=None,
        )
        assert wrote
        count_a = await db_queries.count_request_nudges(
            conn, cid, account_id="acc_wf", request_id="sha:a#0"
        )
        count_b = await db_queries.count_request_nudges(
            conn, cid, account_id="acc_wf", request_id="sha:b#0"
        )
    assert count_a == 1 and count_b == 1  # each counts only nudges naming ITSELF

    # B is now the lone obligation. It needs BUDGET-1 MORE nudges to reach budget,
    # then the next idle turn auto-errors B alone. Crucially, no activity turn ever
    # resets B, and A's earlier resolution does not reset B's count.
    for _ in range(sessions_service.REQUEST_NUDGE_BUDGET - 1):
        r = await sessions_service.append_assistant_and_guard_quiescence(
            pool,
            cid,
            await _idle_assistant_turn(pool, cid),
            account_id="acc_wf",
        )
        assert r.nudged  # still under budget for B

    r = await sessions_service.append_assistant_and_guard_quiescence(
        pool,
        cid,
        await _idle_assistant_turn(pool, cid),
        account_id="acc_wf",
    )
    assert not r.nudged  # B's budget spent -> auto-error
    async with pool.acquire() as conn:
        resp_a = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:a#0"
        )
        resp_b = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:b#0"
        )
    assert resp_a is not None and resp_a["is_error"] is False  # A's real answer stands
    assert resp_b is not None and resp_b["error"] == {"kind": "no_return"}  # sibling not spared


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
        f"        await agent('go', agent_id={wf_agent_id!r})\n"
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

    prev_reg = runtime.inflight_tool_registry
    runtime.inflight_tool_registry = (
        None  # the guard must return before require_inflight_tool_registry
    )
    try:
        await run_session_step(sess.id)  # must NOT raise
    finally:
        runtime.inflight_tool_registry = prev_reg

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
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:reclaim#0")

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
    result = await sessions_service.append_assistant_and_guard_quiescence(
        pool, cid, await _idle_assistant_turn(pool, cid), account_id="acc_wf"
    )
    assert not result.nudged and result.autoerror_caller_run_id is None

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
    run. The service-level archive EAGERLY fails the child's open request through the
    write_child_response seam (error child_gone) + writes its child_done signal, so the
    run is sweep-visible within a tick — no waiting out the 1h agent deadline. The
    harvest then resolves the call as a catchable AgentError(child_gone)."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('go', agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    request_id = await _open_request_id(pool, child_id)  # capture BEFORE archiving

    # Suspended, no signal, not past the agent deadline → not yet sweep-visible.
    assert run_id not in await _needing(pool)

    # Operator archives the child mid-flight, via the SERVICE-level path under test.
    await sessions_service.archive_session(pool, child_id, account_id="acc_wf")

    # CORE #904 ASSERTION: the eager child_done signal makes the run sweep-visible
    # WITHOUT aging the wall-clock — it appears in the needs-step set immediately.
    assert run_id in await _needing(pool)

    # The request was failed immediately, before any further step ran.
    async with pool.acquire() as conn:
        resolved = await db_queries.derive_response(
            conn, child_id, account_id="acc_wf", request_id=request_id
        )
    assert resolved == {"result": None, "is_error": True, "error": {"kind": "child_gone"}}

    await run_workflow_step(run_id)  # harvest -> child_gone -> AgentError -> caught
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"caught": True, "kind": "child_gone"}


async def test_operator_deleted_child_resolves_run_as_child_gone(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Same totality guarantee when the child is hard-deleted. The service-level
    delete EAGERLY fails the open request through write_child_response FIRST (so its
    child_gone response + child_done signal commit before the cascade wipes the child's
    events), making the run sweep-visible within a tick. The signal survives the
    cascade (it lives in wf_run_signals, not the child's events); the harvest resolves
    child_gone from the signal, not a re-read of the now-absent child."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('go', agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return e.kind\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    request_id = await _open_request_id(pool, child_id)  # the agent call_key; capture BEFORE delete

    assert run_id not in await _needing(pool)

    await sessions_service.delete_session(pool, child_id, account_id="acc_wf")

    # Eagerness: the child_done signal makes the run sweep-visible immediately.
    assert run_id in await _needing(pool)

    async with pool.acquire() as conn:
        # The child_done signal survived the events cascade (lives in wf_run_signals).
        signal = await wf_queries.read_run_signal(conn, run_id, request_id)
        # The child session row is gone.
        with pytest.raises(NotFoundError):
            await db_queries.get_session_bare(conn, child_id, account_id="acc_wf")
    assert signal is not None and signal.kind == "child_done"

    await run_workflow_step(run_id)  # harvest -> child_gone -> AgentError -> caught
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed" and run.output == "child_gone"


async def test_archive_child_commits_response_and_child_done_atomically(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Acceptance #3: the child_gone response, its child_done signal, and the
    archived_at flip all commit in ONE transaction — observable together afterward."""
    pool = wf_runtime
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # spawn + suspend
    child_id = await _child_id_of(pool, run_id)
    request_id = await _open_request_id(pool, child_id)

    await sessions_service.archive_session(pool, child_id, account_id="acc_wf")

    async with pool.acquire() as conn:
        # (a) the request_response lifecycle event was written with error child_gone.
        response = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=request_id
        )
        # (b) the child_done signal exists for (run_id, request_id).
        signal = await wf_queries.read_run_signal(conn, run_id, request_id)
        # (c) the session's archived_at is set.
        archived_at = await conn.fetchval(
            "SELECT archived_at FROM sessions WHERE id = $1 AND account_id = $2",
            child_id,
            "acc_wf",
        )
    assert response is not None and response["is_error"] is True
    assert response["error"] == {"kind": "child_gone"}
    assert signal is not None and signal.kind == "child_done"
    assert archived_at is not None


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
        f"    return await parallel([lambda: agent('a', agent_id={wf_agent_id!r}),"
        f" lambda: agent('b', agent_id={wf_agent_id!r})])\n"
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
        f"    return await parallel([lambda: agent('a', agent_id={wf_agent_id!r}),"
        f" lambda: agent('b', agent_id={wf_agent_id!r})])\n"
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


async def test_lifetime_agent_cap_errors_at_the_over_cap_spawn(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A fan-out that would push the run past its lifetime agent-call cap errors with
    ``too_many_agents`` AT the over-cap child: the cap is enforced per-spawn (so a
    rejected cap never tips it), checked just before each ``create_child_session``. The
    children up to the cap are spawned, but the (cap + 1)-th child is NEVER created — the
    run errors the moment that spawn is attempted. Those at-cap children are orphans the
    quiescence sweep reclaims (idle, no sandbox until used); the over-cap child does not
    exist."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 2})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(3)])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # 3 > cap 2 → error AT the 3rd spawn

    # The cap-many children spawned (call_started journaled); the over-cap one did not.
    assert len(await _children_of(pool, run_id)) == 2  # exactly max_agent_calls, never the 3rd
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        # The over-cap child is never created — at most max_agent_calls session rows exist.
        spawned = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert spawned == 2  # the (cap + 1)-th child was never created
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
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(3)])\n"
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
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(2)])\n"
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

    await run_workflow_step(
        run_id
    )  # harvest child0; child1 still inflight; no new agent spawns this step
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"  # NOT errored — no new spawns


async def test_rejected_agent_cap_does_not_count_toward_quota(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A cap the spawn gate REJECTS (agent_not_found / bad_agent_call) spawns no child
    and must NOT count toward ``max_agent_calls``. With cap 2 and one prior real spawn,
    a ``parallel([agent('nope'), agent(real)])`` frontier has only ONE real spawn left
    (the good branch) — so the run must NOT trip ``too_many_agents``: the bad branch
    surfaces a catchable AgentError (→ barrier slot None) and the good branch completes.
    The buggy pre-count saw 2 new caps (1 prior + 2 > 2) and terminated the run."""
    from aios.config import get_settings
    from aios.tools import workflow_completion

    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 2})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)

    pool = wf_runtime
    # One prior real spawn, THEN the bad+good fan-out: prior=1, one real spawn left.
    script = (
        "async def main(input):\n"
        f"    await agent('first', agent_id={wf_agent_id!r})\n"
        "    return await parallel([\n"
        "        lambda: agent('x', agent_id='agent_nope'),\n"
        f"        lambda: agent('y', agent_id={wf_agent_id!r}),\n"
        "    ])\n"
    )
    run_id = await _make_run(pool, script)

    # Wake 1: spawn `first`, suspend. Answer it so the next wake replays to the parallel.
    await run_workflow_step(run_id)
    first_child = (await _children_of(pool, run_id))[0]
    rid_first = await _open_request_id(pool, first_child)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(
            first_child, {"request_id": rid_first, "value": "r_first"}
        )

    # Wake 2: harvest `first` → replay to the parallel frontier. prior=1, new caps={nope, y}.
    # The good 'y' is the only real spawn (count 2 ≤ 2 OK); 'nope' is rejected, NOT counted.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    # NOT terminated by too_many_agents — the rejected cap did not tip the quota.
    assert run is not None and run.status == "running"  # owes the catchable-error drive
    assert not any(
        e.type == "run_completed" and e.payload.get("error", {}).get("kind") == "too_many_agents"
        for e in events
    )
    errors = [e for e in events if e.type == "call_result" and e.payload.get("is_error")]
    assert len(errors) == 1 and errors[0].payload["error"]["kind"] == "agent_not_found"
    # Two real children total: `first` (harvested) + `y` (just spawned).
    children = await _children_of(pool, run_id)
    assert len(children) == 2

    # Drive the owed replay (throws the caught AgentError → re-suspend on `y`), then
    # answer `y` and complete. The bad branch is None; the good branch is `y`'s value.
    await run_workflow_step(run_id)
    y_child = children[1]
    rid_y = await _open_request_id(pool, y_child)
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(y_child, {"request_id": rid_y, "value": "r_y"})

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == [None, "r_y"]  # bad branch → None, good branch → its value


async def test_real_overquota_fanout_still_errors_too_many_agents(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """The guard still fires for genuine over-quota REAL spawns (it isn't disabled by the
    per-spawn counting): a fan-out of three real agents with cap 2 terminates the run with
    ``too_many_agents`` and the over-cap child is never created."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update={"workflow_max_agent_calls": 2})
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)

    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    return await parallel([\n"
        f"        lambda: agent('a', agent_id={wf_agent_id!r}),\n"
        f"        lambda: agent('b', agent_id={wf_agent_id!r}),\n"
        f"        lambda: agent('c', agent_id={wf_agent_id!r}),\n"
        "    ])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # 3 real > cap 2 → too_many_agents

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "too_many_agents"


# ─── #784 — per-run wave admission (bound concurrently in-flight agent children) ─


async def _started_agents(pool: asyncpg.Pool[Any], run_id: str) -> list[str]:
    """call_keys of every agent ``call_started`` (admitted children), in order."""
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    return [
        e.call_key
        for e in events
        if e.type == "call_started"
        and e.payload.get("capability") == "agent"
        and e.call_key is not None
    ]


async def _deferred_keys(pool: asyncpg.Pool[Any], run_id: str) -> list[str]:
    """call_keys of every ``frontier_deferred`` marker (un-admitted agent frontiers)."""
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    return [e.call_key for e in events if e.type == "frontier_deferred" and e.call_key is not None]


async def _resolve_one_admitted(pool: asyncpg.Pool[Any], run_id: str, value: Any) -> None:
    """Return-resolve the FIRST still-open admitted child of the run (frees one slot)."""
    from aios.tools import workflow_completion

    for cid in await _started_agents(pool, run_id):
        child = child_session_id(run_id, cid)
        async with pool.acquire() as conn:
            open_ids = await db_queries.get_open_request_ids(conn, child, account_id="acc_wf")
        if open_ids:
            with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
                await workflow_completion.return_handler(
                    child, {"request_id": open_ids[0], "value": value}
                )
            return
    raise AssertionError("no open admitted child to resolve")


def _wave_cap(monkeypatch: pytest.MonkeyPatch, **overrides: int) -> None:
    """Patch ``aios.workflows.step.get_settings`` to a model_copy with ``overrides``."""
    from aios.config import get_settings

    capped = get_settings().model_copy(update=overrides)
    monkeypatch.setattr("aios.workflows.step.get_settings", lambda: capped)


async def test_wave_admits_in_waves_as_children_resolve(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """AC1: with the per-run wave cap at 2, a parallel(5) fan-out admits exactly 2
    children per wake and journals the other 3 as ``frontier_deferred`` — then admits
    the rest in later waves as children resolve and free slots, completing with all 5
    values in branch order."""
    from aios.tools import workflow_completion

    _wave_cap(monkeypatch, workflow_max_inflight_children_per_run=2)
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(5)])\n"
    )
    run_id = await _make_run(pool, script)
    # Each admitted child returns a value derived from ITS call_key, so the final
    # output pins resume→branch routing (each branch got exactly its own child's value).
    values: dict[str, str] = {}

    async def _resolve_open(value_for: Any) -> bool:
        resolved_any = False
        for key in await _started_agents(pool, run_id):
            child = child_session_id(run_id, key)
            async with pool.acquire() as conn:
                open_ids = await db_queries.get_open_request_ids(conn, child, account_id="acc_wf")
            if open_ids:
                values[key] = value_for(key)
                with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
                    await workflow_completion.return_handler(
                        child, {"request_id": open_ids[0], "value": values[key]}
                    )
                resolved_any = True
        return resolved_any

    # Wake 1: 2 admitted, 3 deferred, still suspended.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"
    first_wave = await _started_agents(pool, run_id)
    assert len(first_wave) == 2
    assert len(await _deferred_keys(pool, run_id)) == 3

    # Resolve ONE admitted child + wake: a freed slot admits one deferred frontier
    # (3 admitted total); the deferred-marker count is unchanged (re-emitted deferred
    # frontiers ON CONFLICT no-op, and one fewer remains over-cap — net same rows).
    one_key = first_wave[0]
    one_child = child_session_id(run_id, one_key)
    async with pool.acquire() as conn:
        one_rid = (await db_queries.get_open_request_ids(conn, one_child, account_id="acc_wf"))[0]
    values[one_key] = f"r{one_key}"
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(
            one_child, {"request_id": one_rid, "value": values[one_key]}
        )
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"
    assert len(await _started_agents(pool, run_id)) == 3
    assert len(await _deferred_keys(pool, run_id)) == 3  # idempotent: no new marker rows

    # Drain: keep resolving every open admitted child + re-waking until complete.
    for _ in range(20):
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status == "completed":
            break
        assert await _resolve_open(lambda k: f"r{k}")  # progress possible each iteration
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    started = await _started_agents(pool, run_id)
    assert len(started) == 5  # all 5 eventually admitted
    # Output is the 5 child values in branch order — each branch routed to its child.
    assert run.output == [values[k] for k in started]


async def test_harvest_only_resuspend_with_deferred_outstanding_no_error(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """AC3: a wake that drives while deferred frontiers are still outstanding must NOT
    error and must NOT double-journal a ``frontier_deferred`` marker (the divergence
    guard sees a waiting agent, not a vanished one)."""
    _wave_cap(monkeypatch, workflow_max_inflight_children_per_run=2)
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(3)])\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # 2 admitted, 1 deferred
    assert len(await _started_agents(pool, run_id)) == 2
    assert await _deferred_keys(pool, run_id) != []

    # Resolve one admitted child so the next wake genuinely DRIVES (a quiet wake would
    # skip the host), with the deferred frontier still outstanding.
    deferred_before = sorted(await _deferred_keys(pool, run_id))
    await _resolve_one_admitted(pool, run_id, "v")
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    # Not errored — a deferred-but-not-re-emitted-as-vanished frontier is a WAIT.
    assert run is not None and run.status == "suspended"
    assert not any(e.type == "run_completed" and e.payload.get("is_error") for e in events)
    # The freed slot admitted the deferred frontier; its marker did not duplicate.
    deferred_after = await _deferred_keys(pool, run_id)
    assert len(deferred_after) == len(set(deferred_after))  # no duplicate markers
    assert set(deferred_after) <= set(deferred_before)  # never grew


async def test_wave_gate_does_not_mask_or_false_trip_lifetime_cap(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """AC3/H1: the wave gate neither masks nor false-trips the lifetime cap. H1 is
    enforced per-spawn at the gate (#779: rejected caps must not count) and only
    ADMITTED frontiers reach it, so with the wave cap at 2 and the LIFETIME cap at 3 a
    parallel(5) admits children in waves until the 4th real spawn attempt — then errors
    ``too_many_agents`` BEFORE that child is created: exactly 3 children ever exist,
    never the 4th (the cap is not MASKED by admission). And a deferred frontier consumes
    no lifetime quota while it waits — wake 1 defers 3 frontiers without erroring, and a
    later admission counts each exactly once (the cap is not FALSE-TRIPPED, no
    double-count across wakes)."""
    _wave_cap(
        monkeypatch,
        workflow_max_inflight_children_per_run=2,
        workflow_max_agent_calls=3,
    )
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(5)])\n"
    )
    run_id = await _make_run(pool, script)

    # Wake 1: 2 admitted (within both caps), 3 deferred. NOT errored — deferral alone
    # never trips the lifetime cap (only real spawns count toward it).
    await run_workflow_step(run_id)
    assert len(await _started_agents(pool, run_id)) == 2
    assert len(await _deferred_keys(pool, run_id)) == 3
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"

    # Drive admission waves: each freed slot admits one deferred frontier through the
    # per-spawn H1 gate, which fires at the 4th real spawn attempt.
    for _ in range(10):
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status == "errored":
            break
        await _resolve_one_admitted(pool, run_id, "v")
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "too_many_agents"
    # The cap was not masked by admission: exactly max_agent_calls children were ever
    # created — the over-cap 4th child never existed.
    assert len(await _started_agents(pool, run_id)) == 3
    assert children == 3


async def test_deferred_frontier_never_re_emitted_errors_nondeterministic(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """AC4: a ``frontier_deferred`` marker whose call_key the script can never re-emit
    is divergence — it is a waiting agent that vanished from the script, so the run
    errors ``nondeterministic_replay`` rather than stranding the frontier forever."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)  # the script emits a REAL gate key
    # Inject a journal whose only "open" key is a deferred frontier the script can't emit.
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn, account_id="acc_wf", run_id=run_id, type="run_started", payload={"input": None}
        )
        await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="frontier_deferred",
            call_key="sha:neveremitted#0",
            payload={"capability": "agent"},
        )

    await run_workflow_step(run_id)  # host emits the REAL gate key, not the deferred one
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    assert events[-1].type == "run_completed"
    assert events[-1].payload["error"]["kind"] == "nondeterministic_replay"


async def test_frontier_deferred_marker_is_idempotent_and_seq_gapless(
    monkeypatch: pytest.MonkeyPatch, wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A deferred frontier journals exactly ONE ``frontier_deferred`` row across the
    waves it waits through (the memo dedups on (run_id, call_key, type)), and the
    journal's seqs stay contiguous 1..N (the gapless-seq invariant)."""
    from aios.tools import workflow_completion

    _wave_cap(monkeypatch, workflow_max_inflight_children_per_run=2)
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        f"    return await parallel([lambda: agent(str(i), agent_id={wf_agent_id!r}) for i in range(4)])\n"
    )
    run_id = await _make_run(pool, script)

    await run_workflow_step(run_id)  # 2 admitted, 2 deferred
    deferred = await _deferred_keys(pool, run_id)
    assert len(deferred) == 2
    # The deferred frontier that will be admitted LAST waits across multiple waves.
    last_deferred = deferred[-1]

    # Drive several admission waves: resolve one open child per wake.
    for _ in range(20):
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status == "completed":
            break
        for cid in await _started_agents(pool, run_id):
            child = child_session_id(run_id, cid)
            async with pool.acquire() as conn:
                open_ids = await db_queries.get_open_request_ids(conn, child, account_id="acc_wf")
            if open_ids:
                with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
                    await workflow_completion.return_handler(
                        child, {"request_id": open_ids[0], "value": "v"}
                    )
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    # Exactly one frontier_deferred row for the long-waiting key (idempotent re-emit).
    fd_for_key = [
        e for e in events if e.type == "frontier_deferred" and e.call_key == last_deferred
    ]
    assert len(fd_for_key) == 1
    # Gapless seq across the whole journal.
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, len(seqs) + 1))


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
        f"        return await agent('go', agent_id={wf_agent_id!r})\n"
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
        f"        return await agent('go', agent_id={wf_agent_id!r})\n"
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
        f"        return await agent('go', agent_id={wf_agent_id!r})\n"
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
        f"        return await agent('go', agent_id={wf_agent_id!r})\n"
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
    script = f"async def main(input):\n    await agent('go', agent_id={wf_agent_id!r})\n    return 'done'\n"
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

    # Simulate the harvest that committed just before the crash. The step's
    # lease means a mid-step crash leaves 'running' (the park/terminal write
    # never happened) — suspended + fully-harvested is unreachable by design.
    async with pool.acquire() as conn:
        await wf_queries.append_run_event(
            conn,
            account_id="acc_wf",
            run_id=run_id,
            type="call_result",
            call_key=gate_key,
            payload={"result": "yes", "is_error": False},
        )
        await wf_queries.set_run_status(conn, run_id, "running", account_id="acc_wf")

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


async def test_create_run_stamps_host_semantics_epoch(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None
    assert run.host_semantics_epoch == HOST_SEMANTICS_EPOCH


async def test_epoch_mismatch_errors_before_replay(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_runs SET host_semantics_epoch = $1 WHERE id = $2",
            HOST_SEMANTICS_EPOCH - 1,
            run_id,
        )
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "errored"
    # Errored runs persist the author-facing error message as output (#926).
    assert "engine semantics changed" in (run.output or "")
    assert [event.type for event in events] == ["run_completed"]
    assert events[-1].payload["is_error"] is True
    assert events[-1].payload["error"]["kind"] == "engine_semantics_changed"
    assert "engine semantics changed" in events[-1].payload["output"]


async def test_epoch_mismatch_fails_child_open_requests(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent('hi', agent_id={wf_agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    children = await _children_of(pool, run_id)
    assert len(children) == 1
    child_id = children[0]
    request_id = (await _events(pool, run_id))[1][2]
    assert request_id is not None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_runs SET host_semantics_epoch = $1 WHERE id = $2",
            HOST_SEMANTICS_EPOCH - 1,
            run_id,
        )
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        response = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=request_id
        )
        signals = await wf_queries.list_run_signals(conn, run_id)
    assert response is not None
    assert response["is_error"] is True
    assert response["error"] == {"kind": "engine_semantics_changed"}
    assert any(s.call_key == request_id and s.kind == "child_done" for s in signals)


async def test_nondeterministic_replay_fails_child_open_requests(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    script = f"async def main(input):\n    return await agent('hi', agent_id={wf_agent_id!r})\n"
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    children = await _children_of(pool, run_id)
    assert len(children) == 1
    child_id = children[0]
    request_id = (await _events(pool, run_id))[1][2]
    assert request_id is not None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_runs SET script = $1, status = 'running' WHERE id = $2",
            "async def main(input):\n    return 1",
            run_id,
        )
    await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        response = await db_queries.read_request_response(
            conn, child_id, account_id="acc_wf", request_id=request_id
        )
    assert run is not None and run.status == "errored"
    assert response is not None
    assert response["is_error"] is True
    assert response["error"] == {"kind": "nondeterministic_replay"}


async def test_gate_parked_run_detects_epoch_mismatch_on_next_wake(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        suspended = await wf_queries.get_run_for_step(conn, run_id)
        await conn.execute(
            "UPDATE wf_runs SET host_semantics_epoch = $1 WHERE id = $2",
            HOST_SEMANTICS_EPOCH - 1,
            run_id,
        )
    assert suspended is not None and suspended.status == "suspended"

    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        completed = await wf_queries.get_run_completed_event(conn, run_id)
    assert run is not None and run.status == "errored"
    assert completed is not None
    assert completed.payload["error"]["kind"] == "engine_semantics_changed"


async def test_host_crash_on_suspended_gate_is_not_divergence(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A host crash/timeout (emitted=[]) on a run suspended at a gate must report
    the REAL infra cause, not a fabricated ``nondeterministic_replay`` — the
    divergence check runs only on a real replay, after the raised branch."""
    pool = wf_runtime
    two_gates = (
        "async def main(input):\n"
        "    return await parallel([lambda: gate({'g': 1}), lambda: gate({'g': 2})])\n"
    )
    run_id = await _make_run(pool, two_gates)
    await run_workflow_step(run_id)  # wake 1: opens both gates, suspends
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"
    # Resume ONE gate so wake 2 harvests (and therefore drives — a quiet wake
    # would skip the host); the other gate stays inflight across the crash.
    first_key = (await _events(pool, run_id))[1][2]
    assert first_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=first_key, result="yes")

    timeout = HostOutcome(
        kind="raised", error_kind="script_host_timeout", error_repr="deadline", emitted=[]
    )
    with mock.patch("aios.workflows.step.run_script_host", new=AsyncMock(return_value=timeout)):
        await run_workflow_step(run_id)  # wake 2: host killed before re-emitting the open gate

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
    # The step LEASE (#780): the crashed step left 'running', which the needs-step
    # sweep matches unconditionally — what makes the re-raise actually retriable.
    assert run.status == "running"
    assert run_id in await _needing(pool)


async def test_post_harvest_crash_leaves_lease_for_the_sweep(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """The filter's hardest loss mode (#780 red-team, critical): a step harvests
    its last pending signal into a call_result, then dies before the re-drive
    parks the run. With the harvest committed, no signal is unharvested and no
    call is inflight — only the per-step 'running' lease keeps the run
    sweep-visible. Without the lease this state was a permanent zombie."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)  # suspends on the gate
    gate_key = (await _events(pool, run_id))[1][2]
    assert gate_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")

    with (
        mock.patch(
            "aios.workflows.step.run_script_host",
            new=AsyncMock(side_effect=RuntimeError("worker died mid-step")),
        ),
        pytest.raises(RuntimeError),
    ):
        await run_workflow_step(run_id)

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
    assert run is not None and run.status == "running"  # the lease, not 'suspended'
    assert any(e.type == "call_result" for e in events)  # the harvest DID commit
    assert run_id in await _needing(pool)  # ...and the sweep still sees the run

    # The next (healthy) step replays with the journaled memo and completes.
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"


async def test_child_done_signal_is_atomic_with_the_response(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """Fault injection: if the child_done insert fails, the response write must
    roll back with it — two separate autocommits would re-open the exact lost-wake
    hole the marker exists to close (a committed response with no marker)."""
    from aios.tools import workflow_completion

    pool = wf_runtime
    run_id, cid = await _spawn_child(pool, wf_agent_id, "sha:at#0")
    with (
        mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()),
        mock.patch(
            "aios.db.queries.workflows.insert_run_signal",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        pytest.raises(RuntimeError),
    ):
        await workflow_completion.return_handler(cid, {"request_id": "sha:at#0", "value": 1})

    async with pool.acquire() as conn:
        response = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="sha:at#0"
        )
    assert response is None  # rolled back WITH the failed signal — still unanswered

    # A retry (signal write healthy again) responds normally, not 'duplicate'.
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        res = await workflow_completion.return_handler(cid, {"request_id": "sha:at#0", "value": 1})
    assert res == {"status": "returned"}
    async with pool.acquire() as conn:
        signals = await wf_queries.list_run_signals(conn, run_id)
    assert [(s.call_key, s.kind) for s in signals] == [("sha:at#0", "child_done")]


async def test_lease_flips_before_the_harvest(wf_runtime: asyncpg.Pool[Any]) -> None:
    """Ordering pin: the 'running' lease must commit BEFORE any harvest write. If
    the flip itself dies, nothing may have been harvested yet — the resume signal
    is still unharvested and the sweep still sees the run. (A flip moved after
    the harvest would leave suspended + fully-harvested on this crash: the
    exact zombie the lease exists to prevent.)"""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)
    gate_key = (await _events(pool, run_id))[1][2]
    assert gate_key is not None
    await service.resume_gate(pool, run_id=run_id, call_key=gate_key, result="yes")

    real_set_run_status = wf_queries.set_run_status

    async def exploding_flip(conn: Any, rid: str, status: WfRunStatus, *, account_id: str) -> None:
        if status == "running":
            raise RuntimeError("died at the lease flip")
        await real_set_run_status(conn, rid, status, account_id=account_id)

    with (
        mock.patch("aios.workflows.step.wf_queries.set_run_status", new=exploding_flip),
        pytest.raises(RuntimeError),
    ):
        await run_workflow_step(run_id)

    events = await _events(pool, run_id)
    assert not any(t == "call_result" for _s, t, _k in events)  # nothing harvested yet
    assert run_id in await _needing(pool)  # the resume signal is still sweep-visible


async def test_quiet_wake_of_a_parked_run_skips_the_replay(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A wake of a parked run with nothing new must NOT reship + replay (the
    lease/sweep resonance guard): the host is never spawned, and the run parks
    straight back. Without this, a replay outliving the 30s sweep interval
    re-arms the next tick's wake forever."""
    pool = wf_runtime
    run_id = await _make_run(pool, _GATE_SCRIPT)
    await run_workflow_step(run_id)  # parks on the gate
    with mock.patch(
        "aios.workflows.step.run_script_host",
        new=AsyncMock(side_effect=AssertionError("quiet wake must not drive the host")),
    ) as host:
        await run_workflow_step(run_id)  # a sweep-style wake with nothing to do
    host.assert_not_awaited()
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "suspended"  # parked straight back


async def test_large_fanout_completes_and_parks_quiet(wf_runtime: asyncpg.Pool[Any]) -> None:
    """#780 acceptance proxy: 200 resolved calls through the REAL step path. The
    discriminating assertions are the needs-step ones — a parked 200-wide fan-out
    is QUIET (the old blanket sweep re-drove it, full memo reship, every tick) and
    flips hot exactly while resumes sit unharvested."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    results = await parallel([(lambda i=i: gate({'i': i})) for i in range(200)])\n"
        "    return [r['v'] for r in results]\n"  # ordered: pins resume->branch routing
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)  # parks on a 200-wide frontier

    gate_keys = [k for _s, t, k in await _events(pool, run_id) if t == "call_started"]
    assert len(gate_keys) == 200 and all(k is not None for k in gate_keys)
    assert run_id not in await _needing(pool)  # parked on gates: quiet

    # A real fan-in IS concurrent — resume all 200 gates at once.
    await asyncio.gather(
        *(
            service.resume_gate(pool, run_id=run_id, call_key=key, result={"v": i})
            for i, key in enumerate(gate_keys)
            if key is not None
        )
    )
    assert run_id in await _needing(pool)  # unharvested resumes: hot

    await run_workflow_step(run_id)  # harvests all 200, replays past them, completes
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == list(range(200))  # each resume reached ITS branch, in order
    assert run_id not in await _needing(pool)  # terminal: out of the sweep for good


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
    _run_id, cid = await _spawn_child(pool, wf_agent_id, "req:a", output_schema=_OBJ_SCHEMA)
    async with pool.acquire() as conn:
        for rid, schema in (("req:b", schema_b), ("req:none", None)):
            # The schema is read off the trusted request_opened edge (#1131), not the
            # display user-message blob.
            edge: dict[str, Any] = {"event": "request_opened", "request_id": rid, "awaited": True}
            if schema is not None:
                edge["output_schema"] = schema
            await db_queries.append_event(
                conn,
                account_id="acc_wf",
                session_id=cid,
                kind="lifecycle",
                data=edge,
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
    assert isinstance(bad, ToolResult) and bad.is_error
    assert isinstance(bad.content, str) and "schema" in bad.content.lower()
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
    wake.assert_awaited_once_with(run_id, batch=True)
    async with pool.acquire() as conn:
        resp = await db_queries.read_request_response(
            conn, cid, account_id="acc_wf", request_id="se:1"
        )
    assert resp is not None and resp["result"] == {"answer": "yes"}


async def test_malformed_output_schema_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A malformed output_schema surfaces as a catchable ``AgentError(bad_agent_call)``
    at the await (preserving the schema message) and never spawns a child."""
    pool = wf_runtime
    bad_schema = {"type": "not-a-real-type"}
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('hi', output_schema={bad_schema!r}, agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output["kind"] == "bad_agent_call"
    assert run.output["msg"].startswith("agent() output_schema is not a valid JSON Schema")
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["error"]["kind"] == "bad_agent_call"
    assert children == 0  # rejected before any spawn


async def test_non_object_output_schema_errors_the_run(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    """A non-object output_schema (a bare boolean — valid JSON Schema, but ``false``
    rejects every value and ``true`` disables enforcement) is a degenerate author input:
    surface a catchable ``AgentError(bad_agent_call)`` rather than spawn a child that can
    never return (or one with no enforcement)."""
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('hi', output_schema=False, agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output["kind"] == "bad_agent_call"
    assert "object schema" in run.output["msg"]
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["error"]["kind"] == "bad_agent_call"
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
        f"    return await agent('task', output_schema={_OBJ_SCHEMA!r}, agent_id={wf_agent_id!r})\n"
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
    script = f"async def main(input):\n    return await agent('pick', output_schema={schema!r}, agent_id={wf_agent_id!r})\n"
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
    script = f"async def main(input):\n    return await agent('x', output_schema={schema!r}, agent_id={wf_agent_id!r})\n"
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
    author-facing at the spawn gate (a catchable ``AgentError``), before any child
    spawns."""
    pool = wf_runtime
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}, "required": ["a"]}
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('x', output_schema={schema!r}, agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n"
        "    return {'caught': False}\n"
    )
    run_id = await _make_run(pool, script)
    await run_workflow_step(run_id)
    await run_workflow_step(run_id)
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        events = await wf_queries.list_run_events(conn, run_id)
        children = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1", run_id
        )
    assert run is not None and run.status == "completed"
    assert run.output["kind"] == "bad_agent_call"
    assert "reference" in run.output["msg"].lower()
    cr = next(e for e in events if e.type == "call_result")
    assert cr.payload["error"]["kind"] == "bad_agent_call"
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

    in_mem = InMemoryConnector()
    with app.replace_connector(in_mem) as _patched:
        await defer_wake(pool, fg.id, account_id="acc_wf", cause="message")
        await defer_wake(pool, cid, account_id="acc_wf", cause="message")
        priorities = {j["args"]["session_id"]: j["priority"] for j in in_mem.jobs.values()}
    assert priorities[fg.id] == _FOREGROUND_PRIORITY  # real foreground → default
    assert priorities[cid] == _BACKGROUND_PRIORITY  # real background child → demoted


# ─── await_task (run arm) — the one awaiter, runs backing ──────────────


async def _await_run(
    pool: asyncpg.Pool[Any], db_url: str, run_id: str, *, account_id: str, timeout_seconds: float
) -> Any:
    """Drive the unified awaiter on a run servicer (request_id is run-irrelevant)."""
    return await tasks_service.await_task(
        pool,
        db_url,
        servicer_kind="run",
        servicer_id=run_id,
        request_id=None,
        account_id=account_id,
        timeout_seconds=timeout_seconds,
    )


async def test_await_run_returns_when_already_completed(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A terminal run is returned immediately (the first post-subscribe read sees it):
    ``outcome='ok'`` + the script's ``result``, no error."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    await run_workflow_step(run_id)  # pure script → completed in one wake
    resp = await _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5)
    assert resp.outcome == "ok"
    assert resp.result == 1 and resp.error is None


async def test_await_run_surfaces_cancelled_as_cancelled_not_false_success(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A cancelled run resolves as ``outcome='cancelled'`` (``error.kind='cancelled'``), not
    the ``{ok: null}`` false-success a status-blind awaiter would surface — a cancelled run
    deliberately writes no ``request_response``."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    await wf_service.cancel_run(pool, run_id=run_id, account_id="acc_wf")
    await run_workflow_step(run_id)  # harvest the cancel signal → finalize cancelled
    resp = await _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5)
    assert resp.outcome == "cancelled" and resp.error == {"kind": "cancelled"}


async def test_await_run_surfaces_sync_def_author_error_message(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A malformed workflow entrypoint returns the host's author-facing diagnostic,
    not just an opaque ``author_exception`` kind."""
    pool = wf_runtime
    run_id = await _make_run(pool, "def main(input):\n    return 1")
    await run_workflow_step(run_id)  # sync def → errored
    resp = await _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5)
    assert resp.outcome == "errored" and resp.result is None
    assert resp.error == {
        "kind": "author_exception",
        "message": "WorkflowScriptError: workflow script must define `async def main(input)`",
    }


async def test_await_run_surfaces_runtime_author_traceback_line(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A mid-script raise carries type, message, and author-visible line number while
    host implementation frames are sanitized out."""
    pool = wf_runtime
    run_id = await _make_run(
        pool,
        "async def main(input):\n    x = 1\n    raise ValueError('boom')\n    return x\n",
    )
    await run_workflow_step(run_id)  # raise → errored
    resp = await _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=5)
    assert resp.outcome == "errored" and resp.result is None
    assert resp.error is not None
    assert resp.error["kind"] == "author_exception"
    assert resp.error["message"] == "ValueError: boom"
    tb = resp.error["traceback"]
    assert 'File "<workflow>", line 3, in main' in tb
    assert "raise ValueError('boom')" in tb
    assert "wf_script_host.py" not in tb


async def test_await_run_times_out_on_non_terminal_run(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """A never-stepped (``pending``) run that doesn't finish within the budget returns
    ``outcome=None`` (still pending) — the re-poll contract, no archive/mutation."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")  # created, not stepped
    resp = await _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=0.1)
    assert resp.outcome is None and resp.result is None and resp.error is None


async def test_await_run_wakes_on_completion_during_wait(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """Completion-during-subscribe: the await blocks on a pending run; a concurrent step
    drives it terminal; the run_completed notify wakes the await → it returns the record
    (not a timeout). The LISTEN-before-read ordering is what closes the race."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 7")

    async def _complete_soon() -> None:
        await asyncio.sleep(0.1)  # let the awaiter subscribe + first-read (pending) first
        await run_workflow_step(run_id)

    resp, _ = await asyncio.gather(
        _await_run(pool, migrated_db_url, run_id, account_id="acc_wf", timeout_seconds=10),
        _complete_soon(),
    )
    assert resp.outcome == "ok" and resp.result == 7


async def test_await_run_cross_tenant_404(
    wf_runtime: asyncpg.Pool[Any], migrated_db_url: str
) -> None:
    """The account scope is checked up front (before subscribing): a foreign account 404s."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1")
    with pytest.raises(NotFoundError):
        await _await_run(pool, migrated_db_url, run_id, account_id="acc_other", timeout_seconds=1)


async def test_derive_run_response_reads_the_terminal_record(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """The harvest resolver reads a sub-run's outcome off its terminal record (§3.6).

    No separate ``request_response`` event: ``completed → ok`` (the run_completed
    output), ``errored → error``, ``cancelled → cancelled`` (the load-bearing case — a
    cancelled sub-run must NOT resolve as the ``child_gone`` a gated-off response implied,
    nor as a false-success ``{ok: null}``), and a still-running sub-run → ``None`` (pending).
    """
    pool = wf_runtime
    ok_run = await _make_run(pool, "async def main(input):\n    return 7", name="ok")
    await run_workflow_step(ok_run)
    err_run = await _make_run(pool, "def main(input):\n    return 1", name="err")  # sync → errored
    await run_workflow_step(err_run)
    cancelled_run = await _make_run(pool, "async def main(input):\n    return 1", name="cancel")
    await wf_service.cancel_run(pool, run_id=cancelled_run, account_id="acc_wf")
    await run_workflow_step(cancelled_run)  # harvest the cancel → finalize cancelled
    pending_run = await _make_run(
        pool, "async def main(input):\n    return 1", name="pending"
    )  # never stepped

    async with pool.acquire() as conn:
        ok = await wf_queries.derive_run_response(conn, ok_run, account_id="acc_wf")
        err = await wf_queries.derive_run_response(conn, err_run, account_id="acc_wf")
        cancelled = await wf_queries.derive_run_response(conn, cancelled_run, account_id="acc_wf")
        pending = await wf_queries.derive_run_response(conn, pending_run, account_id="acc_wf")

    assert ok == {"result": 7, "is_error": False, "error": None}
    assert err is not None and err["is_error"] is True and err["result"] is None
    assert err["error"]["kind"] == "author_exception"
    # The cancel semantic the §3.6 merge had to preserve: cancelled, not child_gone.
    assert cancelled == {"result": None, "is_error": True, "error": {"kind": "cancelled"}}
    assert pending is None


# ─── tool() — a run invokes its declared network/credential tools (slice 2) ───

_WEB_SCRIPT = (
    "async def main(input):\n    r = await tool('web_search', {'query': 'q'})\n    return r\n"
)


async def _make_tool_run(
    pool: asyncpg.Pool[Any],
    script: str,
    *,
    tools: list[ToolSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    vault_ids: list[str] | None = None,
    name: str = "wt",
) -> str:
    """A run whose workflow declares ``tools``/``http_servers``; the run snapshots them."""
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name=name,
            script=script,
            tools=tools,
            http_servers=http_servers,
        )
    run = await service.create_run(
        pool,
        account_id="acc_wf",
        workflow_id=wf.id,
        environment_id="env_wf",
        vault_ids=vault_ids,
    )
    return run.id


async def _drain_tool_tasks() -> None:
    """Await any fire-and-forget tool tasks (``defer_run_wake`` is patched, so they don't
    self-wake; the test drives the harvest step itself)."""
    tasks = list(run_tools._INFLIGHT.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _capturing_client(captured: dict[str, Any]) -> type:
    """A stand-in for ``httpx.AsyncClient`` that records the outbound request + returns 200."""

    class _Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: Any) -> bool:
            return False

        async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return httpx.Response(200, json={"ok": True})

    return _Client


async def _get_run(pool: asyncpg.Pool[Any], run_id: str) -> Any:
    async with pool.acquire() as conn:
        return await wf_queries.get_run_for_step(conn, run_id)


async def _call_starteds(pool: asyncpg.Pool[Any], run_id: str) -> list[Any]:
    async with pool.acquire() as conn:
        return [
            e for e in await wf_queries.list_run_events(conn, run_id) if e.type == "call_started"
        ]


async def test_tool_web_search_park_harvest_replay(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_tool_run(pool, _WEB_SCRIPT, tools=[ToolSpec(type="web_search")])
    with mock.patch(
        "aios.workflows.run_tools.web_search_handler",
        new=AsyncMock(return_value={"results": ["R"]}),
    ):
        # Wake 1: drive to the tool and park; the worker task is launched after commit.
        await run_workflow_step(run_id)
        run = await _get_run(pool, run_id)
        assert run is not None and run.status == "suspended"
        assert [t for _s, t, _k in await _events(pool, run_id)] == ["run_started", "call_started"]
        cs = (await _call_starteds(pool, run_id))[0]
        assert cs.payload["capability"] == "tool" and cs.payload["tool_name"] == "web_search"

        await _drain_tool_tasks()  # the task writes its tool_result signal

        # Wake 2: harvest the signal → fast-forward past the tool → complete.
        await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"results": ["R"]}
    assert [t for _s, t, _k in await _events(pool, run_id)] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]


async def test_tool_undeclared_resolves_as_recoverable_error(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    # web_search is NOT declared on the workflow → a recoverable error value; run COMPLETES.
    run_id = await _make_tool_run(pool, _WEB_SCRIPT, tools=[], name="wt-undeclared")
    await run_workflow_step(run_id)  # park at the tool (no handler patch — gating short-circuits)
    await _drain_tool_tasks()  # task: invoke_run_tool → gating error → signal {"error": …}
    await run_workflow_step(run_id)  # harvest → complete with the error value
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"  # recoverable, NOT errored
    assert isinstance(run.output, dict) and "error" in run.output


async def test_tool_parallel_fanout(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    rs = await parallel([\n"
        "        lambda: tool('web_search', {'query': 'a'}),\n"
        "        lambda: tool('web_search', {'query': 'b'}),\n"
        "        lambda: tool('web_search', {'query': 'c'}),\n"
        "    ])\n"
        "    return rs\n"
    )
    run_id = await _make_tool_run(
        pool, script, tools=[ToolSpec(type="web_search")], name="wt-fanout"
    )

    def _echo(_sid: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"q": args["query"]}

    with mock.patch(
        "aios.workflows.run_tools.web_search_handler", new=AsyncMock(side_effect=_echo)
    ):
        await run_workflow_step(run_id)  # parks with 3 tool frontiers, 3 tasks launched
        cs = await _call_starteds(pool, run_id)
        assert len(cs) == 3 and all(e.payload["capability"] == "tool" for e in cs)
        await _drain_tool_tasks()  # 3 tasks signal
        await run_workflow_step(run_id)  # harvest all 3 → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == [{"q": "a"}, {"q": "b"}, {"q": "c"}]


async def test_tool_crash_redispatches(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    run_id = await _make_tool_run(
        pool, _WEB_SCRIPT, tools=[ToolSpec(type="web_search")], name="wt-crash"
    )

    block = asyncio.Event()  # never set — the first task blocks until cancelled

    async def _blocked(_sid: str, _args: dict[str, Any]) -> dict[str, Any]:
        await block.wait()
        return {"never": True}

    with mock.patch("aios.workflows.run_tools.web_search_handler", new=_blocked):
        await run_workflow_step(run_id)  # parks; the task blocks before it can signal
        # Simulate a hard worker crash: cancel the in-flight task + drop the registry.
        for task in list(run_tools._INFLIGHT.values()):
            task.cancel()
        await asyncio.gather(*list(run_tools._INFLIGHT.values()), return_exceptions=True)
        run_tools._INFLIGHT.clear()

    # No signal exists. Wake 2 (the periodic-sweep re-wake): harvest finds no signal + no
    # live task → re-dispatch — without journaling a second call_started.
    with mock.patch(
        "aios.workflows.run_tools.web_search_handler", new=AsyncMock(return_value={"ok": 1})
    ):
        await run_workflow_step(run_id)
        assert len(await _call_starteds(pool, run_id)) == 1  # exactly one — no double-open
        await _drain_tool_tasks()  # the re-dispatched task signals
        await run_workflow_step(run_id)  # harvest → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {"ok": 1}
    assert [t for _s, t, _k in await _events(pool, run_id)] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]


async def test_tool_http_request_credential_e2e(wf_runtime: asyncpg.Pool[Any]) -> None:
    """The payoff: a run calls http_request and the Authorization header is authored from the
    run's OWN bound vault, via resolve_auth_for_target_url_run, with no agent in the loop."""
    pool = wf_runtime
    box = CryptoBox(os.urandom(32))
    prev_box = runtime.crypto_box
    runtime.crypto_box = box
    try:
        base_url = "https://api.example/v1"
        vault = await vaults_service.create_vault(
            pool, account_id="acc_wf", display_name="v", metadata={}
        )
        await vaults_service.create_vault_credential(
            pool,
            box,
            account_id="acc_wf",
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url=base_url, auth_type="bearer_header", token=SecretStr("tok-XYZ")
            ),
        )
        http_servers = [
            HttpServerSpec(
                name="api", base_url=base_url, routes=[HttpRouteSpec(path_pattern="/things/*")]
            )
        ]
        script = (
            "async def main(input):\n"
            "    r = await tool('http_request',"
            " {'server_ref': 'api', 'path': '/things/1', 'method': 'GET'})\n"
            "    return r\n"
        )
        run_id = await _make_tool_run(
            pool,
            script,
            tools=[ToolSpec(type="http_request")],
            http_servers=http_servers,
            vault_ids=[vault.id],
            name="wt-http",
        )
        captured: dict[str, Any] = {}
        with (
            mock.patch("aios.tools.http_request.httpx.AsyncClient", _capturing_client(captured)),
            mock.patch("aios.tools.http_request.is_safe_url", return_value=True),
        ):
            await run_workflow_step(run_id)  # park at the tool
            await (
                _drain_tool_tasks()
            )  # task: _do_http_request → run resolver authors header → httpx
            await run_workflow_step(run_id)  # harvest → complete
        assert captured["headers"]["Authorization"] == "Bearer tok-XYZ"
        run = await _get_run(pool, run_id)
        assert run is not None and run.status == "completed"
        assert run.output["status"] == 200
    finally:
        runtime.crypto_box = prev_box


# ─── tool('bash', …) — the run-side sandbox executor (#988) ───────────────────
#
# bash rides the EXISTING ``tool`` capability (capability="tool", tool_name="bash")
# but its handler needs a provisioned container, so the step routes it by execution
# class to ``run_sandbox`` instead of the worker tool path. These tests drive the
# real ``run_workflow_step`` against a real Postgres with a ``FakeBackend`` standing
# in for Docker — the integration wiring (routing by ``tool_executes_class``,
# shared ``run_tools._INFLIGHT``, the bare-bash-dict / flat-``{"error"}`` value, the
# park/harvest/replay loop) is the surface under test. The pure-executor assertions
# (preamble byte-shape, timeout clamp, etc.) live in ``tests/unit/test_run_sandbox``.

import hashlib  # noqa: E402
import shlex  # noqa: E402

from aios.sandbox.backends.base import (  # noqa: E402
    CommandResult,
    Mount,
    SandboxBackendError,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.registry import SandboxRegistry  # noqa: E402
from aios.sandbox.spec import ProvisioningPlan  # noqa: E402
from tests.helpers.sandbox import FakeBackend  # noqa: E402

_BASH_SCRIPT = (
    "async def main(input):\n    r = await tool('bash', {'command': 'echo hi'})\n    return r\n"
)


def _fake_run_plan(run_id: str) -> ProvisioningPlan:
    """A minimal :class:`ProvisioningPlan` for a run sandbox — no real Docker/CA/broker.

    ``build_spec_from_run`` is patched to return this, so ``get_or_provision_run``
    exercises only ``backend.create`` + the cache logic against the FakeBackend.
    """
    from pathlib import Path

    from aios.models.environments import EnvironmentConfig

    spec = SandboxSpec(
        session_id=run_id,
        instance_id="inst_test",
        workspace=Mount(host_path=Path("/tmp/wfr-ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="img:test",
    )
    return ProvisioningPlan(
        spec=spec,
        env_config=EnvironmentConfig(),
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
        env_var_credentials=(),
        secret_proxy=None,
    )


@pytest.fixture
async def wf_sandbox_runtime(wf_runtime: asyncpg.Pool[Any]) -> AsyncIterator[Any]:
    """``wf_runtime`` + a :class:`SandboxRegistry` over a :class:`FakeBackend` on
    ``runtime.sandbox_registry``, with the heavy provision path (build_spec_from_run,
    egress CA / package install) patched out so ``get_or_provision_run`` exercises
    only the backend.create + cache logic. The bash task's own ``defer_run_wake`` is
    patched (the harvest is driven manually), and the SHARED ``run_tools._INFLIGHT``
    is cleared both sides. Yields ``(pool, backend)``."""
    pool = wf_runtime
    backend = FakeBackend()
    registry = SandboxRegistry(backend)
    prev = runtime.sandbox_registry
    runtime.sandbox_registry = registry
    run_tools._INFLIGHT.clear()  # bash shares run_tools._INFLIGHT (class-agnostic)
    with (
        mock.patch(
            "aios.sandbox.registry.build_spec_from_run",
            new=AsyncMock(side_effect=lambda run_id: _fake_run_plan(run_id)),
        ),
        mock.patch("aios.sandbox.registry.install_egress_ca", new=AsyncMock()),
        mock.patch("aios.sandbox.registry.install_packages", new=AsyncMock()),
        mock.patch("aios.workflows.run_sandbox.defer_run_wake", new=AsyncMock()),
    ):
        try:
            yield pool, backend
        finally:
            run_tools._INFLIGHT.clear()
            runtime.sandbox_registry = prev


async def _drain_sandbox_tasks() -> None:
    """Await any fire-and-forget bash tasks (they live in the SHARED
    ``run_tools._INFLIGHT``; ``defer_run_wake`` is patched, so they don't self-wake)."""
    tasks = list(run_tools._INFLIGHT.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _backend_exec_count(backend: FakeBackend) -> int:
    return sum(1 for verb, _ in backend.calls if verb == "exec")


def _backend_create_count(backend: FakeBackend) -> int:
    return sum(1 for verb, _ in backend.calls if verb == "create")


def _execed_commands(backend: FakeBackend) -> list[str]:
    """The ``command`` string handed to ``backend.exec`` for each exec call, in order."""
    return [kw["command"] for verb, kw in backend.calls if verb == "exec"]


# (a) ──────────────────────────────────────────────────────────────────────────
async def test_bash_dispatch_signal_one_call_result(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """The headline park/harvest/replay round-trip for bash: wake 1 journals ONE
    ``call_started{capability:"tool",tool_name:"bash"}`` and parks; the task writes
    one ``tool_result`` signal; wake 2's harvest folds one ``call_result`` carrying
    the BARE bash dict; the run completes; exec ran exactly once."""
    pool, backend = wf_sandbox_runtime
    backend.next_result = CommandResult(
        exit_code=0, stdout="hi\n", stderr="", timed_out=False, truncated=False
    )
    run_id = await _make_tool_run(
        pool, _BASH_SCRIPT, tools=[ToolSpec(type="bash")], name="wt-bash-a"
    )

    # Wake 1: drive to the bash frontier and park; the task launches after commit.
    await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "suspended"
    started = await _call_starteds(pool, run_id)
    assert len(started) == 1
    cs = started[0]
    assert cs.payload["capability"] == "tool" and cs.payload["tool_name"] == "bash"
    # The journaled command is the verbatim author command (no preamble).
    assert cs.payload["input"] == {"command": "echo hi"}

    await _drain_sandbox_tasks()  # the task writes its tool_result signal

    # Wake 2: harvest the signal → fast-forward → complete with the BARE bash dict.
    await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert run.output == {
        "exit_code": 0,
        "stdout": "hi\n",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }
    # Exactly ONE call_result; clean journal; exec ran exactly once.
    assert [t for _s, t, _k in await _events(pool, run_id)] == [
        "run_started",
        "call_started",
        "call_result",
        "run_completed",
    ]
    assert _backend_exec_count(backend) == 1


# (b) ──────────────────────────────────────────────────────────────────────────
async def test_bash_memo_law_redrive_while_inflight_execs_once(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """Re-driving a run while its bash task is still in-flight must NOT relaunch: the
    harvest's class-agnostic ``has_inflight`` guard suppresses the double-dispatch, so
    exec is entered exactly once."""
    pool, backend = wf_sandbox_runtime
    block = asyncio.Event()  # released only once the test lets exec finish
    entered = asyncio.Event()  # set when the task is blocking IN exec
    exec_count = 0  # patched exec doesn't populate backend.calls — count here

    async def _blocked(*_a: Any, **_k: Any) -> CommandResult:
        nonlocal exec_count
        exec_count += 1
        entered.set()
        await block.wait()
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)

    run_id = await _make_tool_run(
        pool, _BASH_SCRIPT, tools=[ToolSpec(type="bash")], name="wt-bash-b"
    )
    with mock.patch.object(backend, "exec", new=_blocked):
        await run_workflow_step(run_id)  # parks; launches the task
        await asyncio.wait_for(entered.wait(), timeout=5)  # task now blocked IN exec
        assert exec_count == 1
        call_key = (await _call_starteds(pool, run_id))[0].call_key
        assert run_tools.has_inflight(run_id, call_key)
        # Re-drive while in-flight: the harvest sees has_inflight True → no relaunch.
        await run_workflow_step(run_id)
        assert exec_count == 1  # still just the one in-flight exec
        block.set()
        await _drain_sandbox_tasks()
        await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert exec_count == 1


# (c) ──────────────────────────────────────────────────────────────────────────
async def test_bash_crash_path_at_least_once(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """A ``call_started`` with NO signal and NO live task (the crash signature)
    re-launches on harvest — without journaling a second ``call_started`` — so the
    command runs at-least-once. Total execs across crash + re-dispatch == 2."""
    pool, backend = wf_sandbox_runtime
    block = asyncio.Event()  # never set — the first task blocks until cancelled
    entered = asyncio.Event()
    first_exec_count = 0

    async def _blocked(*_a: Any, **_k: Any) -> CommandResult:
        nonlocal first_exec_count
        first_exec_count += 1
        entered.set()
        await block.wait()
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)

    run_id = await _make_tool_run(
        pool, _BASH_SCRIPT, tools=[ToolSpec(type="bash")], name="wt-bash-c"
    )
    with mock.patch.object(backend, "exec", new=_blocked):
        await run_workflow_step(run_id)  # parks; launches the task
        await asyncio.wait_for(entered.wait(), timeout=5)  # task blocks IN exec
        assert first_exec_count == 1
        # Simulate a hard worker crash: cancel the in-flight task + drop the registry map.
        for task in list(run_tools._INFLIGHT.values()):
            task.cancel()
        await asyncio.gather(*list(run_tools._INFLIGHT.values()), return_exceptions=True)
        run_tools._INFLIGHT.clear()

    # No signal exists. Wake 2 (sweep re-wake): harvest finds no signal + no live task
    # → re-dispatch — without a second call_started. The re-dispatched task uses the
    # REAL (un-patched) FakeBackend.exec, recorded in backend.calls.
    backend.next_result = CommandResult(
        exit_code=0, stdout="2nd\n", stderr="", timed_out=False, truncated=False
    )
    await run_workflow_step(run_id)
    assert len(await _call_starteds(pool, run_id)) == 1  # exactly one — no double-open
    assert first_exec_count + _backend_exec_count(backend) == 2
    assert _backend_exec_count(backend) == 1  # the re-dispatch ran exactly once
    await _drain_sandbox_tasks()
    await run_workflow_step(run_id)  # harvest the re-dispatch → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"


# (c-prime) ─────────────────────────────────────────────────────────────────────
async def test_bash_completed_call_not_reexeced_on_nonterminal_redrive(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """The harvest's memo-skip on a NON-terminal run: a bash call already folded into
    the memo is NEVER re-execed when its run is re-driven while still running — only
    the genuinely-inflight call is. A two-call script: the FIRST completes (in the
    memo), the SECOND blocks in exec (keeping the run non-terminal). Dropping call 1's
    signal makes the memo-skip load-bearing — without ``call_key not in memo`` the
    harvest would see the crash signature (no signal + no live task) and re-exec it."""
    pool, backend = wf_sandbox_runtime
    script = (
        "async def main(input):\n"
        "    a = await tool('bash', {'command': 'echo one'})\n"
        "    b = await tool('bash', {'command': 'echo two'})\n"
        "    return [a, b]\n"
    )
    second_entered = asyncio.Event()
    second_block = asyncio.Event()  # never set — the SECOND task blocks, keeping run live
    execed: list[str] = []  # patched exec doesn't populate backend.calls — record here

    async def _exec(_handle: Any, command: str, **_k: Any) -> CommandResult:
        execed.append(command)
        if "echo two" in command:
            second_entered.set()
            await second_block.wait()  # park here so the run stays non-terminal
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)

    run_id = await _make_tool_run(
        pool, script, tools=[ToolSpec(type="bash")], name="wt-bash-cprime"
    )
    with mock.patch.object(backend, "exec", new=_exec):
        await run_workflow_step(run_id)  # park at call 1; launch task 1 (echo one)
        await _drain_sandbox_tasks()  # task 1 writes its signal
        await run_workflow_step(run_id)  # harvest call 1 → memo; advance to call 2; launch task 2
        await asyncio.wait_for(second_entered.wait(), timeout=5)  # task 2 blocks IN exec

        run = await _get_run(pool, run_id)
        assert run is not None and run.status == "suspended"  # NON-terminal: harvest runs
        started = await _call_starteds(pool, run_id)
        assert len(started) == 2  # both opened; call 1 resolved, call 2 inflight
        assert sum(1 for c in execed if "echo one" in c) == 1  # call 1 ran exactly once

        # Drop call 1's signal so its ONLY remaining record is the journaled call_result
        # — the realistic state once a signal is GC'd. The ``call_key not in memo`` filter
        # is now the only thing keeping the completed call out of the harvest's inflight set.
        call_1_key = started[0].call_key
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM wf_run_signals WHERE run_id = $1 AND call_key = $2",
                run_id,
                call_1_key,
            )

        # Re-drive while non-terminal: call 2 inflight (has_inflight True → no relaunch),
        # call 1 already in the memo (resolved → never re-examined). Neither re-execs.
        await run_workflow_step(run_id)
        assert sum(1 for c in execed if "echo one" in c) == 1  # COMPLETED call 1 NOT re-execed
        assert len(await _call_starteds(pool, run_id)) == 2  # no double-open of either call

        second_block.set()  # let call 2 finish so no blocked task lingers
        await _drain_sandbox_tasks()
    await run_workflow_step(run_id)  # harvest call 2 → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"


# (d) ──────────────────────────────────────────────────────────────────────────
async def test_bash_provision_failure_one_error_value(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """A provision failure is a RECOVERABLE value, not a terminal error: the task
    writes one ``tool_result`` carrying a flat ``{"error": …}``; the harvest folds it
    as a call_result VALUE (``is_error`` False — bash is a tool, so ``tool()`` never
    raises); the run CONTINUES and COMPLETES with that error value."""
    pool, backend = wf_sandbox_runtime

    async def _boom(_spec: Any) -> Any:
        raise SandboxBackendError("create boom")

    run_id = await _make_tool_run(
        pool, _BASH_SCRIPT, tools=[ToolSpec(type="bash")], name="wt-bash-d"
    )
    with mock.patch.object(backend, "create", new=_boom):
        await run_workflow_step(run_id)  # park at the bash frontier
        await _drain_sandbox_tasks()  # task: provision fails → {"error": …}
        await run_workflow_step(run_id)  # harvest the error value → complete
    run = await _get_run(pool, run_id)
    # RECOVERABLE: the run COMPLETES (not errored) carrying the error value.
    assert run is not None and run.status == "completed"
    assert isinstance(run.output, dict) and "error" in run.output
    assert "provisioning failed" in run.output["error"]
    assert "create boom" in run.output["error"]
    # Exactly one call_result; folded as a value (is_error False), not a thrown error.
    async with pool.acquire() as conn:
        rows = await wf_queries.list_run_events(conn, run_id)
    call_results = [e for e in rows if e.type == "call_result"]
    assert len(call_results) == 1
    assert call_results[0].payload["is_error"] is False
    assert "error" in call_results[0].payload["result"]
    assert _backend_exec_count(backend) == 0  # never reached exec


# (e) ──────────────────────────────────────────────────────────────────────────
async def test_bash_lazy_no_provision_until_called(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """A run that never calls ``tool('bash')`` provisions NO container — the run
    sandbox is lazy, created only on first bash exec."""
    pool, backend = wf_sandbox_runtime
    run_id = await _make_tool_run(
        pool,
        "async def main(input):\n    return 'no bash here'\n",
        tools=[ToolSpec(type="bash")],
        name="wt-bash-e",
    )
    await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"
    assert _backend_create_count(backend) == 0  # never provisioned


# (f) ──────────────────────────────────────────────────────────────────────────
async def test_bash_dispatches_while_agent_frontier_deferred(
    monkeypatch: pytest.MonkeyPatch,
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
    wf_agent_id: str,
) -> None:
    """UNTHROTTLED proof: with the per-run wave cap at 1 and a
    ``parallel([agent, agent, tool('bash', …)])`` frontier, one agent is admitted,
    one is ``frontier_deferred``, and the bash tool gets its ``call_started`` + launch
    in the SAME wake — bash is NOT gated by the agent wave slot (gate/tool frontiers
    are unthrottled)."""
    _wave_cap(monkeypatch, workflow_max_inflight_children_per_run=1)
    pool, backend = wf_sandbox_runtime
    script = (
        "async def main(input):\n"
        "    return await parallel([\n"
        f"        lambda: agent('a', agent_id={wf_agent_id!r}),\n"
        f"        lambda: agent('b', agent_id={wf_agent_id!r}),\n"
        "        lambda: tool('bash', {'command': 'echo hi'}),\n"
        "    ])\n"
    )
    run_id = await _make_tool_run(pool, script, tools=[ToolSpec(type="bash")], name="wt-bash-f")
    await run_workflow_step(run_id)
    cs = await _call_starteds(pool, run_id)
    caps = sorted(e.payload["capability"] for e in cs)
    assert caps == ["agent", "tool"]  # one agent + the bash tool both opened
    assert len(await _deferred_keys(pool, run_id)) == 1  # the over-cap agent
    # Launch proof via the DURABLE consequence, not the transient registry: the
    # fire-and-forget task pops itself from _INFLIGHT on completion, so peeking at
    # the registry races a fast worker (flaked on CI 2026-06-12, run 27392429151's
    # merge wave). Drain, then assert the backend saw exactly one exec this wake.
    await _drain_sandbox_tasks()
    assert _backend_exec_count(backend) == 1  # the bash task launched this wake


# (g) ──────────────────────────────────────────────────────────────────────────
async def test_bash_idempotency_token_stable_across_redrive(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """Idempotency-token stability + distinctness (#988 test strategy (g)).

    The command execed for a bash call is prefixed with a shlex-quoted
    ``export AIOS_RUN_ID=<run.id> AIOS_IDEMPOTENCY_KEY=sha256(run_id‖call_key)`` line
    ahead of the verbatim author command. A crash re-drive re-launches with the SAME
    run.id + call_key, so the second exec carries a byte-identical token; two distinct
    bash calls carry distinct call_keys, hence distinct tokens; the journaled
    ``call_started.input.command`` is the verbatim UN-prefixed author command."""
    pool, backend = wf_sandbox_runtime
    script = (
        "async def main(input):\n"
        "    a = await tool('bash', {'command': 'echo one'})\n"
        "    b = await tool('bash', {'command': 'echo two'})\n"
        "    return [a, b]\n"
    )
    block = asyncio.Event()  # never set — the first task blocks until cancelled
    entered = asyncio.Event()
    first_command: str | None = None  # the command the crashed first exec received

    async def _blocked(_handle: Any, command: str, **_k: Any) -> CommandResult:
        nonlocal first_command
        first_command = command
        entered.set()
        await block.wait()
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)

    run_id = await _make_tool_run(pool, script, tools=[ToolSpec(type="bash")], name="wt-bash-g")
    with mock.patch.object(backend, "exec", new=_blocked):
        await run_workflow_step(run_id)  # park at the first bash frontier; launch task
        await asyncio.wait_for(entered.wait(), timeout=5)  # task blocks IN exec
        started = await _call_starteds(pool, run_id)
        first_call_key = started[0].call_key
        # The journaled command is the verbatim UN-prefixed author command.
        assert started[0].payload["input"] == {"command": "echo one"}
        idem = hashlib.sha256(f"{run_id}\0{first_call_key}".encode()).hexdigest()
        expected_preamble = (
            f"export AIOS_RUN_ID={shlex.quote(run_id)} AIOS_IDEMPOTENCY_KEY={shlex.quote(idem)}\n"
        )
        # The execed command is the shlex-quoted preamble + the author command.
        assert first_command == expected_preamble + "echo one"
        # Simulate a hard worker crash: cancel the in-flight task + drop the registry map.
        for task in list(run_tools._INFLIGHT.values()):
            task.cancel()
        await asyncio.gather(*list(run_tools._INFLIGHT.values()), return_exceptions=True)
        run_tools._INFLIGHT.clear()

    # Re-drive: no signal + no live task → re-dispatch with the REAL FakeBackend.exec.
    backend.next_result = CommandResult(
        exit_code=0, stdout="", stderr="", timed_out=False, truncated=False
    )
    await run_workflow_step(run_id)
    assert len(await _call_starteds(pool, run_id)) == 1  # no double-open
    redriven = _execed_commands(backend)
    assert len(redriven) == 1
    # Token STABILITY: the re-driven exec carries the IDENTICAL idempotency token.
    assert redriven[0] == expected_preamble + "echo one"
    assert first_command is not None and redriven[0] == first_command
    await _drain_sandbox_tasks()

    # Harvest the first call; the script advances to the SECOND bash call.
    await run_workflow_step(run_id)
    await _drain_sandbox_tasks()
    started = await _call_starteds(pool, run_id)
    assert len(started) == 2  # the second bash call opened
    second_call_key = started[1].call_key
    # DISTINCTNESS: distinct bash calls ⇒ distinct call_keys ⇒ distinct tokens.
    assert second_call_key != first_call_key
    second_idem = hashlib.sha256(f"{run_id}\0{second_call_key}".encode()).hexdigest()
    second_command = _execed_commands(backend)[1]
    assert f"AIOS_IDEMPOTENCY_KEY={shlex.quote(second_idem)}" in second_command
    assert second_idem != idem
    # Run drives to completion.
    await run_workflow_step(run_id)
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"


# (h) ──────────────────────────────────────────────────────────────────────────
async def test_bash_runtime_lattice_drops_undeclared(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """**MUST-HAVE** — the #794 authority lattice: a run whose frozen ``tools`` does
    NOT declare bash resolves ``tool('bash', …)`` to the recoverable not-declared
    value, never provisioning a container; the run CONTINUES and COMPLETES."""
    pool, backend = wf_sandbox_runtime
    # The workflow declares NO tools → the run snapshots an empty tools list, so bash
    # is run-callable (in RUN_TOOLS) but NOT declared → the not-declared gate value.
    run_id = await _make_tool_run(pool, _BASH_SCRIPT, tools=[], name="wt-bash-h")
    await run_workflow_step(run_id)  # park at the bash frontier (routes to the sandbox executor)
    await _drain_sandbox_tasks()  # task: gate_run_tool → not-declared {"error": …}
    await run_workflow_step(run_id)  # harvest the error value → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"  # recoverable, NOT errored
    assert run.output == {"error": "tool 'bash' is not in the workflow's declared tools"}
    assert _backend_create_count(backend) == 0  # never provisioned
    assert _backend_exec_count(backend) == 0


# (i) ──────────────────────────────────────────────────────────────────────────
async def test_bash_class_other_tool_not_callable(
    wf_sandbox_runtime: tuple[asyncpg.Pool[Any], FakeBackend],
) -> None:
    """Another sandbox-class builtin (``read``) routes to the sandbox executor by
    execution class but is NOT in the run-callable set, so it resolves to the
    not-callable value; the run CONTINUES and COMPLETES, never provisioning."""
    pool, backend = wf_sandbox_runtime
    script = (
        "async def main(input):\n    return await tool('read', {'file_path': '/workspace/x'})\n"
    )
    # Declare read so the gate's FIRST clause (not-in-RUN_TOOLS) is what rejects it —
    # not the declared-tools clause — pinning the not-callable string.
    run_id = await _make_tool_run(pool, script, tools=[ToolSpec(type="read")], name="wt-bash-i")
    await run_workflow_step(run_id)  # park at the read frontier (routes to the sandbox executor)
    await _drain_sandbox_tasks()  # task: gate_run_tool → not-callable {"error": …}
    await run_workflow_step(run_id)  # harvest the error value → complete
    run = await _get_run(pool, run_id)
    assert run is not None and run.status == "completed"  # recoverable, NOT errored
    assert run.output == {"error": "tool 'read' is not callable from a workflow run"}
    assert _backend_create_count(backend) == 0  # never provisioned
    assert _backend_exec_count(backend) == 0


# ─── update_workflow — in-flight runs are immune (snapshot pinning) ───────────


async def test_update_workflow_does_not_disturb_inflight_runs(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """The load-bearing update proof: a run executes the script + surface it snapshotted
    at launch, even when the workflow is updated mid-flight; a NEW run gets the update."""
    pool = wf_runtime
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="w-immune",
            script="async def main(input):\n    return 'OLD'\n",
            tools=[ToolSpec(type="web_search")],
        )
    old_run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
    )

    # Update the workflow mid-flight: new script, surface dropped.
    updated = await wf_service.update_workflow(
        pool,
        wf.id,
        account_id="acc_wf",
        expected_version=1,
        script="async def main(input):\n    return 'NEW'\n",
        tools=[],
    )
    assert updated.version == 2

    # The in-flight run still carries (and executes) its launch snapshot.
    async with pool.acquire() as conn:
        old_run_row = await wf_queries.get_wf_run(conn, old_run.id, account_id="acc_wf")
    assert "OLD" in old_run_row.script
    assert [t.type for t in old_run_row.tools] == ["web_search"]
    await run_workflow_step(old_run.id)
    async with pool.acquire() as conn:
        finished = await wf_queries.get_wf_run(conn, old_run.id, account_id="acc_wf")
    assert finished.status == "completed" and finished.output == "OLD"

    # A run launched after the update gets the new definition.
    new_run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
    )
    await run_workflow_step(new_run.id)
    async with pool.acquire() as conn:
        fresh = await wf_queries.get_wf_run(conn, new_run.id, account_id="acc_wf")
    assert fresh.status == "completed" and fresh.output == "NEW"
    assert fresh.tools == []


async def test_budget_primitive_round_trip_and_no_ceiling(wf_runtime: asyncpg.Pool[Any]) -> None:
    pool = wf_runtime
    script = "async def main(input):\n    return [await budget(), await budget()]\n"
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="budgeted", script=script
        )
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf", budget_usd=1.25
    )
    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        r1 = await wf_queries.get_run_for_step(conn, run.id)
        events1 = await wf_queries.list_run_events(conn, run.id)
    assert r1 is not None and r1.status == "running"
    budget_results = [e for e in events1 if e.type == "call_result"]
    assert len(budget_results) == 1
    assert budget_results[0].payload == {
        "result": {"total_usd": 1.25, "spent_usd": 0.0, "remaining_usd": 1.25},
        "is_error": False,
    }
    assert budget_results[0].call_key not in {
        e.call_key for e in events1 if e.type == "call_started"
    }

    await run_workflow_step(run.id)
    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        r2 = await wf_queries.get_run_for_step(conn, run.id)
        events2 = await wf_queries.list_run_events(conn, run.id)
    assert r2 is not None and r2.status == "completed"
    assert r2.output == [
        {"total_usd": 1.25, "spent_usd": 0.0, "remaining_usd": 1.25},
        {"total_usd": 1.25, "spent_usd": 0.0, "remaining_usd": 1.25},
    ]
    assert len([e for e in events2 if e.type == "call_result"]) == 2

    no_budget = await _make_run(pool, "async def main(input):\n    return await budget()\n")
    await run_workflow_step(no_budget)
    await run_workflow_step(no_budget)
    async with pool.acquire() as conn:
        r3 = await wf_queries.get_run_for_step(conn, no_budget)
    assert r3 is not None and r3.output is None


async def test_over_budget_agent_refusal_is_catchable(
    wf_runtime: asyncpg.Pool[Any], wf_agent_id: str
) -> None:
    pool = wf_runtime
    script = (
        "async def main(input):\n"
        "    try:\n"
        f"        await agent('x', agent_id={wf_agent_id!r})\n"
        "    except AgentError as e:\n"
        "        return {'kind': e.kind, 'msg': str(e)}\n"
    )
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_wf", name="over-budget", script=script
        )
    run = await service.create_run(
        pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf", budget_usd=1.0
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, system, account_id) VALUES ('agent_cost_seed', 'cost-seed', 'm', 's', 'acc_wf')"
        )
        # Named workflow children must carry a pinned agent_version (0095's
        # sessions_agent_version_pair_ck); seed the matching agent_versions row.
        await conn.execute(
            "INSERT INTO agent_versions (agent_id, version, model, system, account_id) "
            "VALUES ('agent_cost_seed', 1, 'm', 's', 'acc_wf')"
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, agent_version, title, metadata, workspace_volume_path, account_id, parent_run_id, cost_microusd) "
            "VALUES ('ses_cost_seed', 'agent_cost_seed', 'env_wf', 1, NULL, '{}'::jsonb, '/tmp/cost', 'acc_wf', $1, 1000000)",
            run.id,
        )
    await run_workflow_step(run.id)
    await run_workflow_step(run.id)
    async with pool.acquire() as conn:
        r = await wf_queries.get_run_for_step(conn, run.id)
        events = await wf_queries.list_run_events(conn, run.id)
        child_count = await conn.fetchval(
            "SELECT count(*) FROM sessions WHERE parent_run_id = $1 AND id != 'ses_cost_seed'",
            run.id,
        )
    assert r is not None and r.status == "completed"
    assert r.output["kind"] == "budget_exceeded"
    assert "run budget exhausted" in r.output["msg"]
    cr = next(e for e in events if e.type == "call_result" and e.payload.get("is_error"))
    assert cr.payload["error"]["kind"] == "budget_exceeded"
    assert child_count == 0


async def _seed_child_session(
    pool: asyncpg.Pool[Any],
    *,
    sid: str,
    run_id: str,
    account_id: str = "acc_wf",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cost_microusd: int = 0,
) -> None:
    """Insert a run-child session carrying known usage (#1324 read-path seed)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, system, account_id) "
            "VALUES ($1, $1, 'm', 's', $2) ON CONFLICT (id) DO NOTHING",
            f"agent_{sid}",
            account_id,
        )
        await conn.execute(
            "INSERT INTO agent_versions (agent_id, version, model, system, account_id) "
            "VALUES ($1, 1, 'm', 's', $2) ON CONFLICT DO NOTHING",
            f"agent_{sid}",
            account_id,
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, agent_version, title, metadata, "
            "workspace_volume_path, account_id, parent_run_id, input_tokens, output_tokens, "
            "cache_read_input_tokens, cache_creation_input_tokens, cost_microusd) "
            "VALUES ($1, $2, 'env_wf', 1, NULL, '{}'::jsonb, $3, $4, $5, $6, $7, $8, $9, $10)",
            sid,
            f"agent_{sid}",
            f"/tmp/{sid}",
            account_id,
            run_id,
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            cost_microusd,
        )


async def test_get_run_surfaces_summed_child_usage_on_read_path(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """get_run sums its child sessions' cost/tokens onto WfRun.usage (#1324).

    The keystone of the machine-observer substrate: a run's realized spend is
    legible from the public read path, summed from the SAME run_children_usage
    source budget() consumes — not buried in a builtin.
    """
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    await _seed_child_session(
        pool,
        sid="ses_a",
        run_id=run_id,
        input_tokens=10,
        output_tokens=20,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=4,
        cost_microusd=123456,
    )
    await _seed_child_session(
        pool,
        sid="ses_b",
        run_id=run_id,
        input_tokens=1,
        output_tokens=2,
        cache_read_input_tokens=5,
        cache_creation_input_tokens=6,
        cost_microusd=654321,
    )

    run = await wf_service.get_run(pool, run_id, account_id="acc_wf")
    assert run.usage is not None
    assert run.usage.input_tokens == 11
    assert run.usage.output_tokens == 22
    assert run.usage.cache_read_input_tokens == 8
    assert run.usage.cache_creation_input_tokens == 10
    assert run.usage.cost_microusd == 777777
    # The run is still pending (no step driven): wall_clock is cannot-determine,
    # iteration_count has no substrate — both EXPLICIT null, never silently 0/omitted.
    assert run.usage.wall_clock_ms is None
    assert run.usage.iteration_count is None
    # ``budget_usd`` (ceiling) and ``usage.cost_microusd`` (spend) are both legible now.
    assert "usage" in run.model_dump()


async def test_get_run_usage_zero_is_observed_not_null(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A childless run sums to a REAL zero (distinct from null cannot-determine)."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    run = await wf_service.get_run(pool, run_id, account_id="acc_wf")
    assert run.usage is not None
    assert run.usage.cost_microusd == 0
    assert run.usage.input_tokens == 0
    assert run.usage.output_tokens == 0


async def test_terminal_run_surfaces_wall_clock_ms(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """A TERMINAL run surfaces a non-null wall_clock_ms (updated_at - created_at)."""
    pool = wf_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_runs SET status = 'completed', "
            "updated_at = created_at + interval '1500 milliseconds' WHERE id = $1",
            run_id,
        )
    run = await wf_service.get_run(pool, run_id, account_id="acc_wf")
    assert run.usage is not None
    assert run.usage.wall_clock_ms == 1500
    assert run.usage.iteration_count is None  # still no substrate


async def test_list_runs_enriches_each_run_with_usage(
    wf_runtime: asyncpg.Pool[Any],
) -> None:
    """list_runs carries per-run usage too — batched, one aggregate for the page (#1324)."""
    pool = wf_runtime
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id="acc_wf",
            name="w_list",
            script="async def main(input):\n    return 1\n",
        )
    run_a = (
        await service.create_run(
            pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
        )
    ).id
    run_b = (
        await service.create_run(
            pool, account_id="acc_wf", workflow_id=wf.id, environment_id="env_wf"
        )
    ).id
    await _seed_child_session(pool, sid="ses_la", run_id=run_a, cost_microusd=500, input_tokens=7)
    # run_b has no children → real zero.

    runs = await wf_service.list_runs(pool, account_id="acc_wf")
    by_id = {r.id: r for r in runs}
    usage_a = by_id[run_a].usage
    assert usage_a is not None
    assert usage_a.cost_microusd == 500
    assert usage_a.input_tokens == 7
    usage_b = by_id[run_b].usage
    assert usage_b is not None
    assert usage_b.cost_microusd == 0
