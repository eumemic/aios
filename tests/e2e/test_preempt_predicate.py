"""Direct tests of the #253 floored wake predicate against real Postgres.

``find_sessions_needing_inference(session_id=..., reacted_floor=...)`` is the
preemption trigger: "would this session be wake-eligible immediately after the
in-flight step (whose context watermark is the floor) finished?". These tests
pin each arm's floored behavior by constructing event-log states manually
(the ``test_confirmed_always_ask_ghost`` idiom) and calling the predicate
directly — no step machinery involved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios.harness.sweep import find_sessions_needing_inference
from aios.models.agents import ToolSpec
from aios.models.events import EventKind
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"  # PR 3 scaffolding


async def _floored(harness: Harness, session_id: str, floor: int) -> set[str]:
    return await find_sessions_needing_inference(
        harness._pool,
        harness._inflight_tool_registry,
        session_id=session_id,
        reacted_floor=floor,
    )


async def _append(harness: Harness, session_id: str, kind: EventKind, data: dict[str, Any]) -> int:
    event = await sessions_service.append_event(
        harness._pool, session_id, kind, data, account_id=_ACCOUNT_ID
    )
    return event.seq


def _batch_turn(tool_call_ids: list[str], *, reacting_to: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": tcid, "type": "function", "function": {"name": "glob", "arguments": "{}"}}
            for tcid in tool_call_ids
        ],
        "reacting_to": reacting_to,
    }


@needs_docker
class TestStimulusArm:
    async def test_user_message_above_floor_admits_at_floor_does_not(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("hi")])
        session = await harness.start("hello")
        events = await harness.all_events(session.id)
        msg_seq = max(e.seq for e in events)

        assert await _floored(harness, session.id, msg_seq - 1) == {session.id}
        # Floor at the message: the in-flight step is already reacting to it.
        assert await _floored(harness, session.id, msg_seq) == set()


@needs_docker
class TestBatchFilterArm:
    async def test_partial_batch_held_completing_result_admits(self, harness: Harness) -> None:
        """Partial-batch tool result does NOT admit at the floor; the
        batch-completing result does (no restart-thrash on tool fan-outs)."""
        harness.script_model([assistant("unused")])
        session = await harness.start("run tools", tools=["glob"])
        msg_seq = 1
        await _append(
            harness, session.id, "message", _batch_turn(["c1", "c2"], reacting_to=msg_seq)
        )

        # The in-flight step's context saw the batch but no results: floor = msg_seq.
        await _append(
            harness,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        )
        assert await _floored(harness, session.id, msg_seq) == set()

        await _append(
            harness,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        )
        assert await _floored(harness, session.id, msg_seq) == {session.id}


@needs_docker
class TestConfirmedArm:
    async def _session_with_confirmed_call(self, harness: Harness) -> tuple[str, int]:
        harness.script_model([assistant("unused")])
        session = await harness.start(
            "find files", tool_specs=[ToolSpec(type="glob", permission="always_ask")]
        )
        await _append(harness, session.id, "message", _batch_turn(["call_g"], reacting_to=1))
        confirm_seq = await _append(
            harness,
            session.id,
            "lifecycle",
            {"event": "tool_confirmed", "tool_call_id": "call_g", "result": "allow"},
        )
        return session.id, confirm_seq

    async def test_confirm_above_floor_undispatched_admits(self, harness: Harness) -> None:
        sid, confirm_seq = await self._session_with_confirmed_call(harness)
        assert await _floored(harness, sid, confirm_seq - 1) == {sid}

    async def test_confirm_at_floor_not_admitted(self, harness: Harness) -> None:
        """A confirm the in-flight step already consumed (dispatched at its
        confirmed-dispatch check) must not re-admit. This also pins the
        skipped #1710 dispatch-narrowing pass: the open, undispatched
        always_ask call alone admits nothing at the floor."""
        sid, confirm_seq = await self._session_with_confirmed_call(harness)
        assert await _floored(harness, sid, confirm_seq) == set()

    async def test_confirm_with_execute_start_span_not_admitted(self, harness: Harness) -> None:
        """The durable dispatch marker (``tool_execute_start``) excludes a
        still-running confirmed tool — without it, a confirmed tool dispatched
        by a PRIOR step would re-admit on every evaluation (preempt thrash)."""
        sid, confirm_seq = await self._session_with_confirmed_call(harness)
        await _append(
            harness,
            sid,
            "span",
            {"event": "tool_execute_start", "tool_call_id": "call_g", "tool_name": "glob"},
        )
        assert await _floored(harness, sid, confirm_seq - 1) == set()

    async def test_confirm_with_inflight_task_not_admitted(self, harness: Harness) -> None:
        """The registry arm of the same exclusion: an in-flight task for the
        confirmed call means it needs no dispatch — no preempt."""
        sid, confirm_seq = await self._session_with_confirmed_call(harness)
        blocker = asyncio.Event()

        async def _running() -> None:
            await blocker.wait()

        task = asyncio.create_task(_running())
        harness._inflight_tool_registry.add(sid, "call_g", task)
        try:
            assert await _floored(harness, sid, confirm_seq - 1) == set()
        finally:
            blocker.set()
            await task
            harness._inflight_tool_registry.remove(sid, "call_g")


@needs_docker
class TestFlooredOnlyRestrictions:
    async def test_dispatched_ghost_admitted_committed_but_not_floored(
        self, harness: Harness
    ) -> None:
        """The F1 restriction, pinned as a committed-vs-floored difference: a
        dispatched-but-taskless open call (crashed-worker ghost shape) is
        admitted by the COMMITTED predicate's #1710 dispatch-narrowing pass
        (wedge-safety: composed sweep repairs it first) but must NOT preempt —
        it was already true when the step's entry guard admitted the step, so
        it cannot represent an arriving event."""
        harness.script_model([assistant("unused")])
        session = await harness.start("run tools", tools=["glob"])
        await _append(harness, session.id, "message", _batch_turn(["c_ghost"], reacting_to=1))
        span_seq = await _append(
            harness,
            session.id,
            "span",
            {"event": "tool_execute_start", "tool_call_id": "c_ghost", "tool_name": "glob"},
        )

        committed = await find_sessions_needing_inference(
            harness._pool, harness._inflight_tool_registry, session_id=session.id
        )
        assert committed == {session.id}
        assert await _floored(harness, session.id, span_seq) == set()


@needs_docker
class TestErroredArm:
    async def test_errored_parked_session_not_admitted(self, harness: Harness) -> None:
        """The errored subtraction is inherited unchanged: a parked session is
        not preempt-eligible even with a fresh stimulus above the floor."""
        harness.script_model([assistant("unused")])
        session = await harness.start("hello")
        floor = 1
        # Latch errored (the ``last_error_seq`` bump keys on this lifecycle
        # shape), then land a tool stimulus above the floor. A USER message
        # would lift the park (last_user_seq > last_error_seq) — the errored
        # arm only holds back non-user stimuli, matching wake semantics.
        await _append(
            harness,
            session.id,
            "lifecycle",
            {"event": "turn_errored", "status": "errored", "stop_reason": "error"},
        )
        await _append(
            harness,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "c_x", "content": "late result"},
        )
        assert await _floored(harness, session.id, floor) == set()
