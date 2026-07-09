"""E2E tests for #253 auto-preemption of the in-flight model phase.

Real Postgres, real step function, scripted model — the model call is
additionally gated on an ``asyncio.Event`` so a step can be held mid-model-
phase while events arrive. ``_PREEMPT_POLL_INTERVAL_S`` is patched down so the
watcher's poll gate ticks fast.

The re-wake assertions use ``harness.sessions_needing_inference`` (the
committed-watermark predicate) rather than a queued procrastinate job because
the e2e fixture noops ``defer_wake`` — the harness drives steps manually.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios.models.events import Event
from tests.conftest import needs_docker
from tests.e2e.conftest import wait_for_predicate
from tests.e2e.harness import Harness, assistant, tool_call

pytestmark = pytest.mark.docker


class _GatedModel:
    """Gate the fixture's scripted model behind an ``asyncio.Event``.

    Each model call signals ``entered`` and then blocks on ``release`` before
    popping the harness's next scripted response — so a cancelled (preempted)
    call never consumes a script entry. EVERY model call pushes an entered
    marker, including release-set (non-blocking) runs — a test that runs a
    step with the gate open must consume that marker too, or the next
    ``wait_entered`` returns stale and races the following step's context
    build (the injected message lands under the floor and correctly never
    preempts).
    """

    def __init__(self, harness: Harness) -> None:
        self._harness = harness
        self.release = asyncio.Event()
        self._entered: asyncio.Queue[None] = asyncio.Queue()

    async def acompletion(self, **kwargs: Any) -> Any:
        self._entered.put_nowait(None)
        await self.release.wait()
        return self._harness._pop_response(**kwargs)

    async def wait_entered(self) -> None:
        await asyncio.wait_for(self._entered.get(), timeout=10.0)

    async def run_released_step(self, session_id: str) -> None:
        """Run one step with the gate held open, consuming its entered marker."""
        self.release.set()
        await self._harness.run_step(session_id)
        await self.wait_entered()
        self.release.clear()


@pytest.fixture
def gated_model(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> _GatedModel:
    monkeypatch.setattr("aios.harness.loop._PREEMPT_POLL_INTERVAL_S", 0.01)
    gated = _GatedModel(harness)
    monkeypatch.setattr("aios.harness.completion.litellm.acompletion", gated.acompletion)
    return gated


def _spans(events: list[Event], name: str) -> list[Event]:
    return [e for e in events if e.kind == "span" and e.data.get("event") == name]


def _assistants(events: list[Event]) -> list[Event]:
    return [e for e in events if e.kind == "message" and e.data.get("role") == "assistant"]


def _user_seq(events: list[Event], content: str) -> int:
    return next(
        e.seq
        for e in events
        if e.kind == "message" and e.data.get("role") == "user" and e.data.get("content") == content
    )


@needs_docker
class TestPreemptOnUserMessage:
    async def test_mid_step_message_preempts_and_next_step_reads_it(
        self, harness: Harness, gated_model: _GatedModel
    ) -> None:
        """The issue's headline scenario: a user message lands while the model
        is mid-flight → the model phase is cancelled with no assistant append,
        the session stays wake-eligible, and the re-run's assistant reacts to
        the new message."""
        harness.script_model([assistant("answered both")])
        session = await harness.start("hello", preempt_policy="preempt")

        step = asyncio.create_task(harness.run_step(session.id))
        await gated_model.wait_entered()
        await harness.inject_message(session.id, "wait, actually...")
        await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        injected_seq = _user_seq(events, "wait, actually...")
        assert _assistants(events) == []

        # Span trace: the cancelled model_request pair closes with cancelled_by
        # and, per the calibration contract, carries no usage/token fields.
        (preempted,) = _spans(events, "step_preempted")
        assert preempted.data["cancelled_by"] == injected_seq
        end_spans = _spans(events, "model_request_end")
        (cancelled_end,) = [s for s in end_spans if "cancelled_by" in s.data]
        assert cancelled_end.data["cancelled_by"] == injected_seq
        assert cancelled_end.data["is_error"] is False
        for poisoned in ("local_tokens", "local_tokens_by_class", "model", "model_usage"):
            assert poisoned not in cancelled_end.data
        start_span = _spans(events, "model_request_start")[0]
        assert cancelled_end.data["model_request_start_id"] == start_span.id

        # Resume: still wake-eligible at the committed watermark; the re-run
        # reacts to the injected message.
        assert await harness.sessions_needing_inference(session.id) == {session.id}
        gated_model.release.set()
        await harness.run_step(session.id)
        events = await harness.all_events(session.id)
        (reply,) = _assistants(events)
        assert reply.data["reacting_to"] >= injected_seq

    async def test_wait_policy_message_mid_step_does_not_preempt(
        self, harness: Harness, gated_model: _GatedModel
    ) -> None:
        """Default-policy regression: same interleaving, ``preempt_policy``
        unset — the step runs to completion on stale context and the queued
        wake handles the message next step (today's behavior, unchanged)."""
        harness.script_model([assistant("stale reply")])
        session = await harness.start("hello")

        step = asyncio.create_task(harness.run_step(session.id))
        await gated_model.wait_entered()
        await harness.inject_message(session.id, "wait, actually...")
        gated_model.release.set()
        await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        injected_seq = _user_seq(events, "wait, actually...")
        (reply,) = _assistants(events)
        assert reply.data["content"] == "stale reply"
        assert reply.data["reacting_to"] < injected_seq
        assert _spans(events, "step_preempted") == []
        # The injected message is the next step's work.
        assert await harness.sessions_needing_inference(session.id) == {session.id}


@needs_docker
class TestTriggerCriterion:
    async def test_partial_batch_result_holds_completing_result_preempts(
        self, harness: Harness, gated_model: _GatedModel
    ) -> None:
        """The load-bearing trigger consequence: a tool result from a
        partially in-flight batch does NOT preempt; the result that completes
        the outstanding set DOES — the wake predicate's batch filter evaluated
        at the in-flight step's watermark."""
        a_proceed = asyncio.Event()
        b_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            await a_proceed.wait()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            await b_proceed.wait()
            return {"result": "b_done"}

        harness.register_tool("tool_a", handler_a)
        harness.register_tool("tool_b", handler_b)
        harness.script_model(
            [assistant(tool_calls=[tool_call("tool_a", {}), tool_call("tool_b", {})])]
        )
        session = await harness.start("run both tools", preempt_policy="preempt")

        # Step 1: dispatch the two gated tools. The scripted tool-call turn
        # must pass the gate too.
        await gated_model.run_released_step(session.id)

        # Step 2: held mid-model-phase (a user message makes it wake-eligible).
        await harness.inject_message(session.id, "status?")
        step = asyncio.create_task(harness.run_step(session.id))
        await gated_model.wait_entered()

        # Partial batch: tool_a's result lands, tool_b still in flight → held.
        a_proceed.set()
        await wait_for_predicate(
            lambda: _has_tool_result(harness, session.id, "a_done"), max_wait_s=10.0
        )
        await asyncio.sleep(0.3)  # ~30 watcher ticks at the patched interval
        assert not step.done()

        # Completing result → preempt.
        b_proceed.set()
        await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        (preempted,) = _spans(events, "step_preempted")
        b_result_seq = max(
            e.seq for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        assert preempted.data["cancelled_by"] == b_result_seq
        # Only step 1's tool-call turn exists; the preempted step appended none.
        assert len(_assistants(events)) == 1


async def _has_tool_result(harness: Harness, session_id: str, needle: str) -> bool:
    events = await harness.all_events(session_id)
    return any(
        e.kind == "message"
        and e.data.get("role") == "tool"
        and needle in str(e.data.get("content"))
        for e in events
    )


@needs_docker
class TestStarvationGuard:
    async def test_fourth_step_runs_unpreemptible_after_three_preempts(
        self, harness: Harness, gated_model: _GatedModel
    ) -> None:
        """The #253 decision comment's required deferral cap: after 3
        consecutive preemptions with no completed assistant turn, the next
        step ignores new arrivals and runs to completion."""
        harness.script_model([assistant("finally answered")])
        session = await harness.start("hello", preempt_policy="preempt")

        for i in range(3):
            step = asyncio.create_task(harness.run_step(session.id))
            await gated_model.wait_entered()
            await harness.inject_message(session.id, f"more input {i}")
            await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        assert len(_spans(events, "step_preempted")) == 3
        assert _assistants(events) == []

        # Fourth step: a new arrival must NOT preempt it.
        step = asyncio.create_task(harness.run_step(session.id))
        await gated_model.wait_entered()
        await harness.inject_message(session.id, "yet more input")
        await asyncio.sleep(
            0.3
        )  # ~30 watcher ticks — a wrongly-armed watcher fires well within this
        assert not step.done()
        gated_model.release.set()
        await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        assert len(_spans(events, "step_preempted")) == 3
        (reply,) = _assistants(events)
        assert reply.data["content"] == "finally answered"

    async def test_completed_turn_resets_the_cap(
        self, harness: Harness, gated_model: _GatedModel
    ) -> None:
        """A completed assistant turn resets the consecutive-preempt count:
        preemption arms again on the next turn."""
        harness.script_model([assistant("turn one"), assistant("turn two")])
        session = await harness.start("hello", preempt_policy="preempt")

        for i in range(3):
            step = asyncio.create_task(harness.run_step(session.id))
            await gated_model.wait_entered()
            await harness.inject_message(session.id, f"more input {i}")
            await asyncio.wait_for(step, timeout=10.0)

        # Cap reached — run the turn to completion.
        await gated_model.run_released_step(session.id)
        events = await harness.all_events(session.id)
        assert len(_assistants(events)) == 1

        # Next turn: preemption is armed again.
        await harness.inject_message(session.id, "new turn")
        step = asyncio.create_task(harness.run_step(session.id))
        await gated_model.wait_entered()
        await harness.inject_message(session.id, "changed my mind")
        await asyncio.wait_for(step, timeout=10.0)

        events = await harness.all_events(session.id)
        assert len(_spans(events, "step_preempted")) == 4
        assert len(_assistants(events)) == 1
