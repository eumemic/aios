"""E2E tests for the event-driven step model.

Each test exercises the real system (Postgres, step function, async tool
dispatch, event log) with only the LLM responses scripted. Tests in the
fast tier use custom inline tools and don't need Docker. The one test in
the full tier uses real containers.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conftest import needs_docker
from tests.e2e.harness import (
    Harness,
    assistant,
    bash,
    cancel,
    first_tool_result,
    last_assistant_content,
    tool_call,
    tool_results,
)

# ─── fast tier (no Docker) ───────────────────────────────────────────────────


@needs_docker  # testcontainer Postgres needs Docker
class TestBasicFlows:
    async def test_basic_chat(self, harness: Harness) -> None:
        """No tools. Model responds with plain text."""
        harness.script_model([assistant("Hello, world!")])
        session = await harness.start("Say hello")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        assert last_assistant_content(events) == "Hello, world!"

        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason == {"type": "end_turn"}

    async def test_single_tool_round_trip(self, harness: Harness) -> None:
        """Custom tool: model calls it, gets result, responds."""

        async def echo_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"output": arguments.get("text", "")}

        harness.register_tool("echo", echo_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("echo", {"text": "ping"})]),
                assistant("Echo said: ping"),
            ]
        )
        session = await harness.start("echo ping", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        tr = first_tool_result(events)
        assert '"ping"' in tr["content"]
        assert last_assistant_content(events) == "Echo said: ping"

    async def test_multi_step_chain(self, harness: Harness) -> None:
        """Model calls tool A, sees result, calls tool B, responds."""

        async def step_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"step": arguments.get("n", 0)}

        harness.register_tool("step_tool", step_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("step_tool", {"n": 1})]),
                assistant(tool_calls=[tool_call("step_tool", {"n": 2})]),
                assistant("Both steps done."),
            ]
        )
        session = await harness.start("do two steps", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        trs = tool_results(events)
        assert len(trs) == 2
        assert last_assistant_content(events) == "Both steps done."

    async def test_batch_completion(self, harness: Harness) -> None:
        """Model requests 3 tools at once. All must complete before next step."""

        async def instant_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"value": arguments.get("n", 0)}

        harness.register_tool("instant", instant_handler)
        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("instant", {"n": 1}),
                        tool_call("instant", {"n": 2}),
                        tool_call("instant", {"n": 3}),
                    ]
                ),
                assistant("All three done."),
            ]
        )
        session = await harness.start("do three things", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        trs = tool_results(events)
        assert len(trs) == 3
        assert last_assistant_content(events) == "All three done."

    async def test_early_out_no_new_events(self, harness: Harness) -> None:
        """After model responds, another run_step is a no-op."""
        harness.script_model([assistant("Done.")])
        session = await harness.start("hi")
        await harness.run_until_idle(session.id)

        # Script is exhausted (1 response consumed). If run_step tried to
        # call the model again, it would raise because no more responses.
        # The early-out prevents this.
        await harness.run_step(session.id)  # should be a no-op

        events = await harness.events(session.id)
        assistants = [e for e in events if e.data.get("role") == "assistant"]
        assert len(assistants) == 1  # still just one


@needs_docker
class TestMidTurnInjection:
    async def test_user_message_while_tool_runs(self, harness: Harness) -> None:
        """Inject a user message while a tool is in flight."""
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"output": "slow done"}

        harness.register_tool("slow", slow_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow", {})]),
                # Step 2: model sees user injection + tool result
                assistant("Tool finished and user said something new."),
            ]
        )
        session = await harness.start("do slow thing", tools=[])

        # Step 1: model calls slow tool
        await harness.run_step(session.id)

        # Tool is now in flight
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # Inject user message while tool is running
        await harness.inject_message(session.id, "what's happening?")

        # Let the tool complete
        tool_proceed.set()
        await harness.wait_for_tools(session.id)

        # Step 2: should_call_model sees both tool result AND user injection
        await harness.run_step(session.id)

        events = await harness.events(session.id)
        user_msgs = [e for e in events if e.data.get("role") == "user"]
        assert len(user_msgs) == 2  # original + injection
        assert last_assistant_content(events) == "Tool finished and user said something new."


@needs_docker
class TestErrorHandling:
    async def test_tool_error_propagates(self, harness: Harness) -> None:
        """Tool raises an exception → error result in event log."""

        async def failing_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("something broke")

        harness.register_tool("fails", failing_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("fails", {})]),
                assistant("I see the tool failed."),
            ]
        )
        session = await harness.start("do the failing thing", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        tr = first_tool_result(events)
        assert tr.get("is_error") is True
        assert "something broke" in tr["content"]
        assert last_assistant_content(events) == "I see the tool failed."


@needs_docker
class TestCancelFlow:
    async def test_cancel_in_flight_tool(self, harness: Harness) -> None:
        """Model cancels a running tool via the cancel tool."""
        tool_started = asyncio.Event()

        async def blocking_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await asyncio.sleep(3600)  # will be cancelled
            return {"output": "should not reach"}  # pragma: no cover

        harness.register_tool("blocking", blocking_handler)

        tc_slow = tool_call("blocking", {}, call_id="call_slow")
        tc_cancel = cancel("call_slow", call_id="call_cancel")

        harness.script_model(
            [
                # Step 1: model calls blocking tool
                assistant(tool_calls=[tc_slow]),
                # Step 2 (after user injection): model calls cancel
                assistant(tool_calls=[tc_cancel]),
                # Step 3: model sees cancel result + blocking error, responds
                assistant("Cancelled and moving on."),
            ]
        )

        session = await harness.start("do blocking thing", tools=["cancel"])

        # Step 1: model calls blocking tool
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # User injects "cancel it" to trigger step 2
        await harness.inject_message(session.id, "cancel it")

        # Step 2: model calls cancel tool
        await harness.run_step(session.id)
        # Cancel tool executes synchronously (it's fast).
        # The blocking tool gets CancelledError, appends error result.
        await harness.wait_for_tools(session.id)

        # Step 3: model sees everything, responds
        await harness.run_step(session.id)

        events = await harness.events(session.id)
        trs = tool_results(events)
        # Should have: blocking tool error (cancelled) + cancel tool result
        cancelled_results = [t for t in trs if "cancelled" in t.get("content", "")]
        assert len(cancelled_results) >= 1
        assert last_assistant_content(events) == "Cancelled and moving on."


# ─── invariant tests ─────────────────────────────────────────────────────────


@needs_docker
class TestPendingResultSynthesis:
    async def test_context_contains_pending_for_inflight_tools(self, harness: Harness) -> None:
        """When model is called with an in-flight tool, the context should
        contain a synthetic 'pending' result for that tool."""
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"output": "done"}

        harness.register_tool("slow", slow_handler)
        tc = tool_call("slow", {}, call_id="call_slow_1")
        harness.script_model(
            [
                assistant(tool_calls=[tc]),
                # Step 2: called because of user injection, tool still pending
                assistant("Working on it, hang on."),
                # Step 3: tool completed
                assistant("All done."),
            ]
        )
        session = await harness.start("do slow thing")

        # Step 1: model calls slow tool
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # Inject user message while tool is running
        await harness.inject_message(session.id, "status?")

        # Step 2: model sees user injection + pending tool
        await harness.run_step(session.id)

        # Inspect the context that was sent to the model in step 2
        assert len(harness.model_calls) == 2
        step2_messages = harness.model_calls[1]["messages"]
        # Find the tool result for call_slow_1 in the context
        tool_msg = next(
            m
            for m in step2_messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_slow_1"
        )
        assert "pending" in tool_msg["content"]

        # Let tool complete, finish up
        tool_proceed.set()
        await harness.wait_for_tools(session.id)
        await harness.run_step(session.id)

    async def test_completed_tool_shows_real_result_not_pending(self, harness: Harness) -> None:
        """After a tool completes, the context should show its real result,
        not a pending placeholder."""

        async def fast_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"value": 42}

        harness.register_tool("fast", fast_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("fast", {}, call_id="call_f1")]),
                assistant("Got 42."),
            ]
        )
        session = await harness.start("do fast thing")
        await harness.run_until_idle(session.id)

        # Step 2 should have received the real result, not pending
        assert len(harness.model_calls) == 2
        step2_messages = harness.model_calls[1]["messages"]
        tool_msg = next(
            m
            for m in step2_messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_f1"
        )
        assert "pending" not in tool_msg["content"]
        assert "42" in tool_msg["content"]


@needs_docker
class TestContextOrdering:
    async def test_tool_result_reordered_before_user_injection(self, harness: Harness) -> None:
        """If a user message arrives before a tool result in the log (by seq),
        the context builder should reorder so tool results appear right after
        their requesting assistant message, before the user message."""
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"output": "result"}

        harness.register_tool("slow", slow_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow", {}, call_id="call_s1")]),
                assistant("Saw both."),
            ]
        )
        session = await harness.start("do it")

        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # User message lands BEFORE tool result in the log
        await harness.inject_message(session.id, "also do Y")

        # Tool completes AFTER user message
        tool_proceed.set()
        await harness.wait_for_tools(session.id)

        # Step 2: model should see: assistant+tool_calls, tool_result, user_msg
        await harness.run_step(session.id)

        step2_messages = harness.model_calls[1]["messages"]
        # Find the positions
        roles = [(m.get("role"), m.get("tool_call_id")) for m in step2_messages]
        # After the assistant with tool_calls, the tool result should come
        # BEFORE the user injection
        asst_idx = next(i for i, (r, _) in enumerate(roles) if r == "assistant" and i > 0)
        tool_idx = next(i for i, (r, tc) in enumerate(roles) if r == "tool" and tc == "call_s1")
        user_inject_idx = next(i for i, (r, _) in enumerate(roles) if r == "user" and i > 1)
        assert tool_idx == asst_idx + 1, (
            "tool result should immediately follow its assistant message"
        )
        assert user_inject_idx > tool_idx, "user injection should come after tool result"


@needs_docker
class TestBatchGating:
    async def test_partial_batch_does_not_trigger_model(self, harness: Harness) -> None:
        """If 2 of 3 tools in a batch complete, should_call_model returns False."""
        completed = asyncio.Event()
        gate = asyncio.Event()

        async def instant_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"n": arguments.get("n")}

        async def gated_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            completed.set()  # signal that we're waiting
            await gate.wait()
            return {"n": "gated"}

        harness.register_tool("instant_b", instant_handler)
        harness.register_tool("gated", gated_handler)

        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("instant_b", {"n": 1}, call_id="call_i1"),
                        tool_call("instant_b", {"n": 2}, call_id="call_i2"),
                        tool_call("gated", {}, call_id="call_g1"),
                    ]
                ),
                assistant("All three done."),
            ]
        )
        session = await harness.start("three tools")

        # Step 1: model calls 3 tools
        await harness.run_step(session.id)

        # Wait for gated tool to enter its handler (instant tools may have already finished)
        await asyncio.wait_for(completed.wait(), timeout=5.0)

        # At this point instant_b tools have completed but gated has not.
        # Try to run a step — should be a no-op (partial batch).
        initial_call_count = len(harness.model_calls)
        await harness.run_step(session.id)
        assert len(harness.model_calls) == initial_call_count, (
            "model should NOT be called with partial batch results"
        )

        # Release the gated tool
        gate.set()
        await harness.wait_for_tools(session.id)

        # Now all three are done — step should call model
        await harness.run_step(session.id)
        assert len(harness.model_calls) == initial_call_count + 1


@needs_docker
class TestSessionStatus:
    async def test_status_transitions(self, harness: Harness) -> None:
        """Session status should go idle → running → idle across a step."""
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")

        s = await harness.session(session.id)
        assert s.status == "idle"

        await harness.run_until_idle(session.id)

        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason == {"type": "end_turn"}

    async def test_stop_reason_end_turn(self, harness: Harness) -> None:
        """When model returns no tool_calls, stop_reason is end_turn."""
        harness.script_model([assistant("Done.")])
        session = await harness.start("do it")
        await harness.run_until_idle(session.id)
        s = await harness.session(session.id)
        assert s.stop_reason == {"type": "end_turn"}


@needs_docker
class TestCancelContract:
    async def test_cancelled_tool_result_has_is_error(self, harness: Harness) -> None:
        """A cancelled tool's result should have is_error=True and contain 'cancelled'."""
        tool_started = asyncio.Event()

        async def blocking_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await asyncio.sleep(3600)
            return {}  # pragma: no cover

        harness.register_tool("blocker", blocking_handler)
        tc_block = tool_call("blocker", {}, call_id="call_block")
        tc_cancel_it = cancel("call_block", call_id="call_do_cancel")

        harness.script_model(
            [
                assistant(tool_calls=[tc_block]),
                assistant(tool_calls=[tc_cancel_it]),
                assistant("Cancelled."),
            ]
        )
        session = await harness.start("block", tools=["cancel"])
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        await harness.inject_message(session.id, "cancel it")
        await harness.run_step(session.id)
        await harness.wait_for_tools(session.id)
        await harness.run_step(session.id)

        events = await harness.events(session.id)
        blocked_result = next(
            e.data
            for e in events
            if e.data.get("role") == "tool" and e.data.get("tool_call_id") == "call_block"
        )
        assert blocked_result.get("is_error") is True
        assert "cancelled" in blocked_result["content"].lower()


# ─── reacting_to / stale-pending detection ───────────────────────────────────


@needs_docker
class TestReactingTo:
    async def test_tool_completing_during_inference_triggers_followup(
        self, harness: Harness
    ) -> None:
        """When a tool result arrives DURING inference (after context was built
        but before the model responds), the model's response is based on a stale
        "pending" snapshot. A follow-up step must fire so the model sees the real
        result.

        Timeline: model calls slow tool → user injects → step 2 builds context
        with pending tool → tool completes during Qwen thinking → model responds
        "still working" → step 3 should fire and see the real result.
        """
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"output": "finally done"}

        harness.register_tool("slow", slow_handler)
        harness.script_model(
            [
                # Step 1: model calls slow tool
                assistant(tool_calls=[tool_call("slow", {}, call_id="call_s")]),
                # Step 2: model sees user injection + pending tool → text response
                assistant("Tool is still running, hang on."),
                # Step 3: model sees real tool result → final response
                assistant("Tool finished with: finally done"),
            ]
        )
        session = await harness.start("do slow thing")

        # Step 1: model calls slow tool
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # Inject user message while tool is running
        await harness.inject_message(session.id, "status?")

        # Step 2: model sees user injection + pending tool result
        await harness.run_step(session.id)

        # Now let the tool complete (simulates tool finishing during/after inference)
        tool_proceed.set()
        await harness.wait_for_tools(session.id)

        # Step 3: should_call_model should return True because the model's
        # last response (step 2) has reacting_to < the tool result's seq.
        # The tool result is "new" relative to what the model reacted to.
        await harness.run_step(session.id)

        events = await harness.events(session.id)
        # Should have 3 assistant messages
        assistants = [e for e in events if e.data.get("role") == "assistant"]
        assert len(assistants) == 3
        assert "finally done" in last_assistant_content(events)

    async def test_reacting_to_field_present_on_assistant_messages(self, harness: Harness) -> None:
        """Every assistant message should carry a reacting_to field."""
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        asst = next(e for e in events if e.data.get("role") == "assistant")
        assert "reacting_to" in asst.data
        # reacting_to should point to the user message's seq
        user_seq = next(e.seq for e in events if e.data.get("role") == "user")
        assert asst.data["reacting_to"] == user_seq

    async def test_no_followup_when_model_saw_real_result(self, harness: Harness) -> None:
        """If the tool result was available when the context was built (not
        pending), reacting_to includes it and no follow-up step fires."""

        async def fast_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"value": 42}

        harness.register_tool("fast", fast_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("fast", {}, call_id="call_f")]),
                assistant("Got 42."),
            ]
        )
        session = await harness.start("do it")
        await harness.run_until_idle(session.id)

        # Only 2 model calls — no unnecessary follow-up step
        assert len(harness.model_calls) == 2

        events = await harness.events(session.id)
        last_asst = next(e for e in reversed(events) if e.data.get("role") == "assistant")
        tool_result = next(e for e in events if e.data.get("role") == "tool")
        # reacting_to should be >= the tool result's seq (model saw the real result)
        assert last_asst.data["reacting_to"] >= tool_result.seq


# ─── streaming inference ────────────────────────────────────────────────────


@needs_docker
class TestStreamingInference:
    async def test_streaming_basic_chat(self, harness: Harness) -> None:
        """Streaming produces the same final result as non-streaming."""
        harness.script_model([assistant("Hello, world!")])
        session = await harness.start("Say hello")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        assert last_assistant_content(events) == "Hello, world!"

        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason == {"type": "end_turn"}

        # Verify stream=True was passed to litellm
        assert harness.model_calls[0].get("stream") is True

    async def test_streaming_tool_round_trip(self, harness: Harness) -> None:
        """Streaming works correctly with tool calls."""

        async def echo_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"output": arguments.get("text", "")}

        harness.register_tool("echo", echo_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("echo", {"text": "streamed"})]),
                assistant("Echo result: streamed"),
            ]
        )
        session = await harness.start("echo streamed", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        tr = first_tool_result(events)
        assert '"streamed"' in tr["content"]
        assert last_assistant_content(events) == "Echo result: streamed"

        # Both calls should be streaming
        assert all(c.get("stream") is True for c in harness.model_calls)

    async def test_streaming_deltas_via_pg_notify(self, harness: Harness) -> None:
        """Delta notifications are sent via pg_notify during streaming."""
        import json

        from aios.db.listen import listen_for_events

        harness.script_model([assistant("ABCDEFGH")])
        session = await harness.start("say abcdefgh")

        # Open a listener BEFORE running the step so we catch deltas
        from aios.config import get_settings

        settings = get_settings()
        async with listen_for_events(settings.db_url, session.id) as queue:
            await harness.run_until_idle(session.id)

            # Drain the queue — collect both deltas and event notifications
            deltas: list[str] = []
            event_ids: list[str] = []
            while not queue.empty():
                payload = queue.get_nowait()
                if payload.startswith("{"):
                    data = json.loads(payload)
                    deltas.append(data["delta"])
                else:
                    event_ids.append(payload)

        # Deltas should reconstruct the original content
        assert "".join(deltas) == "ABCDEFGH"
        # Event notifications should also have arrived (for persisted events)
        assert len(event_ids) > 0

    async def test_streaming_reacting_to_preserved(self, harness: Harness) -> None:
        """reacting_to field is set correctly on streamed responses."""
        harness.script_model([assistant("Streamed response")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        asst = next(e for e in events if e.data.get("role") == "assistant")
        assert "reacting_to" in asst.data
        user_seq = next(e.seq for e in events if e.data.get("role") == "user")
        assert asst.data["reacting_to"] == user_seq

    async def test_streaming_empty_content_chunks_skipped(self, harness: Harness) -> None:
        """Tool-only responses (no text content) don't produce deltas."""
        import json

        from aios.config import get_settings
        from aios.db.listen import listen_for_events

        async def noop_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}

        harness.register_tool("noop", noop_handler)
        harness.script_model(
            [
                # First response: only tool calls, no text content
                assistant(tool_calls=[tool_call("noop", {})]),
                assistant("Done."),
            ]
        )
        session = await harness.start("do noop", tools=[])

        settings = get_settings()
        async with listen_for_events(settings.db_url, session.id) as queue:
            await harness.run_until_idle(session.id)

            deltas: list[str] = []
            while not queue.empty():
                payload = queue.get_nowait()
                if payload.startswith("{"):
                    data = json.loads(payload)
                    if "delta" in data:
                        deltas.append(data["delta"])

        # Only the second response ("Done.") should produce deltas
        assert "".join(deltas) == "Done."


# ─── agent versioning + session mutability ──────────────────────────────────


@needs_docker
class TestAgentVersioning:
    async def test_create_agent_starts_at_version_1(self, harness: Harness) -> None:
        """Newly created agents are version 1."""
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-test-{id(self)}",
            model="fake/test",
            system="v1 system",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        assert agent.version == 1

    async def test_update_bumps_version(self, harness: Harness) -> None:
        """Updating an agent creates a new version."""
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-bump-{id(self)}",
            model="fake/test",
            system="original",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        updated = await agents_service.update_agent(
            harness._pool, agent.id, expected_version=1, system="updated"
        )
        assert updated.version == 2
        assert updated.system == "updated"

    async def test_update_noop_keeps_version(self, harness: Harness) -> None:
        """Updating with no changes doesn't bump the version."""
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-noop-{id(self)}",
            model="fake/test",
            system="same",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        result = await agents_service.update_agent(harness._pool, agent.id, expected_version=1)
        assert result.version == 1  # no bump

    async def test_update_wrong_version_raises(self, harness: Harness) -> None:
        """Optimistic concurrency: wrong version raises ConflictError."""
        from aios.errors import ConflictError
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-conflict-{id(self)}",
            model="fake/test",
            system="original",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        with pytest.raises(ConflictError, match="version mismatch"):
            await agents_service.update_agent(
                harness._pool, agent.id, expected_version=99, system="bad"
            )

    async def test_version_history(self, harness: Harness) -> None:
        """Version history tracks all updates."""
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-hist-{id(self)}",
            model="fake/test",
            system="v1",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        await agents_service.update_agent(harness._pool, agent.id, expected_version=1, system="v2")
        await agents_service.update_agent(harness._pool, agent.id, expected_version=2, system="v3")
        versions = await agents_service.list_agent_versions(harness._pool, agent.id)
        assert len(versions) == 3
        # Newest first
        assert versions[0].version == 3
        assert versions[0].system == "v3"
        assert versions[2].version == 1
        assert versions[2].system == "v1"

    async def test_get_specific_version(self, harness: Harness) -> None:
        """Can retrieve a specific historical version."""
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"ver-get-{id(self)}",
            model="fake/test",
            system="original",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        await agents_service.update_agent(
            harness._pool, agent.id, expected_version=1, system="changed"
        )
        v1 = await agents_service.get_agent_version(harness._pool, agent.id, 1)
        assert v1.system == "original"
        v2 = await agents_service.get_agent_version(harness._pool, agent.id, 2)
        assert v2.system == "changed"


@needs_docker
class TestSessionVersionBinding:
    async def test_session_defaults_to_latest(self, harness: Harness) -> None:
        """Sessions default to agent_version=None (latest)."""
        harness.script_model([assistant("Hi")])
        session = await harness.start("hello")
        assert session.agent_version is None

    async def test_session_uses_latest_after_agent_update(self, harness: Harness) -> None:
        """Unpinned session uses updated agent config on next step."""
        from aios.services import agents as agents_service

        # Create agent + session with system="original"
        harness.script_model(
            [
                assistant("Response 1"),
                assistant("Response 2"),
            ]
        )
        session = await harness.start("hello", system="original")
        await harness.run_until_idle(session.id)

        # Update the agent's system prompt
        agent = await agents_service.get_agent(harness._pool, session.agent_id)
        await agents_service.update_agent(
            harness._pool, agent.id, expected_version=agent.version, system="updated"
        )

        # Send another message — step should use the updated system prompt
        await harness.inject_message(session.id, "hello again")
        await harness.run_step(session.id)

        # The second model call should have the updated system prompt
        assert len(harness.model_calls) == 2
        step2_messages = harness.model_calls[1]["messages"]
        system_msg = next(m for m in step2_messages if m["role"] == "system")
        assert system_msg["content"] == "updated"

    async def test_pinned_session_ignores_agent_update(self, harness: Harness) -> None:
        """Pinned session keeps using the old version after agent update."""
        from aios.services import agents as agents_service, sessions as sessions_service

        harness.script_model(
            [
                assistant("Response 1"),
                assistant("Response 2"),
            ]
        )
        session = await harness.start("hello", system="original")
        await harness.run_until_idle(session.id)

        # Pin session to version 1
        await sessions_service.update_session(harness._pool, session.id, agent_version=1)

        # Update agent to version 2
        agent = await agents_service.get_agent(harness._pool, session.agent_id)
        await agents_service.update_agent(
            harness._pool, agent.id, expected_version=agent.version, system="v2 system"
        )

        # Send another message — pinned session should still use v1 system prompt
        await harness.inject_message(session.id, "hello again")
        await harness.run_step(session.id)

        step2_messages = harness.model_calls[1]["messages"]
        system_msg = next(m for m in step2_messages if m["role"] == "system")
        assert system_msg["content"] == "original"  # v1, not v2

    async def test_session_update_changes_agent(self, harness: Harness) -> None:
        """Sessions can be updated to point at a different agent."""
        from aios.services import agents as agents_service, sessions as sessions_service

        harness.script_model(
            [
                assistant("From agent 1"),
                assistant("From agent 2"),
            ]
        )
        session = await harness.start("hello", system="agent-one")
        await harness.run_until_idle(session.id)

        # Create a second agent
        from aios.ids import make_id
        from aios.models.agents import ToolSpec

        agent2 = await agents_service.create_agent(
            harness._pool,
            name=f"agent-two-{make_id('agent')[-8:]}",
            model="fake/test",
            system="agent-two",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )

        # Switch session to agent 2
        await sessions_service.update_session(harness._pool, session.id, agent_id=agent2.id)

        # Next step should use agent 2's system prompt
        await harness.inject_message(session.id, "hello from agent 2")
        await harness.run_step(session.id)

        step2_messages = harness.model_calls[1]["messages"]
        system_msg = next(m for m in step2_messages if m["role"] == "system")
        assert system_msg["content"] == "agent-two"


# ─── custom tools ───────────────────────────────────────────────────────────


@needs_docker
class TestCustomTools:
    async def test_custom_tool_idles_with_requires_action(self, harness: Harness) -> None:
        """When the model calls a custom tool, the session idles with requires_action."""
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"custom-tool-{id(self)}",
            model="fake/test",
            system="You are a test assistant.",
            tools=[
                ToolSpec(
                    type="custom",
                    name="get_weather",
                    description="Get the weather",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )

        # Script model to call the custom tool
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("get_weather", {"city": "SF"}, call_id="call_w1")]),
            ]
        )

        from aios.ids import make_id
        from aios.services import environments as env_svc, sessions as sess_svc

        if harness._env_id is None:
            env = await env_svc.create_environment(
                harness._pool, name=f"test-env-{make_id('env')[-8:]}"
            )
            harness._env_id = env.id

        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=harness._env_id,
            title="custom-tool-test",
            metadata={},
        )
        await sess_svc.append_user_message(harness._pool, session.id, "What's the weather in SF?")

        # Run one step — model calls custom tool
        await harness.run_step(session.id)

        # Session should be idle with requires_action
        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason is not None
        assert s.stop_reason["type"] == "requires_action"
        assert "call_w1" in s.stop_reason["event_ids"]

    async def test_custom_tool_result_resumes_session(self, harness: Harness) -> None:
        """Submitting a custom tool result via the API resumes the session."""
        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"custom-resume-{id(self)}",
            model="fake/test",
            system="You are a test assistant.",
            tools=[
                ToolSpec(
                    type="custom",
                    name="get_weather",
                    description="Get the weather",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )

        harness.script_model(
            [
                assistant(tool_calls=[tool_call("get_weather", {"city": "SF"}, call_id="call_w2")]),
                assistant("The weather in SF is sunny, 72°F."),
            ]
        )

        from aios.ids import make_id
        from aios.services import environments as env_svc, sessions as sess_svc

        if harness._env_id is None:
            env = await env_svc.create_environment(
                harness._pool, name=f"test-env-{make_id('env')[-8:]}"
            )
            harness._env_id = env.id

        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=harness._env_id,
            title="custom-resume-test",
            metadata={},
        )
        await sess_svc.append_user_message(harness._pool, session.id, "Weather in SF?")

        # Step 1: model calls custom tool → session idles
        await harness.run_step(session.id)

        # Simulate client sending tool result
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "call_w2", "content": "Sunny, 72°F"},
        )

        # Step 2: model sees the tool result and responds
        await harness.run_step(session.id)

        events = await harness.events(session.id)
        assert last_assistant_content(events) == "The weather in SF is sunny, 72°F."

        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason == {"type": "end_turn"}

    async def test_mixed_builtin_and_custom_tools(self, harness: Harness) -> None:
        """Model calls both a built-in tool and a custom tool in the same response."""

        async def echo_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"output": arguments.get("text", "")}

        harness.register_tool("echo", echo_handler)

        from aios.models.agents import ToolSpec
        from aios.services import agents as agents_service

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"mixed-tools-{id(self)}",
            model="fake/test",
            system="You are a test assistant.",
            tools=[
                ToolSpec(
                    type="custom",
                    name="lookup",
                    description="Look up data",
                    input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                ),
            ],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )

        harness.script_model(
            [
                # Model calls both echo (built-in) and lookup (custom) at once
                assistant(
                    tool_calls=[
                        tool_call("echo", {"text": "hi"}, call_id="call_echo"),
                        tool_call("lookup", {"key": "foo"}, call_id="call_lookup"),
                    ]
                ),
                assistant("Got both results."),
            ]
        )

        from aios.ids import make_id
        from aios.services import environments as env_svc, sessions as sess_svc

        if harness._env_id is None:
            env = await env_svc.create_environment(
                harness._pool, name=f"test-env-{make_id('env')[-8:]}"
            )
            harness._env_id = env.id

        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=harness._env_id,
            title="mixed-tools-test",
            metadata={},
        )
        await sess_svc.append_user_message(harness._pool, session.id, "Do both")

        # Step 1: model calls both tools
        await harness.run_step(session.id)
        await harness.wait_for_tools(session.id)

        # Session should be idle with requires_action for the custom tool
        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason is not None
        assert s.stop_reason["type"] == "requires_action"

        # The built-in tool (echo) should have already completed
        events = await harness.events(session.id)
        echo_result = next(
            (
                e.data
                for e in events
                if e.data.get("role") == "tool" and e.data.get("tool_call_id") == "call_echo"
            ),
            None,
        )
        assert echo_result is not None

        # Submit custom tool result
        await sess_svc.append_event(
            harness._pool,
            session.id,
            "message",
            {"role": "tool", "tool_call_id": "call_lookup", "content": '{"value": "bar"}'},
        )

        # Step 2: model sees both results
        await harness.run_step(session.id)
        assert last_assistant_content(await harness.events(session.id)) == "Got both results."


# ─── full tier (with Docker) ─────────────────────────────────────────────────


@needs_docker
@pytest.mark.e2e
class TestDockerIntegration:
    async def test_bash_real_container(self, docker_harness: Harness) -> None:
        """Real container, real docker exec."""
        harness = docker_harness
        harness.script_model(
            [
                assistant(tool_calls=[bash("echo hello from e2e")]),
                assistant("The output was hello."),
            ]
        )
        session = await harness.start("echo hello", tools=["bash"])
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        tr = first_tool_result(events)
        content = tr["content"]
        assert "hello from e2e" in content
