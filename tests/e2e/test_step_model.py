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
        assert s.stop_reason == "end_turn"

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
        assert s.stop_reason == "end_turn"

    async def test_stop_reason_end_turn(self, harness: Harness) -> None:
        """When model returns no tool_calls, stop_reason is end_turn."""
        harness.script_model([assistant("Done.")])
        session = await harness.start("do it")
        await harness.run_until_idle(session.id)
        s = await harness.session(session.id)
        assert s.stop_reason == "end_turn"


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
    async def test_tool_completing_during_inference_triggers_followup(self, harness: Harness) -> None:
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
