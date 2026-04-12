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
