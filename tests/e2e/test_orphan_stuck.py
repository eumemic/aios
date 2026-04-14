"""Recovery from SIGKILL during tool execution.

Scenario:
    1. Model requests two parallel tool calls [A, B].
    2. Tool A completes and appends its result.
    3. Tool B is mid-execution when the worker is killed (SIGKILL).
       No CancelledError handler runs — no result event is appended.
    4. On restart, the sweep detects tool B as a ghost, appends a
       synthetic error result, and the session proceeds.

This test asserts the correct recovery behavior via the sweep.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, tool_call


@needs_docker
class TestOrphanRecovery:
    async def test_incomplete_batch_recovers_after_sigkill(self, harness: Harness) -> None:
        """After SIGKILL, ghost repair synthesizes an error result for
        the lost tool and the session can proceed to inference.
        """
        tool_a_started = asyncio.Event()
        tool_b_started = asyncio.Event()
        tool_b_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_b_started.set()
            await tool_b_proceed.wait()
            return {"result": "b_done"}

        harness.register_tool("tool_a", handler_a)
        harness.register_tool("tool_b", handler_b)

        call_a = tool_call("tool_a", {}, call_id="call_a")
        call_b = tool_call("tool_b", {}, call_id="call_b")
        harness.script_model(
            [
                assistant(tool_calls=[call_a, call_b]),
                assistant("Tool A succeeded, tool B was interrupted."),
            ]
        )

        session = await harness.start("run both tools", tools=[])

        # Step 1: model responds with two parallel tool calls.
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)
        await asyncio.wait_for(tool_b_started.wait(), timeout=5.0)

        # Wait for tool A's result to appear in the event log.
        for _ in range(50):
            events = await harness.events(session.id)
            tool_a_done = any(
                e.kind == "message"
                and e.data.get("role") == "tool"
                and e.data.get("tool_call_id") == "call_a"
                for e in events
            )
            if tool_a_done:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("tool A result never appeared in event log")

        # Verify precondition: tool B has no result in the log.
        events = await harness.events(session.id)
        tool_b_has_result = any(
            e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_b"
            for e in events
        )
        assert not tool_b_has_result, "tool B should NOT have a result yet"

        # Simulate SIGKILL: cancel tool B without letting cleanup append results.
        await harness.simulate_sigkill(session.id)

        # Ghost repair detects tool B as a ghost and appends a synthetic error.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_b")

        # Session should now need inference.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs, (
            "Session should need inference after ghost repair synthesized "
            "an error result for the lost tool."
        )

        # Model responds to the error.
        await harness.run_step(session.id)
        events = await harness.events(session.id)
        last_asst = next(
            e
            for e in reversed(events)
            if e.kind == "message"
            and e.data.get("role") == "assistant"
            and not e.data.get("tool_calls")
        )
        assert last_asst.data.get("content"), "model should have responded"
