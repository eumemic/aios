"""E2E tests for the unified session sweep.

Tests assert the **correct behavior** of ghost repair and inference
detection. Some of these tests exercise scenarios that were previously
broken (SIGKILL stuck sessions) and should now pass with the sweep.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, tool_call

# ─── ghost recovery ──────────────────────────────────────────────────────────


@needs_docker
class TestGhostRecovery:
    async def test_all_tools_lost_after_sigkill(self, harness: Harness) -> None:
        """SIGKILL before any tool completes — all tool calls lost.

        After ghost repair: synthetic errors appear for both tools.
        After running inference: model sees the errors and responds.
        """
        tool_a_started = asyncio.Event()
        tool_b_started = asyncio.Event()
        tool_a_proceed = asyncio.Event()
        tool_b_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            await tool_a_proceed.wait()
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
                assistant("Both tools failed — I'll try a different approach."),
            ]
        )

        session = await harness.start("run both tools", tools=[])

        # Step 1: model calls both tools.
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)
        await asyncio.wait_for(tool_b_started.wait(), timeout=5.0)

        # Simulate SIGKILL: cancel tasks without appending results.
        await harness.simulate_sigkill(session.id)

        # Verify: no tool results in the log.
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 0

        # Ghost repair should detect and fix both.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 2
        repaired_ids = {tcid for _, tcid in repaired}
        assert repaired_ids == {"call_a", "call_b"}

        # Synthetic error results should now be in the log.
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 2
        for tr in tool_results:
            assert tr.data.get("is_error") is True
            assert "No result was received" in tr.data.get("content", "")

        # Session should now need inference.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model responds to the errors.
        await harness.run_step(session.id)
        events = await harness.events(session.id)
        last_asst = next(
            e
            for e in reversed(events)
            if e.kind == "message"
            and e.data.get("role") == "assistant"
            and not e.data.get("tool_calls")
        )
        assert "failed" in last_asst.data.get("content", "").lower()

    async def test_crash_before_tool_launch(self, harness: Harness) -> None:
        """Assistant message with tool_calls exists, but tools never dispatched.

        Simulates a crash between appending the assistant message and
        calling launch_tool_calls. Ghost repair should detect and fix it.
        """
        # Create a session and manually append an assistant message with
        # tool_calls, bypassing the step function entirely.
        session = await harness.start("do something", tools=[])

        async def dummy_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "done"}

        harness.register_tool("my_tool", dummy_handler)

        # Manually append the assistant message with tool_calls.
        call_x = tool_call("my_tool", {}, call_id="call_x")
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call_x],
                "reacting_to": 1,
            },
        )

        # No tools launched — simulates crash before dispatch.
        # Ghost repair should find call_x.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_x")

        # Session should need inference.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model can now see the error and respond.
        harness.script_model(
            [
                assistant("The tool failed, let me try again."),
            ]
        )
        await harness.run_step(session.id)

    async def test_ghost_in_earlier_batch(self, harness: Harness) -> None:
        """Multi-batch conversation. Tool lost from first batch.

        Model responded to partial results + user messages. The ghost
        from batch 1 is detected even though a later batch completed.
        """
        tool_a_started = asyncio.Event()
        tool_a_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            await tool_a_proceed.wait()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "b_done"}

        async def handler_c(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "c_done"}

        harness.register_tool("slow_tool", handler_a)
        harness.register_tool("fast_tool", handler_b)
        harness.register_tool("other_tool", handler_c)

        harness.script_model(
            [
                # Batch 1: slow_tool (will be lost) + fast_tool (completes).
                assistant(
                    tool_calls=[
                        tool_call("slow_tool", {}, call_id="call_slow"),
                        tool_call("fast_tool", {}, call_id="call_fast"),
                    ]
                ),
                # Model sees fast_tool result + user message, slow_tool pending.
                assistant("Fast tool done, still waiting on slow tool..."),
                # After ghost repair: model sees slow_tool error.
                assistant("Slow tool failed. Moving on."),
            ]
        )

        session = await harness.start("run tools", tools=[])

        # Step 1: model calls both tools.
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)

        # Wait for fast_tool to complete.
        for _ in range(50):
            events = await harness.events(session.id)
            if any(
                e.data.get("tool_call_id") == "call_fast"
                for e in events
                if e.kind == "message" and e.data.get("role") == "tool"
            ):
                break
            await asyncio.sleep(0.05)

        # Simulate SIGKILL of slow_tool.
        await harness.simulate_sigkill(session.id)

        # Inject user message to move the conversation forward.
        await harness.inject_message(session.id, "what's taking so long?")

        # Step 2: model responds to user + fast_tool result.
        await harness.run_step(session.id)

        # Now ghost repair should find slow_tool from batch 1.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_slow")

        # Session should need inference (ghost error is unreacted).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Step 3: model sees the ghost error.
        await harness.run_step(session.id)

    async def test_confirmed_always_ask_ghost(self, harness: Harness) -> None:
        """always_ask tool confirmed-allow, dispatched, then lost. Is a ghost.

        Manually constructs the event log state: an assistant message
        calling glob (a built-in tool), a tool_confirmed allow lifecycle
        event, but no tool result and no in-flight task. Ghost repair
        should detect it as a dispatched-but-lost tool.
        """
        from aios.models.agents import ToolSpec

        harness.script_model([assistant("The glob tool was interrupted.")])

        session = await harness.start(
            "find files",
            tool_specs=[ToolSpec(type="glob", permission="always_ask")],
        )

        # Manually append assistant message with tool_calls.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_g",
                        "type": "function",
                        "function": {"name": "glob", "arguments": "{}"},
                    }
                ],
                "reacting_to": 1,
            },
        )
        # Append lifecycle: client confirmed allow.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "lifecycle",
            {"event": "tool_confirmed", "tool_call_id": "call_g", "result": "allow"},
        )
        # No tool result, no in-flight task → ghost.

        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_g")

        # Session should need inference after ghost repair.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model sees the error and responds.
        await harness.run_step(session.id)


# ─── ghost exclusions ────────────────────────────────────────────────────────


@needs_docker
class TestGhostExclusions:
    async def test_unconfirmed_always_ask_not_ghost(self, harness: Harness) -> None:
        """always_ask tool waiting for client confirmation is NOT a ghost.

        Manually constructs event log: assistant calls glob (always_ask),
        no confirmation submitted. Ghost repair should skip it.
        """
        from aios.models.agents import ToolSpec

        harness.script_model([])
        session = await harness.start(
            "find files",
            tool_specs=[ToolSpec(type="glob", permission="always_ask")],
        )

        # Manually append assistant message with tool_calls.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_g",
                        "type": "function",
                        "function": {"name": "glob", "arguments": "{}"},
                    }
                ],
                "reacting_to": 1,
            },
        )
        # No confirmation, no result, no in-flight task.
        # glob is always_ask for this agent → not dispatched → NOT a ghost.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0

    async def test_tool_calls_null_not_ghost(self, harness: Harness) -> None:
        """Assistant message with tool_calls: null (JSON null) doesn't crash sweep.

        Some LiteLLM providers return tool_calls: null instead of omitting
        the key. Stored as JSONB null, this used to crash the ghost sweep's
        jsonb_array_length query. The message has no tool calls, so ghost
        repair should return nothing and the inference query should not crash.
        """
        harness.script_model([])
        session = await harness.start("hi", tools=[])

        # Manually append an assistant message with tool_calls: null.
        # This simulates what reaches the DB from providers like kimi-k2.5
        # (the ingestion fix strips it, but existing rows may have it).
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "I have no tools to call.",
                "tool_calls": None,
                "reacting_to": 1,
            },
        )

        # Ghost repair must not crash and must find no ghosts.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0

        # Inference detection must not crash either (exercises
        # _filter_incomplete_batches which has the same query pattern).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

    async def test_custom_tool_not_ghost(self, harness: Harness) -> None:
        """Custom (client-executed) tool waiting for result is NOT a ghost."""
        harness.script_model(
            [
                assistant(
                    tool_calls=[tool_call("ask_user", {"question": "yes?"}, call_id="call_u")]
                ),
            ]
        )

        # Don't register "ask_user" — it's a custom tool (not in registry).
        session = await harness.start("ask the user", tools=[])

        # Step 1: model calls the custom tool. Session idles.
        await harness.run_step(session.id)

        # Ghost repair should NOT flag it.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0


# ─── sweep waking ────────────────────────────────────────────────────────────


@needs_docker
class TestSweepWaking:
    async def test_sweep_finds_first_turn_session(self, harness: Harness) -> None:
        """Session with user message and no assistant — needs inference."""
        harness.script_model([assistant("Hello!")])
        session = await harness.start("hi")

        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

    async def test_batch_completion_gating_via_sweep(self, harness: Harness) -> None:
        """Sweep respects batch completion: waits for all tools in a group."""
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

        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("tool_a", {}, call_id="call_a"),
                        tool_call("tool_b", {}, call_id="call_b"),
                    ]
                ),
                assistant("Both done."),
            ]
        )

        session = await harness.start("run both", tools=[])
        await harness.run_step(session.id)

        # Wait for tool A to complete.
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)
        await asyncio.wait_for(tool_b_started.wait(), timeout=5.0)

        # Wait for tool A result to appear in the log.
        for _ in range(50):
            events = await harness.events(session.id)
            if any(
                e.data.get("tool_call_id") == "call_a"
                for e in events
                if e.kind == "message" and e.data.get("role") == "tool"
            ):
                break
            await asyncio.sleep(0.05)

        # Tool B is still in-flight. Sweep should say "not ready."
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

        # Let tool B complete.
        tool_b_proceed.set()
        await harness.wait_for_tools(session.id)

        # Now the batch is complete. Sweep should say "ready."
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

    async def test_user_message_bypasses_batch_gate(self, harness: Harness) -> None:
        """User message always triggers inference, even with in-flight tools."""
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"result": "done"}

        harness.register_tool("slow", slow_handler)

        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow", {}, call_id="call_s")]),
                assistant("Working on it..."),
            ]
        )

        session = await harness.start("do slow thing", tools=[])
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # Tool is in-flight. Sweep says "not ready" (batch incomplete).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

        # User sends a message. Sweep should now say "ready."
        await harness.inject_message(session.id, "status?")
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Cleanup.
        tool_proceed.set()
        await harness.wait_for_tools(session.id)
