"""E2E tests for the ``tool_execute_*`` span pair (issue #78, second stage)."""

from __future__ import annotations

from typing import Any

from tests.e2e.harness import Harness, assistant, first_tool_result, tool_call, tool_results

# NOTE: ``harness.register_tool`` auto-offers each registered tool on the agent's
# surface (see Harness.start), so a plain ``start(..., tools=[])`` is enough for the
# scripted model's call to reach REAL dispatch under the #1773 ``callable ≡ offered``
# invariant. The assertions below pin the real path (result content, not a
# "tool not offered" rejection) so the tests cannot silently re-hollow.


class TestToolExecuteSpan:
    async def test_span_pair_on_success(self, harness: Harness) -> None:
        async def echo_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"echoed": arguments.get("text", "")}

        harness.register_tool("echo", echo_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("echo", {"text": "ping"})]),
                assistant("Done."),
            ]
        )
        session = await harness.start("echo ping", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "tool_execute_start"]
        ends = [s for s in spans if s.data["event"] == "tool_execute_end"]

        assert len(starts) == 1, starts
        assert len(ends) == 1, ends

        start = starts[0]
        end = ends[0]
        assert start.data["tool_name"] == "echo"
        assert start.data["tool_call_id"]
        assert end.data["tool_execute_start_id"] == start.id
        assert end.data["tool_name"] == "echo"
        assert end.data["tool_call_id"] == start.data["tool_call_id"]
        assert end.data["is_error"] is False
        # The real echo_handler ran (not a "tool not offered" rejection): its return
        # value is round-tripped into the tool-result content.
        result = first_tool_result(events)
        assert "echoed" in str(result.get("content", ""))
        assert "tool not offered" not in str(result.get("content", ""))

    async def test_span_pair_flags_error_on_handler_exception(self, harness: Harness) -> None:
        async def failing_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        harness.register_tool("fails", failing_handler)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("fails", {})]),
                assistant("I see the tool failed."),
            ]
        )
        session = await harness.start("do the thing", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        ends = [e for e in events if e.kind == "span" and e.data["event"] == "tool_execute_end"]
        assert len(ends) == 1
        assert ends[0].data["is_error"] is True
        assert ends[0].data["tool_name"] == "fails"
        # is_error is True because the HANDLER raised — not because the tool was
        # unoffered. Pin the real exception path so this can't silently regress to a
        # "tool not offered" rejection.
        result = first_tool_result(events)
        assert "tool not offered" not in str(result.get("content", ""))
        assert "boom" in str(result.get("content", ""))

    async def test_span_pair_flags_error_on_schema_violation(self, harness: Harness) -> None:
        """Schema validation failure is still a tool-execute failure for span purposes."""

        async def strict_handler(
            session_id: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:  # pragma: no cover - never reached
            return {}

        harness.register_tool(
            "strict",
            strict_handler,
            schema={
                "type": "object",
                "properties": {"required_arg": {"type": "string"}},
                "required": ["required_arg"],
                "additionalProperties": False,
            },
        )
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("strict", {})]),  # missing required_arg
                assistant("Got the schema error."),
            ]
        )
        session = await harness.start("call strict", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        ends = [e for e in events if e.kind == "span" and e.data["event"] == "tool_execute_end"]
        assert len(ends) == 1
        assert ends[0].data["is_error"] is True
        # is_error is True because argument-schema validation failed on the real
        # dispatch path — NOT because the tool was unoffered.
        result = first_tool_result(events)
        assert "tool not offered" not in str(result.get("content", ""))
        assert "schema validation" in str(result.get("content", "")).lower()

    async def test_span_pair_per_tool_call_on_batch(self, harness: Harness) -> None:
        """Three parallel tool calls produce three span pairs, each linked correctly."""

        async def noop_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"n": arguments.get("n", 0)}

        harness.register_tool("noop", noop_handler)
        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("noop", {"n": 1}),
                        tool_call("noop", {"n": 2}),
                        tool_call("noop", {"n": 3}),
                    ]
                ),
                assistant("All three done."),
            ]
        )
        session = await harness.start("batch", tools=[])
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "tool_execute_start"]
        ends = [s for s in spans if s.data["event"] == "tool_execute_end"]

        assert len(starts) == 3
        assert len(ends) == 3

        # Each end links to its own start by both id and tool_call_id.
        start_ids = {s.id for s in starts}
        for e in ends:
            assert e.data["tool_execute_start_id"] in start_ids
        start_call_ids = {s.data["tool_call_id"] for s in starts}
        end_call_ids = {e.data["tool_call_id"] for e in ends}
        assert start_call_ids == end_call_ids
        # All three dispatched through the real noop_handler (is_error False), not
        # rejected as unoffered.
        assert all(e.data["is_error"] is False for e in ends)
        results = tool_results(events)
        assert len(results) == 3
        assert all("tool not offered" not in str(r.get("content", "")) for r in results)
