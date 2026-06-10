"""E2E tests for model refusal handling (``finish_reason == content_filter``).

A refusal is a bricked turn: litellm maps Anthropic's ``stop_reason: "refusal"``
(and OpenAI/Azure ``content_filter``) to ``finish_reason: "content_filter"``. The
response is often truncated mid-generation (a tool call whose argument was cut and
closed into valid-but-wrong JSON) or empty (refused at token 1). The harness must
NOT dispatch its tool calls and must NOT persist it as a normal assistant turn —
instead it surfaces the refusal as an errored turn (console "Errored" pill) and
parks the session until a user message recovers it.

These tests are model-agnostic: they key on the standardized ``finish_reason``,
never on a provider/model name.
"""

from __future__ import annotations

from typing import Any

from aios.harness.loop import REFUSAL_FINISH_REASON
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, tool_call


@needs_docker
class TestRefusalHandling:
    async def test_refusal_with_truncated_tool_call_does_not_dispatch(
        self, harness: Harness
    ) -> None:
        """A content_filter refusal carrying a (truncated) tool call latches the
        session into errored without dispatching the tool."""
        dispatched: list[str] = []

        async def echo_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            dispatched.append(arguments.get("text", ""))  # pragma: no cover
            return {"output": arguments.get("text", "")}  # pragma: no cover

        harness.register_tool("echo", echo_handler)
        harness.script_model(
            [
                # Refusal truncated mid-tool-call: a half-written argument the API
                # closed into valid-but-wrong JSON. Must NOT be dispatched.
                assistant(
                    tool_calls=[tool_call("echo", {"text": "do something dang"})],
                    finish_reason=REFUSAL_FINISH_REASON,
                ),
            ]
        )
        session = await harness.start("please do the thing", tools=[])
        await harness.run_step(session.id)

        # The tool was never dispatched.
        await harness.wait_for_tools(session.id)
        assert dispatched == []

        # Session is errored: idle + stop_reason.type == "error" (the console pill).
        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason is not None
        assert s.stop_reason["type"] == "error"
        assert s.stop_reason["finish_reason"] == REFUSAL_FINISH_REASON

        # The refused turn is NOT persisted as an assistant message.
        events = await harness.events(session.id)
        assistants = [e for e in events if e.data.get("role") == "assistant"]
        assert assistants == []

        # It IS recorded as a (non-replayed) span for debugging.
        all_evts = await harness.all_events(session.id)
        refusal_spans = [
            e for e in all_evts if e.kind == "span" and e.data.get("event") == "model_refusal"
        ]
        assert len(refusal_spans) == 1
        assert refusal_spans[0].data["finish_reason"] == REFUSAL_FINISH_REASON
        assert refusal_spans[0].data["is_error"] is True

    async def test_refusal_with_empty_content_latches_error(self, harness: Harness) -> None:
        """A refusal at token 1 (empty content, no tool calls) still latches error."""
        harness.script_model(
            [assistant("", finish_reason=REFUSAL_FINISH_REASON)],
        )
        session = await harness.start("something objectionable")
        await harness.run_step(session.id)

        s = await harness.session(session.id)
        assert s.status == "idle"
        assert s.stop_reason is not None
        assert s.stop_reason["type"] == "error"

        events = await harness.events(session.id)
        assert [e for e in events if e.data.get("role") == "assistant"] == []

    async def test_errored_session_not_swept(self, harness: Harness) -> None:
        """After a refusal the sweep parks the session — no re-wake into a cascade."""
        harness.script_model(
            [assistant("", finish_reason=REFUSAL_FINISH_REASON)],
        )
        session = await harness.start("trigger refusal")
        await harness.run_step(session.id)

        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

    async def test_refused_turn_absent_from_next_context(self, harness: Harness) -> None:
        """A user message recovers the session; the refused (poison) content must
        NOT appear in the recovery turn's context."""
        poison = "this half-written poison must never replay"
        harness.script_model(
            [
                # Step 1: refusal with poisonous partial content + truncated tool call.
                assistant(
                    poison,
                    tool_calls=[tool_call("noop", {"arg": "truncated"}, call_id="call_bad")],
                    finish_reason=REFUSAL_FINISH_REASON,
                ),
                # Step 2: after recovery user message, model responds normally.
                assistant("Recovered, here is a fresh answer."),
            ]
        )

        async def noop_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}  # pragma: no cover

        harness.register_tool("noop", noop_handler)

        session = await harness.start("do the thing", tools=[])
        await harness.run_step(session.id)

        # A user message lifts errored → pending (existing recovery path).
        await harness.inject_message(session.id, "let's try a different approach")
        await harness.run_until_idle(session.id)

        # The recovery step's context must not contain the refused content or its
        # truncated tool call — build_messages skips the refusal span.
        assert len(harness.model_calls) == 2
        recovery_messages = harness.model_calls[1]["messages"]
        flattened = str(recovery_messages)
        assert poison not in flattened
        assert "call_bad" not in flattened

        # And the session recovered normally.
        events = await harness.events(session.id)
        last_asst = next(e for e in reversed(events) if e.data.get("role") == "assistant")
        assert last_asst.data["content"] == "Recovered, here is a fresh answer."

    async def test_normal_finish_reason_is_not_treated_as_refusal(self, harness: Harness) -> None:
        """A normal turn (finish_reason unset / "stop") is unaffected — guards
        against an over-broad refusal predicate."""
        harness.script_model([assistant("Hello, world!", finish_reason="stop")])
        session = await harness.start("hi")
        await harness.run_until_idle(session.id)

        s = await harness.session(session.id)
        assert s.stop_reason == {"type": "end_turn"}

        events = await harness.events(session.id)
        assistants = [e for e in events if e.data.get("role") == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].data["content"] == "Hello, world!"
