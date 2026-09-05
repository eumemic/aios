"""E2E: the outbound message list is a prefix-stable function of the log.

Runs the REAL ``run_session_step`` against the testcontainer Postgres with the
scripted model and asserts the prompt-prefix-cache invariant on the captured
outbound payloads: the ``messages`` sent for step N must be a message-for-
message prefix of the ``messages`` sent for step N+1.

The mid-inference case is the one that broke: an inbound that lands while the
model is generating is blind to the assistant turn that inference produces.
``build_messages`` used to defer such a message to the tail of the window and
then, once a newer assistant existed, render it back at its seq position —
re-ordering the prefix on both providers.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import litellm

from aios.models.events import Event, is_reminder_event
from tests.e2e.harness import Harness, assistant, tool_call
from tests.support import assert_message_prefix

_AGENT_MODEL = "fake/test"


def _reminder_rows(events: list[Event]) -> list[Event]:
    return [e for e in events if is_reminder_event(e.kind, e.data)]


def _context_build_ends(events: list[Event]) -> list[dict[str, Any]]:
    return [
        e.data
        for e in events
        if e.kind == "span"
        and e.data.get("event") == "context_build_end"
        and not e.data["is_error"]
    ]


class TestPrefixStability:
    async def test_inbound_during_inference_keeps_prefix_across_three_steps(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("a1"), assistant("a2"), assistant("a3")])
        session = await harness.start("hello")

        # Inject an inbound while the FIRST agent inference is in flight: wrap
        # the harness's fake completion so the append happens inside the model
        # phase of step 1 (after the window was read, before the assistant is
        # appended) — exactly the blind-spot shape.
        # The harness fixture already patched ``litellm.acompletion`` on the
        # shared module object; wrap whatever is installed there.
        original = litellm.acompletion
        injected = False

        async def hooked(**kwargs: Any) -> Any:
            nonlocal injected
            if not injected and kwargs.get("model") == _AGENT_MODEL:
                injected = True
                await harness.inject_message(session.id, "mid-inference note")
            return await original(**kwargs)

        with mock.patch("aios.harness.completion.litellm.acompletion", hooked):
            await harness.run_step(session.id)
        assert injected, "the hook never saw the agent's model call"
        # Step 2: the model replies to the blind-spot note.
        await harness.run_until_idle(session.id)
        # Step 3: a later inbound, after the note has been answered.
        await harness.inject_message(session.id, "follow-up")
        await harness.run_until_idle(session.id)

        calls = [c["messages"] for c in harness.model_calls]
        assert len(calls) == 3, f"expected 3 agent calls, got {len(calls)}"
        assert_message_prefix(calls[0], calls[1])
        assert_message_prefix(calls[1], calls[2])

    async def test_concise_agent_writes_its_reminder_once_and_keeps_prefix(
        self, harness: Harness
    ) -> None:
        """A concise agent through a tool-call step and a follow-up: the nag is
        a durable row written on the first build only, every later build is a
        prefix extension, a writing step does not re-wake the session, and the
        ``context_build_end`` span carries the change-gate telemetry."""

        async def echo(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"echo": arguments.get("text")}

        harness.register_tool("echo", echo)
        harness.script_model(
            [
                assistant("", tool_calls=[tool_call("echo", {"text": "x"})]),
                assistant("done"),
                assistant("again"),
            ]
        )
        session = await harness.start("hello", output_style="concise")
        await harness.run_until_idle(session.id)
        await harness.inject_message(session.id, "follow-up")
        await harness.run_until_idle(session.id)

        calls = [c["messages"] for c in harness.model_calls]
        assert len(calls) == 3, f"expected 3 agent calls, got {len(calls)}"
        assert_message_prefix(calls[0], calls[1])
        assert_message_prefix(calls[1], calls[2])

        messages = await harness.events(session.id)
        rows = _reminder_rows(messages)
        assert len(rows) == 1, [r.data for r in rows]
        # Written on the first build: after the opening inbound, before the
        # first assistant turn that build produced.
        first_assistant = next(e for e in messages if e.data.get("role") == "assistant")
        assert messages[0].seq < rows[0].seq < first_assistant.seq
        # The row is not a stimulus: nothing left to react to.
        assert await harness.sessions_needing_inference(session.id) == set()

        ends = _context_build_ends(await harness.all_events(session.id))
        assert [e["reminders_written"] for e in ends] == [["concise"], [], []]
        assert [e["reminders_skipped"] for e in ends] == [0, 1, 1]
        # The slate as read: the first build read ONLY the inbound — the row
        # it wrote is not counted.
        assert ends[0]["event_count_read"] == 1
