"""E2E regression: a user message that arrives mid-inference must not strand a
prefill.

When a user message lands while the model is composing a reply, the resulting
assistant turn's ``reacting_to`` is behind that message. In seq order the message
is then stranded *before* a trailing assistant turn, so the recovery step's
context would END on an assistant message. Current reasoning models reject a
trailing assistant turn as an unsupported prefill ("the conversation must end
with a user message"), which errors the session and freezes the conversation.

The context builder must defer such a stranded message to the tail so the context
ends on a user turn (the faithful chronology — that assistant turn never saw it).
This drives the real step function, gate, and context builder through Postgres;
the scripted model faithfully rejects a trailing-assistant prefill, so the bug
errors the session pre-fix and the session recovers post-fix.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

from tests.e2e.harness import Harness, assistant


def _last_role(messages: list[dict[str, Any]]) -> str:
    return messages[-1]["role"] if messages else ""


async def test_mid_inference_user_message_does_not_strand_a_prefill(harness: Harness) -> None:
    h = harness
    h.script_model(
        [
            assistant("Octopuses have three hearts and blue blood."),
            assistant("Some jellyfish are biologically immortal."),
        ]
    )
    session = await h.start("Tell me a fun fact about octopuses.")
    sid = session.id

    # Tail role of every context the model is asked to continue.
    seen_last_roles: list[str] = []

    async def racing_model(**kwargs: Any) -> Any:
        msgs = kwargs.get("messages") or []
        seen_last_roles.append(_last_role(msgs))
        # Faithfully reject a trailing-assistant prefill, exactly as current
        # Anthropic reasoning models do — this is what errors the session.
        if _last_role(msgs) == "assistant":
            raise RuntimeError(
                "simulated provider error: this model does not support assistant "
                "message prefill; the conversation must end with a user message"
            )
        # On the FIRST step only, a second user message lands mid-inference:
        # after build_messages computed reacting_to (it saw only the first
        # message), but before this assistant turn commits. That strands it.
        if len(seen_last_roles) == 1:
            await h.inject_message(sid, "Actually, make it about jellyfish instead.")
        if kwargs.get("stream"):
            return h._pop_streaming_response(**kwargs)
        return h._pop_response(**kwargs)

    with mock.patch("aios.harness.completion.litellm.acompletion", racing_model):
        # Step 1: the model answers about octopuses while the jellyfish message
        # lands; the committed assistant turn never reacted to it.
        await h.run_step(sid)

        # Precondition sanity: the stranded condition actually exists —
        # assistant.reacting_to < jellyfish.seq < assistant.seq.
        msgs = await h.events(sid)
        asst = [e for e in msgs if e.data.get("role") == "assistant"]
        jelly = [
            e
            for e in msgs
            if e.data.get("role") == "user" and "jellyfish" in (e.data.get("content") or "")
        ]
        assert asst and jelly, "expected an assistant turn and the injected user message"
        a, u2 = asst[-1], jelly[-1]
        assert a.data["reacting_to"] < u2.seq < a.seq, (
            f"stranded precondition not met: reacting_to={a.data['reacting_to']} "
            f"jellyfish.seq={u2.seq} assistant.seq={a.seq}"
        )

        # The gate correctly wants another step — the jellyfish message is unreacted.
        assert sid in await h.sessions_needing_inference(sid)

        # Step 2: recovery. Pre-fix this builds a context ending on the assistant
        # turn (a prefill) → the model rejects it → the session errors.
        await h.run_step(sid)

    # The fix: BOTH model calls were handed a context ending on a USER turn.
    # Pre-fix the recovery context ends on the assistant turn → ["user", "assistant"].
    assert seen_last_roles == ["user", "user"], seen_last_roles

    # The session recovered cleanly and actually answered the second message.
    s = await h.session(sid)
    assert s.status == "idle", s.status
    assert s.stop_reason == {"type": "end_turn"}, s.stop_reason
    answers = [
        e.data.get("content") for e in await h.events(sid) if e.data.get("role") == "assistant"
    ]
    assert any("jellyfish" in (c or "") for c in answers), answers
