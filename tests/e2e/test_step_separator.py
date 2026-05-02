"""E2E coverage for the adjacent-user-message separator at the LiteLLM boundary."""

from __future__ import annotations

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, msg_text

_TAIL_HEADER = "━━━ Channels ━━━"


@needs_docker
class TestSeparatorAtLiteLLMBoundary:
    async def test_adjacent_user_messages_reach_litellm_separated(self, harness: Harness) -> None:
        """When a session has interacted with a channel, ``build_channels_tail_block``
        appends a user-role message.  Paired with the user's inbound, this
        is the exact adjacency PR #69 targets.  Assert that by the time
        the message list reaches ``litellm.acompletion``, the separator is
        in place immediately before the tail block.

        Channels are now derived from event-log ``channel`` stamps (the
        connector redesign #200 replaced the explicit ``channel_bindings``
        table with this view).  An inbound with ``metadata.channel`` set
        gives the session its single bound channel.
        """
        from aios.services import sessions as sess_svc

        harness.script_model([assistant("ok")])
        session = await harness.start("hello")
        # Append an inbound stamped with channel metadata — this populates
        # ``events.channel`` and feeds the new derived bound-channels path.
        await sess_svc.append_user_message(
            harness._pool,
            session.id,
            "channel inbound",
            metadata={"channel": "signal/test/1"},
        )
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 1, (
            "expected exactly one model call — no tool round-trip in this flow"
        )
        msgs = harness.model_calls[0]["messages"]

        # Locate the channels tail block by its content header.  Content
        # may be a plain string or a cache-wrapped block list — flatten
        # before matching.
        tail_idx = next(
            (
                i
                for i, m in enumerate(msgs)
                if m.get("role") == "user" and _TAIL_HEADER in msg_text(m)
            ),
            None,
        )
        assert tail_idx is not None, f"tail block not found in messages: {msgs!r}"
        assert tail_idx > 0, "tail block should not be first — there's an inbound before it"

        prev = msgs[tail_idx - 1]
        # The separator is an empty-assistant message.  ``reasoning_content``
        # is stubbed empty too so thinking-mode providers don't reject the
        # transcript (see ``stub_missing_reasoning_content``).
        assert prev.get("role") == "assistant" and prev.get("content") == "", (
            f"expected empty-assistant separator before tail, got prev={prev!r}, tail={msgs[tail_idx]!r}"
        )
        assert prev.get("reasoning_content") == "", (
            f"expected reasoning_content stub on separator, got prev={prev!r}"
        )

    async def test_no_separator_when_tail_block_absent(self, harness: Harness) -> None:
        """Without channel bindings, ``build_channels_tail_block`` returns
        ``None`` and no adjacency arises.  Assert the separator is NOT
        inserted — guards against a future change that always inserts an
        empty assistant turn."""
        harness.script_model([assistant("ok")])
        session = await harness.start("hello")
        # No binding created → tail block is None → no user/user adjacency.
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 1
        msgs = harness.model_calls[0]["messages"]
        assert not any(m == {"role": "assistant", "content": ""} for m in msgs), (
            f"gratuitous empty-assistant separator in messages: {msgs!r}"
        )
