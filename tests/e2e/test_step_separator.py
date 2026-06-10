"""E2E coverage for the adjacent-user-message merge at the LiteLLM boundary.

Two consecutive user-role messages must reach ``litellm.acompletion`` as a
single merged user turn (Anthropic requires alternating roles). The earlier
design inserted a placeholder ``{"role": "assistant", "content": "."}``
separator between them; that degenerate turn taught literal-minded models
(claude-fable-5) to imitate silence, so it was replaced by an in-place merge.
These tests pin that no ``"."`` placeholder ever reaches the wire and that the
adjacency arrives merged.
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, msg_text

pytestmark = pytest.mark.docker

_TAIL_HEADER = "━━━ Channels ━━━"


def _is_dot_placeholder(m: dict) -> bool:
    return m.get("role") == "assistant" and m.get("content") == "."


@needs_docker
class TestAdjacentUserMergeAtLiteLLMBoundary:
    async def test_adjacent_user_inbounds_reach_litellm_merged(self, harness: Harness) -> None:
        """Two user inbounds that land before the agent acts are adjacent
        user-role messages. By the time the message list reaches
        ``litellm.acompletion`` they must be a single merged user turn —
        not separated by a ``"."`` placeholder assistant.

        The merge folds string contents with a blank-line join, so both
        inbounds' text is present in one user message.
        """
        harness.script_model([assistant("ok")])
        session = await harness.start("first inbound")
        # A second inbound arrives before any inference step runs — the two
        # user messages are now back-to-back in the event log.
        await harness.inject_message(session.id, "second inbound")
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 1, (
            "expected exactly one model call — no tool round-trip in this flow"
        )
        msgs = harness.model_calls[0]["messages"]

        # No degenerate "." placeholder anywhere on the wire.
        assert not any(_is_dot_placeholder(m) for m in msgs), (
            f"degenerate '.' placeholder reached litellm: {msgs!r}"
        )

        # Exactly one user-role message, carrying BOTH inbounds (merged).
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1, f"expected one merged user turn, got {user_msgs!r}"
        merged = msg_text(user_msgs[0])
        assert "first inbound" in merged
        assert "second inbound" in merged

    async def test_inbound_then_tail_block_merge_no_placeholder(self, harness: Harness) -> None:
        """When a session has interacted with a channel,
        ``build_channels_tail_block`` appends a user-role tail. If the
        conversation already ends with an assistant turn (idle re-check),
        the tail lands adjacent to nothing problematic; but when an
        inbound precedes it the two user turns merge. Either way, assert
        no ``"."`` placeholder reaches litellm and the tail content is
        carried by a user-role message.

        Channels are derived from event-log ``channel`` stamps (connector
        redesign #200). An inbound with ``metadata.channel`` set gives the
        session its single bound channel; a following assistant turn means
        the conversation ends assistant-side, so the tail is appended.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sess_svc

        # Two scripted replies: one for the channel inbound, one for the
        # idle re-check step where the tail block is appended.
        harness.script_model([assistant("ack"), assistant("idle")])
        session = await harness.start("hello")
        await sess_svc.append_user_message(
            harness._pool,
            session.id,
            "channel inbound",
            metadata={"channel": "signal/test/1"},
            account_id=account_id,
        )
        await harness.run_until_idle(session.id)

        # No "." placeholder on any model call.
        for call in harness.model_calls:
            assert not any(_is_dot_placeholder(m) for m in call["messages"]), (
                f"degenerate '.' placeholder reached litellm: {call['messages']!r}"
            )

        # Across all calls, when the channels tail block appears it is on a
        # user-role message (merged into the trailing inbound or standalone).
        saw_tail = False
        for call in harness.model_calls:
            for m in call["messages"]:
                if m.get("role") == "user" and _TAIL_HEADER in msg_text(m):
                    saw_tail = True
        assert saw_tail, (
            "channels tail block never appeared on a user-role message across "
            f"model calls: {[c['messages'] for c in harness.model_calls]!r}"
        )

    async def test_no_placeholder_when_tail_block_absent(self, harness: Harness) -> None:
        """Without channel bindings, ``build_channels_tail_block`` returns
        ``None`` and no adjacency arises from the tail. Assert no
        placeholder assistant turn is inserted — guards against a future
        change that always inserts a degenerate empty assistant turn."""
        harness.script_model([assistant("ok")])
        session = await harness.start("hello")
        # No binding created → tail block is None → no user/user adjacency.
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 1
        msgs = harness.model_calls[0]["messages"]
        assert not any(_is_dot_placeholder(m) for m in msgs), (
            f"gratuitous placeholder-assistant separator in messages: {msgs!r}"
        )
