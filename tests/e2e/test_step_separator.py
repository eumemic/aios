"""E2E coverage for the adjacent-user-message separator at the LiteLLM boundary.

PR #69 inserts ``{"role": "assistant", "content": ""}`` between any two
consecutive user-role messages in ``run_session_step`` so LiteLLM's
Anthropic translator doesn't merge them into one multi-content turn.
These tests exercise the full pipeline through ``run_session_step`` and
assert on ``harness.model_calls`` (captured kwargs at the
``litellm.acompletion`` boundary) to pin down that the fix actually
reaches the wire.
"""

from __future__ import annotations

from typing import Any

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant

_TAIL_HEADER = "━━━ Channels ━━━"


def _msg_text(msg: dict[str, Any]) -> str:
    """Flatten a chat-completions message's content to a string.

    LiteLLM's Anthropic adapter wraps cache-eligible content as
    ``[{"type": "text", "text": "...", "cache_control": ...}]`` blocks
    before the request leaves the boundary, so tests that read
    ``harness.model_calls[...]['messages']`` must handle both the plain
    string shape and the multi-block shape.
    """
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(str(block.get("text", "")) for block in c if isinstance(block, dict))
    return ""


@needs_docker
class TestSeparatorAtLiteLLMBoundary:
    async def test_adjacent_user_messages_reach_litellm_separated(self, harness: Harness) -> None:
        """When a session has channel bindings, ``build_channels_tail_block``
        appends a user-role message.  Paired with the user's inbound, this
        is the exact adjacency PR #69 targets.  Assert that by the time
        the message list reaches ``litellm.acompletion``, the separator is
        in place immediately before the tail block."""
        from aios.services import channels as ch_svc

        harness.script_model([assistant("ok")])
        session = await harness.start("hello")
        await ch_svc.create_binding(harness._pool, address="signal/test/1", session_id=session.id)
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
                if m.get("role") == "user" and _TAIL_HEADER in _msg_text(m)
            ),
            None,
        )
        assert tail_idx is not None, f"tail block not found in messages: {msgs!r}"
        assert tail_idx > 0, "tail block should not be first — there's an inbound before it"

        # The message immediately before the tail block must defeat the
        # Anthropic merge: either a role transition (non-user) or
        # explicitly the empty-assistant separator the fix inserts.
        prev = msgs[tail_idx - 1]
        assert prev.get("role") != "user" or prev == {"role": "assistant", "content": ""}, (
            f"user/user adjacency survived to litellm boundary: prev={prev!r}, tail={msgs[tail_idx]!r}"
        )

        # Stronger: in this specific setup (one inbound + tail block) the
        # preceding message should be the empty-assistant separator.
        assert prev == {"role": "assistant", "content": ""}, (
            f"expected empty-assistant separator before tail, got {prev!r}"
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
