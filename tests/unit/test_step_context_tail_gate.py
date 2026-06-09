"""Tests for the channels tail-block gate (``_agent_owes_response``).

The ephemeral channels tail block (unread counts / previews) is appended
*after* ``build_messages`` so its per-step churn doesn't bust the prefix
cache. But when the agent owes a response — the last event-sourced message
is a focal user inbound or a tool result — appending the tail makes a "0 unread"
status block the literal final message. Literal-minded models (claude-fable-5)
anchor on it and emit an empty turn instead of answering. The gate suppresses
the tail in that case. It is kept (1) on idle/sweep re-checks where the
conversation already ends with an assistant turn, and (2) when the trailing
message is a non-focal notification marker (``🔔 …``), whose navigation
companion *is* the channels tail listing.
"""

from __future__ import annotations

from aios.harness.step_context import _agent_owes_response

# Mirrors the shape of context._format_notification_marker output.
_NOTIFICATION = (
    "🔔 channel_id=signal/+1/chat-a · channel inbound\n"
    "(to respond, call switch_channel(channel_id='signal/+1/chat-a') first)"
)


class TestAgentOwesResponse:
    def test_last_message_focal_user_owes_response(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[received=...]\nplease respond"},
        ]
        assert _agent_owes_response(messages) is True

    def test_last_message_notification_marker_does_not_owe_response(self) -> None:
        # A non-focal 🔔 marker is a navigation prompt, not a direct stimulus —
        # keep the tail so the agent can see channel state and switch_channel.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[received=...]\nhi"},
            {"role": "assistant", "content": "."},
            {"role": "user", "content": _NOTIFICATION},
        ]
        assert _agent_owes_response(messages) is False

    def test_notification_marker_in_list_content_does_not_owe(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": _NOTIFICATION}]},
        ]
        assert _agent_owes_response(messages) is False

    def test_last_message_tool_owes_response(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        assert _agent_owes_response(messages) is True

    def test_last_message_assistant_does_not_owe_response(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _agent_owes_response(messages) is False

    def test_empty_list_does_not_owe_response(self) -> None:
        assert _agent_owes_response([]) is False
