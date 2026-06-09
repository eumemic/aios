"""Tests for the channels tail-block gate (``_agent_owes_response``).

The ephemeral channels tail block (unread counts / previews) is appended
*after* ``build_messages`` so its per-step churn doesn't bust the prefix
cache. But when the agent owes a response — the last event-sourced message
is a user inbound or a tool result — appending the tail makes a "0 unread"
status block the literal final message. Literal-minded models (claude-fable-5)
anchor on it and emit an empty turn instead of answering. The gate suppresses
the tail in that case; it's only appended on idle/sweep re-checks where the
conversation already ends with an assistant turn.
"""

from __future__ import annotations

from aios.harness.step_context import _agent_owes_response


class TestAgentOwesResponse:
    def test_last_message_user_owes_response(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        assert _agent_owes_response(messages) is True

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
