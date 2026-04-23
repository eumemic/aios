"""Unit tests for the approx_tokens estimator."""

from __future__ import annotations

from aios.harness.context import render_user_event
from aios.harness.tokens import approx_tokens


class TestApproxTokens:
    """approx_tokens delegates to litellm's local tokenizer with no
    model specified.  Exact counts depend on tokenizer version, so
    these tests pin invariants rather than specific numbers.
    """

    def test_non_empty_message_costs_something(self) -> None:
        assert approx_tokens([{"role": "user", "content": "hello world"}]) >= 1

    def test_longer_content_costs_more(self) -> None:
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "hello, this is a longer message"}]
        assert approx_tokens(long) > approx_tokens(short)

    def test_empty_list_is_cheap(self) -> None:
        """litellm charges a small constant for chat framing even on an
        empty list; we just need it not to raise and not to explode.
        """
        assert 0 <= approx_tokens([]) < 10

    def test_tool_calls_increase_cost(self) -> None:
        plain = [{"role": "assistant", "content": ""}]
        with_call = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            }
        ]
        assert approx_tokens(with_call) > approx_tokens(plain)

    def test_multi_message_list_sums(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        assert approx_tokens(msgs) > approx_tokens(msgs[:1])
        assert approx_tokens(msgs) > approx_tokens(msgs[:2])

    def test_tools_increase_cost(self) -> None:
        """Tool schemas occupy space in Anthropic's prompt; excluding them
        from the local estimate is the systematic bias issue #158 calibrates
        away. Including them narrows the gap.
        """
        msgs = [{"role": "user", "content": "list files"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command in the sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "shell command"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
        assert approx_tokens(msgs, tools=tools) > approx_tokens(msgs)


class TestApproxTokensUnderFocalRendering:
    """The cumulative_tokens column is computed against the as-rendered
    form of user events.  A non-focal inbound that renders as a short
    notification marker must cost far fewer tokens than the same inbound
    rendered with full content + metadata header.  This invariant keeps
    the chunked-window slicer honest even in busy many-to-one sessions.
    """

    _CHAN = "signal/bot/alice"

    def test_notification_smaller_than_full(self) -> None:
        long_text = "this is a moderately long message body " * 10
        metadata = {
            "channel": self._CHAN,
            "sender_uuid": "u1",
            "sender_name": "Alice",
            "timestamp_ms": 1776401210703,
        }
        data = {"role": "user", "content": long_text, "metadata": metadata}

        focal_render = render_user_event(data, self._CHAN, self._CHAN)
        notif_render = render_user_event(data, self._CHAN, "signal/bot/other")

        assert approx_tokens([notif_render]) < approx_tokens([focal_render])

    def test_legacy_null_matches_raw_data(self) -> None:
        """orig=None → Phase 2 rendering: tokens match the raw data
        (minus metadata stripping, which is content-preserving)."""
        data = {"role": "user", "content": "hello"}
        rendered = render_user_event(data, None, None)
        assert approx_tokens([rendered]) == approx_tokens([data])
