"""Unit tests for the approx_tokens estimator."""

from __future__ import annotations

from aios.harness.context import render_user_event
from aios.harness.tokens import approx_tokens


class TestApproxTokens:
    def test_content_only(self) -> None:
        msg = {"role": "user", "content": "hello world"}
        # 11 chars / 4 = 2
        assert approx_tokens(msg) == 2

    def test_empty_content_returns_one(self) -> None:
        msg = {"role": "user", "content": ""}
        assert approx_tokens(msg) == 1

    def test_no_content_key_returns_one(self) -> None:
        msg = {"role": "assistant"}
        assert approx_tokens(msg) == 1

    def test_tool_calls_counted(self) -> None:
        msg = {
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
        # "bash" (4) + '{"command": "ls"}' (17) = 21 chars / 4 = 5
        assert approx_tokens(msg) == 5

    def test_content_plus_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/tmp/x"}'},
                }
            ],
        }
        # "Let me check." (13) + "read" (4) + '{"path": "/tmp/x"}' (18) = 35 / 4 = 8
        assert approx_tokens(msg) == 8

    def test_tool_result_message(self) -> None:
        msg = {"role": "tool", "tool_call_id": "tc_1", "content": "file contents here"}
        # 18 chars / 4 = 4
        assert approx_tokens(msg) == 4

    def test_multiple_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            ],
        }
        # "bash" (4) + "{}" (2) + "read" (4) + "{}" (2) = 12 / 4 = 3
        assert approx_tokens(msg) == 3


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

        assert approx_tokens(notif_render) < approx_tokens(focal_render)

    def test_legacy_null_matches_raw_data(self) -> None:
        """orig=None → Phase 2 rendering: tokens match the raw data
        (minus metadata stripping, which is content-preserving)."""
        data = {"role": "user", "content": "hello"}
        rendered = render_user_event(data, None, None)
        assert approx_tokens(rendered) == approx_tokens(data)
