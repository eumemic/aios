"""Unit tests for the approx_tokens estimator."""

from __future__ import annotations

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
