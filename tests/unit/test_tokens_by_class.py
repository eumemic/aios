"""Unit tests for the per-content-class token estimator (issue #1609).

``content_class`` / ``approx_tokens_by_class`` are the Layer-1 training-data
producers: they attribute the model-neutral local token count across content
classes without touching the neutral baseline counter itself.
"""

from __future__ import annotations

from aios.harness.tokens import (
    CONTENT_CLASSES,
    approx_tokens,
    approx_tokens_by_class,
    content_class,
)


class TestContentClass:
    def test_system(self) -> None:
        assert content_class("system", {"role": "system", "content": "x"}) == "system"

    def test_tool_result(self) -> None:
        assert content_class("tool", {"role": "tool", "content": "r"}) == "tool_result"

    def test_user_is_text(self) -> None:
        assert content_class("user", {"role": "user", "content": "hi"}) == "text"

    def test_plain_assistant_is_text(self) -> None:
        assert content_class("assistant", {"role": "assistant", "content": "ok"}) == "text"

    def test_assistant_tool_calls_is_tool_use(self) -> None:
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]}
        assert content_class("assistant", msg) == "tool_use"

    def test_assistant_thinking_is_thinking(self) -> None:
        msg = {"role": "assistant", "content": "x", "reasoning_content": "deliberation"}
        assert content_class("assistant", msg) == "thinking"

    def test_tool_use_dominates_thinking(self) -> None:
        # A turn with both thinking AND tool calls classifies as tool_use at
        # the message level (the windower's dominant-class re-derivation).
        msg = {
            "role": "assistant",
            "content": "x",
            "reasoning_content": "deliberation",
            "tool_calls": [{"id": "a"}],
        }
        assert content_class("assistant", msg) == "tool_use"


class TestApproxTokensByClass:
    def test_keys_are_the_class_set(self) -> None:
        bc = approx_tokens_by_class([{"role": "user", "content": "hi"}])
        assert set(bc.keys()) == set(CONTENT_CLASSES)

    def test_empty_is_all_zero(self) -> None:
        bc = approx_tokens_by_class([])
        assert all(v == 0 for v in bc.values())

    def test_text_attributed_to_text(self) -> None:
        bc = approx_tokens_by_class([{"role": "user", "content": "hello there friend"}])
        assert bc["text"] > 0
        assert bc["tool_result"] == 0
        assert bc["thinking"] == 0
        assert bc["tool_use"] == 0

    def test_tool_result_attributed_to_tool_result(self) -> None:
        bc = approx_tokens_by_class(
            [{"role": "tool", "tool_call_id": "a", "content": "result payload text"}]
        )
        assert bc["tool_result"] > 0
        assert bc["text"] == 0

    def test_thinking_turn_credits_both_text_and_thinking(self) -> None:
        msg = {
            "role": "assistant",
            "content": "the visible answer",
            "reasoning_content": "a long internal chain of deliberation about the answer",
        }
        bc = approx_tokens_by_class([msg])
        assert bc["thinking"] > 0
        assert bc["text"] > 0
        assert bc["tool_use"] == 0

    def test_tool_use_turn_credits_tool_use(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": '{"x": 1}'},
                }
            ],
        }
        bc = approx_tokens_by_class([msg])
        assert bc["tool_use"] > 0

    def test_tools_overhead_attributed_to_tools(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "description": "does a thing",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        bc = approx_tokens_by_class([{"role": "user", "content": "hi"}], tools=tools)
        assert bc["tools"] > 0

    def test_sum_is_positive_and_nonnegative_per_class(self) -> None:
        # The per-class split must be a positive, deterministic function of the
        # same neutral counter — never crash, never negative. It is NOT the
        # stored baseline: loop.py stamps local_tokens = approx_tokens(...) (a
        # single call) and by_class separately, so sum(by_class.values()) may
        # exceed local_tokens (each slice carries its own framing overhead).
        # The implementation deliberately does not enforce that equality.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ans", "reasoning_content": "think"},
            {"role": "tool", "tool_call_id": "a", "content": "res"},
        ]
        bc = approx_tokens_by_class(msgs)
        assert sum(bc.values()) > 0
        assert all(v >= 0 for v in bc.values())
        # The per-class split sums to AT LEAST the neutral baseline (each slice
        # is costed in isolation and re-pays per-message framing), so it must
        # never be MISTAKEN for the stored baseline: loop.py keeps them as two
        # separate counts on purpose (issue #1609 constraint #1).
        assert sum(bc.values()) >= approx_tokens(msgs)

    def test_neutral_counter_unchanged(self) -> None:
        # Constraint #1: approx_tokens (the stored baseline) is untouched.
        msgs = [{"role": "user", "content": "hello there"}]
        # Sanity: still returns the plain neutral count.
        assert approx_tokens(msgs) > 0
