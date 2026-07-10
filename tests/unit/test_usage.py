"""Unit tests for completion.py: usage normalization and cache breakpoints."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from aios.harness.completion import (
    _CACHE_CONTROL,
    CacheChannel,
    _normalize_usage,
    inject_cache_breakpoints,
    model_descriptor,
)
from aios.harness.context import EPHEMERAL_TAIL_KEY


@pytest.fixture(autouse=True)
def _clear_model_descriptor_cache() -> None:
    """Reset the ``@cache``'d resolver before each test.

    ``model_descriptor`` is ``@cache``'d, so verdicts would otherwise stick
    across tests (including any that monkeypatch ``litellm``), producing
    order-dependent failures.
    """
    model_descriptor.cache_clear()


# Model string that LiteLLM routes through the Anthropic provider — used to
# exercise the branch of inject_cache_breakpoints that actually mutates
# messages. See TestInjectCacheBreakpointsProviderGuard for the non-Anthropic
# case.
_ANTHROPIC_MODEL = "anthropic/claude-opus-4-6"


class TestNormalizeUsage:
    """Tests for _normalize_usage which maps LiteLLM fields to our names."""

    def test_standard_openai_fields(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_fields(self) -> None:
        """Anthropic passes cache fields at the top level via LiteLLM."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 30,
        }

    def test_openai_cached_tokens_in_details(self) -> None:
        """OpenAI puts cache reads in prompt_tokens_details.cached_tokens."""
        raw = {
            "prompt_tokens": 300,
            "completion_tokens": 100,
            "total_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 120},
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 300,
            "output_tokens": 100,
            "cache_read_input_tokens": 120,
            "cache_creation_input_tokens": 0,
        }

    def test_anthropic_cache_read_takes_precedence(self) -> None:
        """Top-level cache_read_input_tokens wins over prompt_tokens_details."""
        raw = {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "cache_read_input_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 999},
        }
        result = _normalize_usage(raw)
        assert result["cache_read_input_tokens"] == 50

    def test_empty_dict(self) -> None:
        result = _normalize_usage({})
        assert result == {}

    def test_none_values_treated_as_zero(self) -> None:
        raw = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "prompt_tokens_details": None,
        }
        result = _normalize_usage(raw)
        assert result == {}

    def test_model_dump_flattened_prompt_tokens_details(self) -> None:
        """Regression: model_dump() flattens Pydantic objects to dicts with extra keys."""
        raw = {
            "prompt_tokens": 400,
            "completion_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": None},
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 400,
            "output_tokens": 120,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 0,
        }

    def test_zero_values_preserved(self) -> None:
        raw = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        result = _normalize_usage(raw)
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }


# ─── inject_cache_breakpoints ─────────────────────────────────────────────


def _msg(role: str, content: str = "") -> dict[str, Any]:
    """Build a minimal message dict."""
    return {"role": role, "content": content}


def _tail(content: str) -> dict[str, Any]:
    """Build an ephemeral-tail user message, tagged out-of-band like the
    real channels/obligations tail producers (carries
    :data:`EPHEMERAL_TAIL_KEY`)."""
    return {"role": "user", "content": content, EPHEMERAL_TAIL_KEY: True}


def _tool_def(name: str) -> dict[str, Any]:
    """Build a minimal OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": {}},
    }


class TestInjectCacheBreakpoints:
    def test_system_message_annotated(self) -> None:
        msgs = [_msg("system", "you are helpful"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[0]["content"] == [
            {"type": "text", "text": "you are helpful", "cache_control": _CACHE_CONTROL}
        ]

    def test_last_tool_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        tools = [_tool_def("bash"), _tool_def("read")]
        inject_cache_breakpoints(msgs, tools, _ANTHROPIC_MODEL)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == _CACHE_CONTROL

    def test_last_conversation_message_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": _CACHE_CONTROL}
        ]

    def test_no_system_message(self) -> None:
        """First non-system message is not annotated; only last is."""
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[0]["content"] == "hi"  # untouched
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hello", "cache_control": _CACHE_CONTROL}
        ]

    def test_no_tools(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        # No crash; system and last message still annotated via content blocks.
        assert msgs[0]["content"][0]["cache_control"] == _CACHE_CONTROL
        assert msgs[1]["content"][0]["cache_control"] == _CACHE_CONTROL

    def test_empty_messages(self) -> None:
        inject_cache_breakpoints([], None, _ANTHROPIC_MODEL)  # no crash

    def test_system_only_no_double_annotate(self) -> None:
        """When the only message is the system message, it gets one
        annotation from the system-message rule.  The last-message rule
        skips it to avoid redundancy."""
        msgs = [_msg("system", "sys")]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[0]["content"] == [
            {"type": "text", "text": "sys", "cache_control": _CACHE_CONTROL}
        ]

    def test_tool_result_as_last_message(self) -> None:
        msgs = [
            _msg("system", "sys"),
            _msg("user", "do it"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "done"},
        ]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[3]["content"] == [
            {"type": "text", "text": "done", "cache_control": _CACHE_CONTROL}
        ]

    def test_all_three_breakpoints(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        tools = [_tool_def("bash")]
        inject_cache_breakpoints(msgs, tools, _ANTHROPIC_MODEL)
        assert msgs[0]["content"][0]["cache_control"] == _CACHE_CONTROL
        assert tools[0]["cache_control"] == _CACHE_CONTROL

    def test_skips_tail_block_and_marks_prior_message(self) -> None:
        """The channels tail block mutates every step; caching it is
        pointless — every next step's tail is different, so the prefix
        cache never hits.  Breakpoint goes on the last *stable*
        message (the event-sourced one just before the tail).
        """
        tail = _tail("━━━ Channels ━━━\n▸ channel_id=x (focal)")
        msgs = [
            _msg("system", "sys"),
            _msg("user", "hi there"),
            _msg("assistant", "hello"),
            tail,
        ]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        # Last stable message (the assistant) gets the breakpoint.
        assert msgs[2]["content"] == [
            {"type": "text", "text": "hello", "cache_control": _CACHE_CONTROL}
        ]
        # Tail block stays un-annotated.
        assert msgs[3]["content"] == tail["content"]

    def test_skips_tail_and_adjacency_separator(self) -> None:
        """``inject_cache_breakpoints`` must skip *both* the tail block and
        a single-byte ``"."`` placeholder assistant separator before it —
        annotating the placeholder would be a wasted breakpoint that also
        wouldn't survive content normalization on some routes.

        ``merge_adjacent_user_messages`` no longer *produces* this
        placeholder (adjacent users are merged in place), but
        ``_is_separator_placeholder`` / the breakpoint-skip logic must
        still recognise one from any other source — pinned here against a
        hand-constructed separator.
        """
        tail = _tail("━━━ Channels ━━━\n▸ channel_id=x (focal)")
        stable = _msg("user", "real peer message")
        separator = {"role": "assistant", "content": "."}
        msgs = [_msg("system", "sys"), stable, separator, tail]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        # Breakpoint lands on the stable user message, not the separator.
        assert msgs[1]["content"] == [
            {"type": "text", "text": "real peer message", "cache_control": _CACHE_CONTROL}
        ]
        # Separator stays bare; tail stays bare.
        assert msgs[2] == {"role": "assistant", "content": "."}
        assert msgs[3]["content"] == tail["content"]

    def test_tail_only_context_falls_back_to_system_and_tool(self) -> None:
        """Degenerate case: only system + tail.  No stable conversation
        message exists — the last-stable-message rule produces nothing,
        but the system breakpoint still applies."""
        tail = _tail("━━━ Channels ━━━\n▸ channel_id=x (focal)")
        msgs = [_msg("system", "sys"), tail]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert msgs[0]["content"] == [
            {"type": "text", "text": "sys", "cache_control": _CACHE_CONTROL}
        ]
        assert msgs[1]["content"] == tail["content"]  # un-annotated

    def test_content_already_list(self) -> None:
        """When content is already a list of blocks, annotate the last block."""
        msgs = [
            _msg("system", "sys"),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part 1"},
                    {"type": "text", "text": "part 2"},
                ],
            },
        ]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert "cache_control" not in msgs[1]["content"][0]
        assert msgs[1]["content"][1]["cache_control"] == _CACHE_CONTROL

    def test_obligations_tail_after_tool_result_skipped_via_marker(self) -> None:
        """Regression for #1535: an Anthropic-route session owing a response
        (last log line a tool result) appends the always-on obligations tail
        as the final user-role line. The breakpoint must land on the last
        *stable* message (the tool result), NOT on the per-step-mutating
        obligations block — which the old ``━━━ Channels ━━━`` substring
        recognizer never matched, so it busted the prefix cache every step.
        """
        from aios.harness.obligations import _HEADER as _OBLIGATIONS_HEADER

        obligations_tail = _tail(f"{_OBLIGATIONS_HEADER}\n• req_1 [api] (open 3s)")
        msgs = [
            _msg("system", "sys"),
            _msg("user", "do the thing"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "tool done"},
            obligations_tail,
        ]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        # Breakpoint on the tool result (last stable), not the obligations tail.
        assert msgs[3]["content"] == [
            {"type": "text", "text": "tool done", "cache_control": _CACHE_CONTROL}
        ]
        # Obligations tail stays bare (content unchanged) and carries no marker.
        assert msgs[4]["content"] == obligations_tail["content"]
        assert EPHEMERAL_TAIL_KEY not in msgs[4]

    def test_merged_inbound_plus_obligation_is_ephemeral(self) -> None:
        """A trailing real user inbound + an open obligation fold into one
        ``user`` dict via ``merge_adjacent_user_messages``; the merged dict
        must inherit the ephemeral marker (sticky under OR) so the breakpoint
        skips it. This is the latent merged-dict case the old substring
        recognizer silently mishandled.
        """
        from aios.harness.context import merge_adjacent_user_messages

        inbound = _msg("user", "real peer message")
        obligation = _tail(
            "━━━ Open obligations (answer with return/error) ━━━\n• r [api] (open 3s)"
        )
        merged = merge_adjacent_user_messages([inbound, obligation])
        assert len(merged) == 1  # folded into one user turn
        assert merged[0][EPHEMERAL_TAIL_KEY] is True  # marker is sticky

        msgs = [_msg("system", "sys"), _msg("assistant", "hello"), merged[0]]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        # Breakpoint lands on the stable assistant turn, skipping the merged
        # ephemeral user dict.
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hello", "cache_control": _CACHE_CONTROL}
        ]
        assert "cache_control" not in str(msgs[2].get("content"))
        assert EPHEMERAL_TAIL_KEY not in msgs[2]

    def test_marker_stripped_on_anthropic_route(self) -> None:
        """No ``_aios_ephemeral_tail`` key may reach a provider: strip runs on
        the Anthropic path after the recognizer consumes the markers."""
        tail = _tail("━━━ Channels ━━━\n▸ channel_id=x (focal)")
        msgs = [_msg("system", "sys"), _msg("user", "hi"), tail]
        inject_cache_breakpoints(msgs, None, _ANTHROPIC_MODEL)
        assert not any(EPHEMERAL_TAIL_KEY in m for m in msgs)


class TestInjectCacheBreakpointsProviderGuard:
    """The gate that prevents cache_control injection for non-Anthropic providers.

    Reason: some OpenAI-compatible servers (notably MLX-based local model
    servers) return empty completions when a ``tool``-role message arrives
    in content-block format. Silently swallowing responses is worse than
    skipping the optimization, so we only mutate for providers that need it.
    """

    def test_openai_model_unchanged(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        tools = [_tool_def("bash")]
        inject_cache_breakpoints(msgs, tools, "openai/gpt-4o")
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "hi"
        assert "cache_control" not in tools[0]

    def test_local_openai_compat_model_unchanged(self) -> None:
        """The exact shape that surfaced the MLX-Qwen empty-response bug:
        a tool-role last message with plain-string content must remain a
        plain string after the gate kicks in."""
        msgs = [
            _msg("system", "sys"),
            _msg("user", "do it"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "done"},
        ]
        inject_cache_breakpoints(msgs, None, "openai/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ")
        assert msgs[3]["content"] == "done"  # still a string, no blocks

    def test_unknown_model_unchanged(self) -> None:
        """Unknown model strings fall through the LiteLLM provider probe and
        default to the safe no-op."""
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "completely-unknown-model-string")
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "hi"

    def test_openrouter_anthropic_annotated(self) -> None:
        """OpenRouter's ``anthropic/*`` routes forward ``cache_control`` to
        Anthropic, so they must still get injection. This is the path the
        README quickstart uses."""
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "openrouter/anthropic/claude-opus-4")
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": _CACHE_CONTROL}
        ]

    def test_openrouter_non_anthropic_unchanged(self) -> None:
        """A non-Claude model on OpenRouter stays gated out — we don't know
        whether the downstream provider tolerates content-block format."""
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "openrouter/openai/gpt-4o")
        assert msgs[1]["content"] == "hi"

    def test_bedrock_claude_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": _CACHE_CONTROL}
        ]

    def test_bedrock_non_claude_unchanged(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "bedrock/amazon.titan-text-express-v1")
        assert msgs[1]["content"] == "hi"

    def test_vertex_claude_annotated(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        inject_cache_breakpoints(msgs, None, "vertex_ai/claude-3-5-sonnet@20240620")
        assert msgs[1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": _CACHE_CONTROL}
        ]

    def test_marker_stripped_on_non_anthropic_route(self) -> None:
        """The ephemeral-tail marker is non-standard and must never reach a
        provider — even on non-Anthropic routes that place no breakpoint but
        still carry the tail dict to the wire (#1535: strip is unconditional,
        not gated behind the Anthropic early return)."""
        tail = _tail("━━━ Channels ━━━\n▸ channel_id=x (focal)")
        msgs = [_msg("system", "sys"), _msg("user", "hi"), tail]
        inject_cache_breakpoints(msgs, None, "openai/gpt-4o")
        # Gate held: no cache_control injected (content stays a plain string)…
        assert msgs[1]["content"] == "hi"
        # …but the marker is still stripped from every message.
        assert not any(EPHEMERAL_TAIL_KEY in m for m in msgs)


class TestModelDescriptor:
    """The consolidated provider-quirk resolver.

    ``model_descriptor`` collapses the three former sniffs
    (``_supports_anthropic_cache_control``, ``_supports_openai_prompt_cache_key``,
    and the inline thinking gate in ``context.py``) behind one ``@cache``'d
    pure function of the model string.
    """

    @pytest.mark.parametrize(
        ("model", "channel"),
        [
            ("anthropic/claude-opus-4-8", CacheChannel.ANTHROPIC),
            ("openrouter/anthropic/claude-fable-5", CacheChannel.ANTHROPIC),
            ("openrouter/openai/gpt-5", CacheChannel.OPENAI),
            ("openai/gpt-5", CacheChannel.OPENAI),
            ("openrouter/meta-llama/llama-4", CacheChannel.NONE),
            ("bedrock/amazon.titan", CacheChannel.NONE),
            ("completely-unknown-garbage-model-string", CacheChannel.NONE),
        ],
    )
    def test_cache_channel_table(self, model: str, channel: CacheChannel) -> None:
        assert model_descriptor(model).cache_channel is channel

    def test_garbage_string_does_not_raise(self) -> None:
        """An unroutable string collapses to a safe NONE / False, no raise."""
        desc = model_descriptor("completely-unknown-garbage-model-string")
        assert desc.cache_channel is CacheChannel.NONE
        assert desc.supports_thinking is False

    def test_thinking_true_for_claude(self) -> None:
        assert model_descriptor("anthropic/claude-opus-4-8").supports_thinking is True

    def test_thinking_true_for_bare_claude(self) -> None:
        """``supports_thinking`` matches the full model string (incl. when the
        provider probe can't resolve it), so a bare ``claude-*`` is True."""
        assert model_descriptor("claude-fable-5").supports_thinking is True


class TestModelDescriptorEquivalence:
    """Pins the resolver to the pre-refactor truth table of the two deleted
    predicates.

    The literals below ARE the old ``_supports_anthropic_cache_control`` /
    ``_supports_openai_prompt_cache_key`` verdicts for each model string,
    inlined because the predicates no longer exist. The
    ``cache_channel is ANTHROPIC`` arm must equal the old Anthropic predicate
    and the ``is OPENAI`` arm the old OpenAI predicate, for every model.
    """

    # (model, old_supports_anthropic_cache_control, old_supports_openai_prompt_cache_key)
    _TRUTH_TABLE: ClassVar[list[tuple[str, bool, bool]]] = [
        ("anthropic/claude-opus-4-8", True, False),
        ("openrouter/anthropic/claude-fable-5", True, False),
        ("openrouter/openai/gpt-5", False, True),
        ("openai/gpt-5", False, True),
        ("openrouter/meta-llama/llama-4", False, False),
        ("bedrock/amazon.titan", False, False),
        ("completely-unknown-garbage-model-string", False, False),
    ]

    @pytest.mark.parametrize(("model", "old_anthropic", "old_openai"), _TRUTH_TABLE)
    def test_equivalence(self, model: str, old_anthropic: bool, old_openai: bool) -> None:
        desc = model_descriptor(model)
        assert (desc.cache_channel is CacheChannel.ANTHROPIC) is old_anthropic
        assert (desc.cache_channel is CacheChannel.OPENAI) is old_openai
