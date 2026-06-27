"""Unit tests for assistant message normalization in completion.py."""

from __future__ import annotations

import litellm

from aios.harness.completion import _is_persistable_thinking_block, _normalize_message


class TestNormalizeMessage:
    """Tests for _normalize_message which strips provider quirks."""

    def test_tool_calls_null_stripped(self) -> None:
        """tool_calls: None (from providers like kimi-k2.5) is removed."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "hello",
            "tool_calls": None,
        }
        result = _normalize_message(msg)
        assert "tool_calls" not in result
        assert result["content"] == "hello"

    def test_tool_calls_empty_list_preserved(self) -> None:
        """tool_calls: [] is a valid (if unusual) value — keep it."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        }
        result = _normalize_message(msg)
        assert result["tool_calls"] == []

    def test_tool_calls_with_calls_preserved(self) -> None:
        """Normal tool_calls list passes through unchanged."""
        tc = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tc],
        }
        result = _normalize_message(msg)
        assert result["tool_calls"] == [tc]

    def test_tool_calls_absent_unchanged(self) -> None:
        """Message without tool_calls key is not modified."""
        msg: dict[str, object] = {"role": "assistant", "content": "just text"}
        result = _normalize_message(msg)
        assert result == {"role": "assistant", "content": "just text"}
        assert "tool_calls" not in result

    def test_content_null_becomes_empty_string(self) -> None:
        """content: None (from providers like GPT-5.4-mini) becomes ""."""
        msg: dict[str, object] = {"role": "assistant", "content": None}
        result = _normalize_message(msg)
        assert result["content"] == ""

    def test_content_absent_becomes_empty_string(self) -> None:
        """Message missing content key entirely gets content: ""."""
        msg: dict[str, object] = {"role": "assistant"}
        result = _normalize_message(msg)
        assert result["content"] == ""

    def test_content_nonempty_preserved(self) -> None:
        """Non-null content passes through unchanged."""
        msg: dict[str, object] = {"role": "assistant", "content": "hello"}
        result = _normalize_message(msg)
        assert result["content"] == "hello"

    def test_content_null_with_tool_calls_null_both_fixed(self) -> None:
        """Both null content and null tool_calls are normalized together."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }
        result = _normalize_message(msg)
        assert result["content"] == ""
        assert "tool_calls" not in result


class TestNormalizeMessageThinkingLift:
    """``_normalize_message`` lifts Anthropic thinking blocks from
    ``provider_specific_fields.thinking_blocks`` to the top-level
    ``thinking_blocks`` key (only ``_strip_to_spec``'s whitelisted
    top-level key survives replay). Blocks with empty ``thinking`` text are
    NOT lifted: a signature without its content fails Anthropic-side
    validation on replay."""

    def test_lifts_nonempty_thinking_block_to_top_level(self) -> None:
        block = {"type": "thinking", "thinking": "real reasoning", "signature": "sig"}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {"thinking_blocks": [block]},
        }
        result = _normalize_message(msg)
        assert result["thinking_blocks"] == [block]

    def test_empty_thinking_block_lifted_then_dropped(self) -> None:
        """A block whose only payload is a signature (empty ``thinking``
        text, the ``display: "omitted"`` default) is dropped, not lifted —
        no top-level ``thinking_blocks`` results."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig"}]
            },
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_whitespace_only_thinking_block_dropped(self) -> None:
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "   ", "signature": "sig"}]
            },
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_mixed_blocks_keeps_only_nonempty(self) -> None:
        good = {"type": "thinking", "thinking": "kept", "signature": "s1"}
        empty = {"type": "thinking", "thinking": "", "signature": "s2"}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {"thinking_blocks": [good, empty]},
        }
        result = _normalize_message(msg)
        assert result["thinking_blocks"] == [good]

    def test_non_dict_provider_specific_fields_unchanged(self) -> None:
        """A non-dict ``provider_specific_fields`` must not crash, and no
        top-level ``thinking_blocks`` is synthesised."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": "not-a-dict",
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result
        assert result["content"] == "answer"

    def test_absent_provider_specific_fields_unchanged(self) -> None:
        msg: dict[str, object] = {"role": "assistant", "content": "answer"}
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_existing_top_level_thinking_blocks_preserved(self) -> None:
        """A truthy top-level ``thinking_blocks`` is left untouched — the
        lift only fills it when absent/empty, and does not clobber a value
        already present (e.g. blocks already lifted on a prior pass)."""
        existing = [{"type": "thinking", "thinking": "already here", "signature": "sig"}]
        psf_block = {"type": "thinking", "thinking": "should be ignored", "signature": "other"}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": existing,
            "provider_specific_fields": {"thinking_blocks": [psf_block]},
        }
        result = _normalize_message(msg)
        assert result["thinking_blocks"] == existing

    def test_falsy_top_level_thinking_blocks_removed_when_no_psf(self) -> None:
        """An empty/falsy top-level ``thinking_blocks`` with no liftable
        provider blocks is removed (a falsy value would otherwise confuse
        downstream replay)."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [],
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result


class TestNormalizeMessageSignatureGuard:
    """The by-construction invariant of issue #1588: a persisted thinking
    block always has a complete signature, or it is not persisted as a
    thinking block. ``_normalize_message`` is the single persist
    choke-point; a poison block (non-empty thinking text + empty/missing
    signature, the streamed-reassembly delta-loss shape) must be DROPPED so
    it can never reach the transcript and 400 on every replay."""

    def test_empty_signature_block_dropped_from_psf(self) -> None:
        """The RCA shape: real thinking text but ``signature == ""`` (the
        signature delta lost during streaming reassembly). Lifted from
        provider_specific_fields, it must be dropped, not persisted."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "real reasoning", "signature": ""}
                ]
            },
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_missing_signature_block_dropped_from_psf(self) -> None:
        """A block with thinking text but NO ``signature`` key at all
        (incomplete reassembly) is dropped — Anthropic 400s ``Field
        required`` on replay."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "real reasoning"}]
            },
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_empty_signature_block_dropped_from_top_level(self) -> None:
        """A poison block already at the top level (the streaming
        ``stream_chunk_builder`` path assembles ``thinking_blocks`` there)
        is also guarded — not just the provider_specific_fields lift."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "real reasoning", "signature": ""}
            ],
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_whitespace_signature_block_dropped(self) -> None:
        """A signature that is only whitespace is as invalid as empty."""
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "real reasoning", "signature": "   "}
            ],
        }
        result = _normalize_message(msg)
        assert "thinking_blocks" not in result

    def test_mixed_keeps_only_fully_signed(self) -> None:
        """In a mixed list, only blocks with BOTH content and a complete
        signature survive; the poison (empty-signature) block is dropped."""
        good = {"type": "thinking", "thinking": "kept", "signature": "sig"}
        poison = {"type": "thinking", "thinking": "lost-sig", "signature": ""}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [good, poison],
        }
        result = _normalize_message(msg)
        assert result["thinking_blocks"] == [good]

    def test_fully_signed_block_preserved(self) -> None:
        """A complete block (content + signature) is never dropped — the
        guard is surgical, not a blanket thinking strip."""
        good = {"type": "thinking", "thinking": "real reasoning", "signature": "sig"}
        msg: dict[str, object] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [good],
        }
        result = _normalize_message(msg)
        assert result["thinking_blocks"] == [good]


class TestIsPersistableThinkingBlock:
    """Direct unit coverage of the persist-path predicate."""

    def test_complete_block_persistable(self) -> None:
        assert _is_persistable_thinking_block(
            {"type": "thinking", "thinking": "x", "signature": "s"}
        )

    def test_empty_signature_not_persistable(self) -> None:
        assert not _is_persistable_thinking_block(
            {"type": "thinking", "thinking": "x", "signature": ""}
        )

    def test_missing_signature_not_persistable(self) -> None:
        assert not _is_persistable_thinking_block({"type": "thinking", "thinking": "x"})

    def test_empty_thinking_not_persistable(self) -> None:
        assert not _is_persistable_thinking_block(
            {"type": "thinking", "thinking": "", "signature": "s"}
        )

    def test_non_dict_not_persistable(self) -> None:
        assert not _is_persistable_thinking_block("not-a-dict")
        assert not _is_persistable_thinking_block(None)


def test_modify_params_enabled_on_import() -> None:
    """Importing aios.harness.completion sets litellm.modify_params = True.

    Without this flag, replaying an event log with ``content: ""`` +
    ``tool_calls`` assistant turns (emitted by some OpenRouter models)
    against an Anthropic-routed model fails with "text content blocks
    must be non-empty". See
    https://docs.litellm.ai/docs/completion/message_sanitization.
    """
    assert litellm.modify_params is True
