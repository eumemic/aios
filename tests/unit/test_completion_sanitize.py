"""Unit tests for assistant message normalization in completion.py."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import litellm

from aios.harness import completion
from aios.harness.completion import _normalize_message, stream_litellm


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


class TestMessageSanitizationFlag:
    """Tests for the LiteLLM ``modify_params`` cross-model-replay fix.

    Background: some OpenRouter models (e.g. minimax/Gemma) emit assistant
    turns with ``content: ""`` + ``tool_calls``. Replaying such a log
    through an Anthropic-routed model fails with::

        AnthropicException: messages: text content blocks must be non-empty

    LiteLLM's ``modify_params`` flag enables its built-in message
    sanitization — see
    https://docs.litellm.ai/docs/completion/message_sanitization.
    """

    def test_modify_params_enabled_on_import(self) -> None:
        """Importing completion.py flips the global LiteLLM flag to True."""
        # completion is already imported via the module-level import above,
        # which is what actually matters: no code path that calls acompletion
        # ever runs without completion.py being imported first.
        assert completion is not None  # anchor the import against F401
        assert litellm.modify_params is True

    async def test_stream_litellm_accepts_empty_content_with_tool_calls(
        self,
    ) -> None:
        """``stream_litellm`` completes for mixed events including ``content: ""``
        + ``tool_calls`` assistant messages against a mocked Anthropic-routed
        model — no ``BadRequestError`` raised, and ``acompletion`` sees the
        messages with the ``modify_params`` flag globally enabled so LiteLLM
        would sanitize them before hitting the Anthropic API.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "ls"},
            # The problematic shape: empty content + tool_calls.
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "README.md\n"},
        ]

        captured_kwargs: dict[str, Any] = {}

        class _FakeDelta:
            def __init__(self, content: str | None) -> None:
                self.content = content

        class _FakeChoice:
            def __init__(self, content: str | None) -> None:
                self.delta = _FakeDelta(content)

        class _FakeChunk:
            def __init__(self, content: str | None) -> None:
                self.choices = [_FakeChoice(content)]

        async def _stream() -> AsyncIterator[_FakeChunk]:
            yield _FakeChunk("done")

        async def _fake_acompletion(**kwargs: Any) -> AsyncIterator[_FakeChunk]:
            captured_kwargs.update(kwargs)
            return _stream()

        def _fake_chunk_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        # _notify_delta touches the DB; stub it so we don't need a real pool.
        async def _noop_notify(*args: Any, **kwargs: Any) -> None:
            pass

        with (
            mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
            mock.patch(
                "aios.harness.completion.litellm.stream_chunk_builder",
                _fake_chunk_builder,
            ),
            mock.patch("aios.harness.completion._notify_delta", _noop_notify),
        ):
            message, usage = await stream_litellm(
                model="anthropic/claude-sonnet-4-5",
                messages=messages,
                pool=mock.MagicMock(),
                session_id="sess_test",
            )

        assert message["content"] == "done"
        assert usage["input_tokens"] == 1
        # Anthropic-routed; modify_params is the flag LiteLLM checks
        # inside its Anthropic translation path.
        assert captured_kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert litellm.modify_params is True
