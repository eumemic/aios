"""Unit tests for :mod:`aios.harness.stop_hooks`.

The helpers here are pure-Python — no DB, no model call — but each one
encodes a contract that the harness's continuation logic depends on:

* ``evaluate_self_check`` is the entire decision boundary between "stop"
  and "continue" for self-check hooks; a false positive on the stop_on
  word would terminate sessions mid-task.
* ``build_stop_hook_block`` is appended to the system prompt; the
  rendered prose has to mention the actual ``stop_on`` value (not a
  placeholder) so the model knows the exact word to use.
* ``continuation_message`` is what the agent sees on the next wake;
  silent defaults are how operators get a sane out-of-the-box loop.
* ``extract_assistant_text`` has to survive every chat-completions
  ``content`` shape (``None``, ``str``, multimodal parts) without
  raising — otherwise the no-tools branch crashes on a tool-only
  response.
"""

from __future__ import annotations

import pytest

from aios.harness.stop_hooks import (
    build_stop_hook_block,
    continuation_message,
    evaluate_self_check,
    extract_assistant_text,
    should_inject_task_complete,
)
from aios.models.sessions import (
    DEFAULT_CONTINUATION_MESSAGE,
    DEFAULT_SELF_CHECK_CONTINUATION_MESSAGE,
    AlwaysContinueStopHook,
    SelfCheckStopHook,
    TaskCallStopHook,
)


class TestEvaluateSelfCheck:
    """Stop iff the assistant's first word matches ``stop_on``."""

    def _hook(self, stop_on: str = "STOP") -> SelfCheckStopHook:
        return SelfCheckStopHook(prompt="Are you done?", stop_on=stop_on)

    def test_exact_first_word_match_stops(self) -> None:
        assert evaluate_self_check(self._hook(), "STOP") is True

    def test_case_insensitive_match(self) -> None:
        assert evaluate_self_check(self._hook(), "stop") is True
        assert evaluate_self_check(self._hook(), "Stop") is True

    def test_leading_whitespace_tolerated(self) -> None:
        assert evaluate_self_check(self._hook(), "   STOP   ") is True

    def test_trailing_punctuation_tolerated(self) -> None:
        # The regex picks the first word-shape; trailing "." should
        # not break the comparison (otherwise the model would have to
        # know to drop punctuation).
        assert evaluate_self_check(self._hook(), "STOP. Task complete.") is True

    def test_mid_sentence_mention_does_not_stop(self) -> None:
        """A model that explains "I am NOT going to STOP yet" must not
        terminate the session — only the FIRST word counts."""
        assert evaluate_self_check(self._hook(), "I am NOT going to STOP yet") is False

    def test_different_word_does_not_stop(self) -> None:
        assert evaluate_self_check(self._hook(), "continuing") is False

    def test_empty_text_does_not_stop(self) -> None:
        assert evaluate_self_check(self._hook(), "") is False
        assert evaluate_self_check(self._hook(), "   ") is False

    def test_custom_stop_word(self) -> None:
        hook = self._hook(stop_on="done")
        assert evaluate_self_check(hook, "done") is True
        assert evaluate_self_check(hook, "STOP") is False

    def test_punctuation_only_does_not_stop(self) -> None:
        # The first-word regex rejects bare punctuation; without a
        # match the helper must return False rather than raising.
        assert evaluate_self_check(self._hook(), "!!!") is False


class TestBuildStopHookBlock:
    """The rendered prose must reference the concrete configuration so
    the model can act on it — not template placeholders."""

    def test_none_returns_empty_string(self) -> None:
        assert build_stop_hook_block(None) == ""

    def test_self_check_includes_prompt_and_stop_on(self) -> None:
        hook = SelfCheckStopHook(prompt="Have you written the test?", stop_on="DONE")
        block = build_stop_hook_block(hook)
        assert "Have you written the test?" in block
        assert "DONE" in block
        assert "self-check" in block.lower()

    def test_task_call_mentions_task_complete(self) -> None:
        block = build_stop_hook_block(TaskCallStopHook())
        assert "task_complete" in block

    def test_always_continue_mentions_supervisor(self) -> None:
        block = build_stop_hook_block(AlwaysContinueStopHook())
        # The block has to make clear the only way out is an external
        # interrupt — otherwise the model assumes it controls termination.
        assert "supervisor" in block.lower() or "external" in block.lower()


class TestShouldInjectTaskComplete:
    """Only ``task_call`` hooks need the tool injected."""

    def test_none_hook_does_not_inject(self) -> None:
        assert should_inject_task_complete(None) is False

    def test_task_call_injects(self) -> None:
        assert should_inject_task_complete(TaskCallStopHook()) is True

    def test_self_check_does_not_inject(self) -> None:
        hook = SelfCheckStopHook(prompt="?", stop_on="STOP")
        assert should_inject_task_complete(hook) is False

    def test_always_continue_does_not_inject(self) -> None:
        assert should_inject_task_complete(AlwaysContinueStopHook()) is False


class TestContinuationMessage:
    """Operator override wins; otherwise the default depends on hook type."""

    def test_self_check_default(self) -> None:
        hook = SelfCheckStopHook(prompt="?", stop_on="STOP")
        assert continuation_message(hook) == DEFAULT_SELF_CHECK_CONTINUATION_MESSAGE

    def test_task_call_default(self) -> None:
        # task_call's no-tools branch is the same shape as always_continue:
        # both default to the generic continuation reminder.
        assert continuation_message(TaskCallStopHook()) == DEFAULT_CONTINUATION_MESSAGE

    def test_always_continue_default(self) -> None:
        assert continuation_message(AlwaysContinueStopHook()) == DEFAULT_CONTINUATION_MESSAGE

    def test_operator_override_takes_precedence(self) -> None:
        # The override is what makes per-hook continuations possible —
        # otherwise every always_continue session gets the same prose.
        custom = "[Keep working, your shift ends at midnight]"
        hook = AlwaysContinueStopHook(continuation_message=custom)
        assert continuation_message(hook) == custom

    def test_override_works_for_self_check_too(self) -> None:
        custom = "[Refine your answer until DONE]"
        hook = SelfCheckStopHook(prompt="?", stop_on="DONE", continuation_message=custom)
        assert continuation_message(hook) == custom


class TestExtractAssistantText:
    """All shapes a chat-completions ``content`` field can take."""

    def test_none_returns_empty(self) -> None:
        # Tool-only assistant responses carry content=None — must not raise.
        assert extract_assistant_text({"role": "assistant", "content": None}) == ""

    def test_missing_content_returns_empty(self) -> None:
        assert extract_assistant_text({"role": "assistant"}) == ""

    def test_plain_string(self) -> None:
        assert extract_assistant_text({"role": "assistant", "content": "hello"}) == "hello"

    def test_multimodal_text_parts(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "second"},
            ],
        }
        # Image parts skipped; text parts concatenated with newlines so
        # multi-block responses remain readable to the first-word match.
        assert extract_assistant_text(msg) == "first\nsecond"

    def test_empty_list_returns_empty(self) -> None:
        assert extract_assistant_text({"role": "assistant", "content": []}) == ""

    def test_non_text_parts_only_returns_empty(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
        }
        assert extract_assistant_text(msg) == ""


class TestModelValidation:
    """The Pydantic discriminated union must reject invalid stop_on / prompt
    so callers can't sneak past via the API layer."""

    def test_self_check_requires_non_empty_prompt(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SelfCheckStopHook(prompt="", stop_on="STOP")

    def test_self_check_requires_non_empty_stop_on(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SelfCheckStopHook(prompt="?", stop_on="")

