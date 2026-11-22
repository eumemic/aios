"""Render-time scrub of legacy transcript markers (#2092).

Earlier harness behavior left two strings in transcripts: the monologue stamp
channel sessions prefixed onto assistant text, and LiteLLM's empty-text
placeholder, which models copied from the wire into their own tool-call turns.
On claude-opus-5-5 a history carrying the stamp trips Anthropic's
``reasoning_extraction`` refusal, so the renderer drops both while the event log
keeps them. And on an Anthropic route a text-less tool-call turn renders with
``content: None``, so LiteLLM stops putting the placeholder on the wire at all.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from litellm.litellm_core_utils.prompt_templates import factory as litellm_factory

from aios.harness.channels import LEGACY_MONOLOGUE_STAMP
from aios.harness.context import LITELLM_EMPTY_TEXT_PLACEHOLDER, build_messages
from aios.models.events import Event

_T0 = datetime(2026, 11, 1, tzinfo=UTC)
_STAMP = "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: "
_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"
_PLAIN_MODEL = "openai/gpt-4o-mini"
_ANTHROPIC_MODEL = "anthropic/claude-opus-5-5"


def _event(seq: int, data: dict[str, Any], **extra: Any) -> Event:
    return Event(
        id=f"evt_{seq:03d}",
        session_id="sess_scrub",
        seq=seq,
        kind="message",
        data=data,
        created_at=_T0 + timedelta(seconds=seq),
        **extra,
    )


def _call(call_id: str, name: str = "search_events") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _render(events: list[Event], model: str = _PLAIN_MODEL) -> list[dict[str, Any]]:
    return build_messages(events, system_prompt=None, model=model).messages


def _assistants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "assistant"]


def test_constants_match_the_strings_in_the_log() -> None:
    assert _STAMP.strip() == LEGACY_MONOLOGUE_STAMP
    assert LITELLM_EMPTY_TEXT_PLACEHOLDER == _PLACEHOLDER
    # The placeholder is LiteLLM's: if an upgrade changes it, models will copy
    # the new string and the scrub must learn it too.
    assert litellm_factory._EMPTY_TEXT_PLACEHOLDER == LITELLM_EMPTY_TEXT_PLACEHOLDER


class TestAssistantText:
    @pytest.mark.parametrize(
        ("stored", "rendered"),
        [
            (_STAMP + "Nothing to send.", "Nothing to send."),
            (_STAMP + _STAMP.strip() + "\n\nDoubled.", "Doubled."),
            ("First.\n\n" + _STAMP + "Second.", "First.\n\nSecond."),
            (_STAMP.strip() + "\n\nNo space after the colon.", "No space after the colon."),
            (_STAMP + _PLACEHOLDER + "\n\n" + _STAMP + "Real text.", "Real text."),
            ("Plain text, untouched.", "Plain text, untouched."),
        ],
    )
    def test_markers_are_removed(self, stored: str, rendered: str) -> None:
        events = [
            _event(1, {"role": "user", "content": "hi"}),
            _event(2, {"role": "assistant", "content": stored, "reacting_to": 1}),
            _event(3, {"role": "user", "content": "next"}),
        ]
        assert _assistants(_render(events))[0]["content"] == rendered

    def test_a_turn_of_only_markers_is_dropped_as_degenerate(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "hi"}),
            _event(2, {"role": "assistant", "content": _STAMP + _PLACEHOLDER, "reacting_to": 1}),
            _event(3, {"role": "user", "content": "next"}),
        ]
        assert _assistants(_render(events)) == []

    def test_list_content_drops_blocks_emptied_by_the_scrub(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "hi"}),
            _event(
                2,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": _STAMP.strip()},
                        {"type": "text", "text": _STAMP + "kept"},
                    ],
                    "reacting_to": 1,
                },
            ),
            _event(3, {"role": "user", "content": "next"}),
        ]
        assert _assistants(_render(events))[0]["content"] == [{"type": "text", "text": "kept"}]

    def test_the_event_log_is_untouched(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "hi"}),
            _event(2, {"role": "assistant", "content": _STAMP + "x", "reacting_to": 1}),
            _event(3, {"role": "user", "content": "next"}),
        ]
        before = copy.deepcopy([e.data for e in events])
        _render(events)
        assert [e.data for e in events] == before


class TestToolCallTurns:
    def _log(self, content: str) -> list[Event]:
        return [
            _event(1, {"role": "user", "content": "hi"}),
            _event(
                2,
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [_call("t1")],
                    "reacting_to": 1,
                },
            ),
            _event(3, {"role": "tool", "tool_call_id": "t1", "content": "[]"}),
        ]

    def test_stamped_placeholder_text_is_emptied(self) -> None:
        (asst,) = _assistants(_render(self._log(_STAMP + _PLACEHOLDER)))
        assert asst["content"] == ""
        assert asst["tool_calls"][0]["id"] == "t1"

    @pytest.mark.parametrize("content", ["", "  ", _STAMP + _PLACEHOLDER])
    def test_anthropic_route_sends_no_content(self, content: str) -> None:
        (asst,) = _assistants(_render(self._log(content), model=_ANTHROPIC_MODEL))
        assert asst["content"] is None

    def test_anthropic_route_keeps_real_text(self) -> None:
        (asst,) = _assistants(_render(self._log("Checking."), model=_ANTHROPIC_MODEL))
        assert asst["content"] == "Checking."

    def test_litellm_puts_no_placeholder_on_the_anthropic_wire(self) -> None:
        """The point of ``content: None``: LiteLLM's Anthropic translation turns
        the rendered turn into ``tool_use`` blocks alone, with no placeholder text
        for the model to copy."""
        messages: Any = copy.deepcopy(
            _render(self._log(_STAMP + _PLACEHOLDER), model=_ANTHROPIC_MODEL)
        )
        wire: Any = litellm_factory.anthropic_messages_pt(
            messages=messages, model="claude-opus-5-5", llm_provider="anthropic"
        )
        assert _PLACEHOLDER not in repr(wire)
        assistant_blocks = next(m for m in wire if m["role"] == "assistant")["content"]
        assert [b["type"] for b in assistant_blocks] == ["tool_use"]

    def test_empty_string_content_would_carry_the_placeholder(self) -> None:
        """Control for the test above: the same turn with ``content: ""`` gets
        LiteLLM's placeholder on the wire — which models then imitate."""
        messages: Any = copy.deepcopy(_render(self._log(""), model=_PLAIN_MODEL))
        wire: Any = litellm_factory.anthropic_messages_pt(
            messages=messages, model="claude-opus-5-5", llm_provider="anthropic"
        )
        assert _PLACEHOLDER in repr(wire)


class TestOtherRoles:
    def test_tool_result_quoting_the_stamp_is_scrubbed(self) -> None:
        events = [
            _event(1, {"role": "user", "content": "find it"}),
            _event(
                2,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("t1")],
                    "reacting_to": 1,
                },
            ),
            _event(
                3,
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": [
                        {"type": "text", "text": f'[{{"content": "{_STAMP}an old note"}}]'},
                    ],
                },
            ),
        ]
        tool = next(m for m in _render(events) if m["role"] == "tool")
        text = repr(tool["content"])
        assert LEGACY_MONOLOGUE_STAMP not in text
        assert "an old note" in text

    def test_user_text_loses_the_stamp_but_keeps_the_placeholder(self) -> None:
        """The stamp is never meaningful content, so it goes wherever it is
        quoted. The placeholder scrub is assistant-only: an empty user text gets
        LiteLLM's placeholder back on the wire anyway."""
        events = [
            _event(1, {"role": "user", "content": f"quoting {_STAMP}this and {_PLACEHOLDER}"})
        ]
        (user,) = _render(events)
        assert LEGACY_MONOLOGUE_STAMP not in user["content"]
        assert _PLACEHOLDER in user["content"]

    def test_system_prompt_is_not_touched(self) -> None:
        system = f"Explaining the old {_STAMP}label."
        ctx = build_messages([], system_prompt=system, model=_PLAIN_MODEL)
        assert ctx.messages[0] == {"role": "system", "content": system}
