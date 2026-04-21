"""Tests for ``aios chat`` rendering helpers.

Focus: regression for issue #125 — on fast/cached turns no deltas arrive,
so the final ``message`` event must render the full content (not just a
newline). The ``turn_state`` dict threads that flag through the loop.
"""

from __future__ import annotations

import pytest

from aios.cli.commands.chat import _render_chat_event, _write_delta


def test_write_delta_reports_true_when_chunk_written(capsys: pytest.CaptureFixture[str]) -> None:
    assert _write_delta('{"delta": "hello"}') is True
    assert capsys.readouterr().out == "hello"


def test_write_delta_reports_false_for_empty_chunk(capsys: pytest.CaptureFixture[str]) -> None:
    assert _write_delta('{"delta": ""}') is False
    assert capsys.readouterr().out == ""


def test_write_delta_reports_false_for_missing_chunk(capsys: pytest.CaptureFixture[str]) -> None:
    assert _write_delta('{"other": "x"}') is False
    assert capsys.readouterr().out == ""


def test_write_delta_reports_false_for_bad_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert _write_delta("not-json") is False
    assert capsys.readouterr().out == ""


def test_final_assistant_message_renders_content_when_no_deltas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fast/cached turn: no delta events fired. The final message event
    must print the full content so the user sees the assistant's reply."""
    turn_state = {"streamed_any_delta": False}
    obj = {
        "kind": "message",
        "data": {"role": "assistant", "content": "hello world"},
    }
    _render_chat_event(obj, verbose=False, turn_state=turn_state)
    assert capsys.readouterr().out == "hello world\n"
    # Should be reset so the next assistant message starts fresh.
    assert turn_state["streamed_any_delta"] is False


def test_final_assistant_message_only_newline_when_deltas_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Normal streaming turn: deltas already printed the content, the
    final message event must not duplicate it — only terminate the line."""
    turn_state = {"streamed_any_delta": True}
    obj = {
        "kind": "message",
        "data": {"role": "assistant", "content": "hello world"},
    }
    _render_chat_event(obj, verbose=False, turn_state=turn_state)
    assert capsys.readouterr().out == "\n"
    # Flag resets so a subsequent assistant message in the same turn starts fresh.
    assert turn_state["streamed_any_delta"] is False
