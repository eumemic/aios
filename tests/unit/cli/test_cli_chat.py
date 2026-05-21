"""Tests for ``aios chat`` rendering helpers.

Focus: regression for issue #125 — on fast/cached turns no deltas arrive,
so the final ``message`` event must render the full content (not just a
newline). The ``turn_state`` dict threads that flag through the loop.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aios.cli.commands.chat import _render_chat_event, _resolve_session, _write_delta


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


def test_resolve_session_join_returns_last_event_seq(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``aios chat --session ID`` must thread the joined session's current
    ``last_event_seq`` into the SSE backfill cursor; otherwise the very
    first ``stream_session(after_seq=0)`` replays the entire event log
    of the joined session before the user can send a message. For a
    long-running session (thousands of events) this floods stdout and
    blocks interaction for many seconds.

    The fix is to read ``last_event_seq`` off the GET response (the
    field is already on ``Session``) and return it from
    ``_resolve_session``. Same after-seq footgun class as the standing
    ``aios sessions stream`` lesson."""
    client: Any = MagicMock()
    client.request.return_value = {
        "id": "sess_X",
        "agent_id": "agt_A",
        "status": "idle",
        "last_event_seq": 1234,
    }

    result = _resolve_session(
        client,
        agent=None,
        environment_id=None,
        session="sess_X",
        title=None,
    )

    # Post-fix the function returns (session_id, last_event_seq) so the
    # REPL initialises its SSE cursor from the joined session's tail
    # rather than backfilling from 0.
    assert result == ("sess_X", 1234), (
        f"expected ('sess_X', 1234); got {result!r}. Pre-fix only the "
        f"session id was returned and the REPL hardcoded last_seq=0, "
        f"causing the first stream call to backfill the entire session "
        f"history before live events arrive."
    )
