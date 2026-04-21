"""Tests for the CLI's synchronous SSE parser."""

from __future__ import annotations

from aios.cli.sse import SseMessage, parse_sse_lines


def test_single_message():
    lines = ["event: event", 'data: {"seq": 1}', ""]
    messages = list(parse_sse_lines(lines))
    assert messages == [SseMessage(event="event", data='{"seq": 1}')]


def test_multiple_messages_separated_by_blank_lines():
    lines = [
        "event: event",
        'data: {"seq": 1}',
        "",
        "event: delta",
        'data: {"delta": "he"}',
        "",
        "event: done",
        "data: {}",
        "",
    ]
    messages = list(parse_sse_lines(lines))
    assert len(messages) == 3
    assert messages[0].event == "event"
    assert messages[1].event == "delta"
    assert messages[2].event == "done"


def test_multiline_data_joined_with_newlines():
    lines = ["event: event", "data: line one", "data: line two", ""]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="event", data="line one\nline two")]


def test_comments_ignored():
    lines = [":keep-alive", "event: event", 'data: {"seq": 7}', ""]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="event", data='{"seq": 7}')]


def test_trailing_message_without_blank_line_is_flushed():
    lines = ["event: done", "data: {}"]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="done", data="{}")]


def test_missing_event_defaults_to_message():
    lines = ["data: hello", ""]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="message", data="hello")]


def test_leading_space_after_colon_stripped():
    # Spec: "If value starts with a single space, remove it." Anything beyond
    # that is preserved verbatim.
    lines = ["event: event", "data:  two spaces", ""]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="event", data=" two spaces")]


def test_unrelated_fields_ignored():
    lines = ["id: abc", "retry: 1000", "event: event", 'data: {"k":1}', ""]
    msgs = list(parse_sse_lines(lines))
    assert msgs == [SseMessage(event="event", data='{"k":1}')]
