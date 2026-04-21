"""Structured one-line formatter for session events (used by ``aios tail``).

Emits summaries keyed by event kind + role + content shape so operators
can catch silent turns, wrong-channel sends, tool-call-id corruption,
etc. at a glance. Ported from the pre-typer ``src/aios/tail.py``.

Skips transient streaming-delta payloads; only persisted events (kind
``message`` or ``lifecycle``) produce output.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from aios.cli.sse import SseMessage

MONOLOGUE_PREFIX = "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: "
CONTENT_PREVIEW_MAX = 240


def iter_formatted_events(messages: Iterable[SseMessage]) -> Iterator[str]:
    """Consume SSE messages, yield pretty-formatted event lines.

    Skips ``delta`` and non-``event`` messages; returns cleanly on ``done``.
    Lets callers drive the loop so they can decide how/where to write.
    """
    for msg in messages:
        if msg.event == "done":
            return
        if msg.event != "event":
            continue
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            continue
        line = format_event(event)
        if line is not None:
            yield line


def format_event(event: dict[str, Any]) -> str | None:
    """Format one persisted session event as a one-line summary.

    Returns ``None`` for events the tail viewer should silently skip
    (spans, unknown kinds).
    """
    seq = event.get("seq", "?")
    kind = event.get("kind")

    if kind == "message":
        return _format_message(seq, event)
    if kind == "lifecycle":
        data = event.get("data") or {}
        ev = data.get("event", "?")
        status = data.get("status", "?")
        stop = data.get("stop_reason", "?")
        return f"#{seq} LIFECYCLE {ev} (status={status}, stop_reason={stop})"
    return None


def _format_message(seq: Any, event: dict[str, Any]) -> str | None:
    data = event.get("data") or {}
    role = data.get("role")

    if role == "user":
        content = _preview(_as_text(data.get("content")))
        channel = event.get("orig_channel")
        tag = f"USER[{channel}]" if channel else "USER"
        return f"#{seq} {tag}: {content}"

    if role == "assistant":
        tool_calls = data.get("tool_calls") or []
        if tool_calls:
            rendered = ", ".join(_render_tool_call(tc) for tc in tool_calls)
            return f"#{seq} AGENT→{rendered}"
        text = _as_text(data.get("content"))
        if not text:
            return f"#{seq} AGENT(silent) ⚠ no tool, no text"
        if text.startswith(MONOLOGUE_PREFIX):
            return f"#{seq} AGENT(mono): {_preview(text[len(MONOLOGUE_PREFIX) :])}"
        return f"#{seq} AGENT(bare): {_preview(text)}"

    if role == "tool":
        tcid = data.get("tool_call_id") or "?"
        content = _preview(_as_text(data.get("content")))
        if data.get("is_error"):
            return f"#{seq} TOOL⚠ ERROR[{tcid}]: {content}"
        return f"#{seq} TOOL[{tcid}]: {content}"

    return None


def _render_tool_call(tc: dict[str, Any]) -> str:
    fn = tc.get("function") or {}
    name = fn.get("name") or "?"
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = ""
    return f"{name}: {_preview(args)}" if args else f"{name}()"


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _preview(text: str) -> str:
    text = text.replace("\n", " ")
    if len(text) <= CONTENT_PREVIEW_MAX:
        return text
    return text[: CONTENT_PREVIEW_MAX - 1] + "…"
