"""``aios tail <session_id>`` — structured real-time session event viewer.

Subscribes to the session's SSE stream and emits one-line summaries keyed
by event kind + role + content shape.  Feedback loop for catching silent
turns, wrong-channel sends, tool-call-id corruption, etc. during live
operator work.

Transient streaming-delta payloads (``event: delta``) are skipped so the
terminal doesn't drown in per-token notifications; only persisted events
(``event: event``) reach the formatter.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

MONOLOGUE_PREFIX = "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: "
CONTENT_PREVIEW_MAX = 240


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


# ─── SSE subscription ───────────────────────────────────────────────────────


async def _iter_sse_events(
    response: Any,
) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event_type, data)`` tuples from an httpx streaming response."""
    event_type = "message"
    data_parts: list[str] = []
    async for raw in response.aiter_lines():
        line = raw.rstrip("\r")
        if not line:
            if data_parts:
                yield event_type, "\n".join(data_parts)
            event_type = "message"
            data_parts = []
            continue
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            data_parts.append(line[6:])


async def stream(
    session_id: str,
    *,
    api_url: str,
    api_key: str,
    from_seq: int = 0,
) -> int:
    """Subscribe and print formatted events until the server closes the stream."""
    import httpx

    url = f"{api_url.rstrip('/')}/v1/sessions/{session_id}/stream"
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, str | int] = {}
    if from_seq:
        params["after_seq"] = from_seq

    try:
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("GET", url, headers=headers, params=params) as response,
        ):
            if response.status_code != 200:
                body = await response.aread()
                print(
                    f"aios tail: HTTP {response.status_code}: {body.decode(errors='replace')}",
                    file=sys.stderr,
                )
                return 2
            async for ev_type, payload in _iter_sse_events(response):
                if ev_type != "event":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                line = format_event(event)
                if line is not None:
                    print(line, flush=True)
    except KeyboardInterrupt:
        return 130
    return 0


def run(argv: list[str]) -> int:
    from_seq = 0
    positional: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"-h", "--help"}:
            print("usage: aios tail <session_id> [--from-seq N]", file=sys.stderr)
            return 0
        if arg == "--from-seq":
            if i + 1 >= len(argv):
                print("aios tail: --from-seq requires a value", file=sys.stderr)
                return 2
            try:
                from_seq = int(argv[i + 1])
            except ValueError:
                print("aios tail: --from-seq must be an integer", file=sys.stderr)
                return 2
            i += 2
            continue
        positional.append(arg)
        i += 1

    if len(positional) != 1:
        print("usage: aios tail <session_id> [--from-seq N]", file=sys.stderr)
        return 2

    session_id = positional[0]
    api_key = os.environ.get("AIOS_API_KEY")
    if not api_key:
        print("aios tail: AIOS_API_KEY is required", file=sys.stderr)
        return 2
    api_url = os.environ.get(
        "AIOS_API_URL",
        f"http://{os.environ.get('AIOS_API_HOST', '127.0.0.1')}"
        f":{os.environ.get('AIOS_API_PORT', '8080')}",
    )

    import asyncio

    return asyncio.run(stream(session_id, api_url=api_url, api_key=api_key, from_seq=from_seq))
