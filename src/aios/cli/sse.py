"""Synchronous Server-Sent Events parser for the client.

The aios API emits three event names on ``GET /v1/sessions/{id}/stream``:

* ``event``  — a full event row (``{"id","session_id","seq","kind","data",...}``)
* ``delta``  — a streaming text chunk for an in-progress assistant message
  (``{"delta": "..."}``).
* ``done``   — session-terminated marker; stream will close after this.

Each SSE message is a block of lines followed by a blank line. We only
care about ``event:`` and ``data:`` lines; other directives (``id:``,
``retry:``, comments beginning with ``:``) are ignored.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SseMessage:
    """A single parsed SSE message."""

    event: str
    data: str


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SseMessage]:
    """Yield :class:`SseMessage` values from an SSE line iterator.

    ``lines`` is expected to be individual lines WITHOUT trailing newlines
    (what ``httpx.Response.iter_lines()`` produces). A blank line closes the
    current message. Missing ``event:`` defaults to ``"message"`` per the
    SSE spec, but aios always sets it explicitly.
    """
    event: str = "message"
    data_parts: list[str] = []

    for raw in lines:
        if raw == "":
            if data_parts:
                yield SseMessage(event=event, data="\n".join(data_parts))
            event = "message"
            data_parts = []
            continue
        if raw.startswith(":"):
            # Comment / keep-alive ping. Ignore.
            continue
        field, _, value = raw.partition(":")
        # Per SSE spec, a single leading space after the colon is stripped.
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_parts.append(value)
        # Other fields (id, retry) are ignored.

    # Flush any trailing message without a blank-line terminator.
    if data_parts:
        yield SseMessage(event=event, data="\n".join(data_parts))
