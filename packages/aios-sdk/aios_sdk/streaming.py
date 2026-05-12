"""Hand-written SSE consumer for the aios session-event stream.

The generated SDK at ``aios_sdk._generated`` covers JSON request/response
operations only — the session-event stream endpoint
(``GET /v1/sessions/{id}/stream``) is annotated
``x-codegen.targets: []`` because Server-Sent Events don't fit
``openapi-python-client``'s response-shape model. This module provides
the streaming surface as a companion to the generated client.

Three event names are emitted on the wire:

* ``event``  — a full event row (``{"id","session_id","seq","kind","data",...}``)
* ``delta``  — a streaming text chunk for an in-progress assistant message
  (``{"delta": "..."}``)
* ``done``   — session-terminated marker; stream will close after this
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from aios_sdk._generated import AuthenticatedClient


@dataclass(frozen=True, slots=True)
class SseMessage:
    """A single parsed SSE message."""

    event: str
    data: str


@contextmanager
def stream_session(
    client: AuthenticatedClient,
    session_id: str,
    *,
    after_seq: int = 0,
) -> Iterator[Iterator[SseMessage]]:
    """Stream events for a session as :class:`SseMessage` values.

    Used as a context manager so the underlying HTTP connection closes when
    iteration ends. The yielded iterator is consumed lazily.

    Auth is taken from the SDK client (the underlying ``httpx.Client``
    already carries the Bearer token in its default headers). Read
    timeout is unbounded — the stream can sit idle waiting for the next
    event without tripping a client-side timeout.
    """
    params: dict[str, int] = {"after_seq": after_seq} if after_seq else {}
    httpx_client = client.get_httpx_client()
    with httpx_client.stream(
        "GET",
        f"/v1/sessions/{session_id}/stream",
        params=params,
        headers={"Accept": "text/event-stream"},
        timeout=httpx.Timeout(60.0, read=None),
    ) as response:
        response.raise_for_status()
        yield parse_sse_lines(response.iter_lines())


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SseMessage]:
    """Yield :class:`SseMessage` values from an SSE line iterator.

    ``lines`` is expected to be individual lines WITHOUT trailing newlines
    (what ``httpx.Response.iter_lines()`` produces). A blank line closes
    the current message. Missing ``event:`` defaults to ``"message"`` per
    the SSE spec, but aios always sets it explicitly.
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
            # Comment / keep-alive ping.
            continue
        field, _, value = raw.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_parts.append(value)

    if data_parts:
        yield SseMessage(event=event, data="\n".join(data_parts))
