"""Hand-written SSE consumers for aios's Server-Sent Event endpoints.

The generated SDK at ``aios_sdk._generated`` covers JSON request/response
operations only — SSE endpoints are annotated
``x-codegen.targets: []`` because Server-Sent Events don't fit
``openapi-python-client``'s response-shape model. This module provides
the streaming surface as a companion to the generated client.

Three SSE endpoints are wrapped here:

* :func:`stream_session` — session events (``GET /v1/sessions/{id}/stream``).
* :func:`stream_connector_calls` — runtime container's pending tool calls
  across N connections of one ``connector`` type
  (``GET /v1/connectors/runtime/calls``); each event is keyed ``"call"``
  with the call payload including ``connection_id`` (#328 PR 5).
* :func:`stream_connection_discovery` — runtime container's ``added`` /
  ``removed`` events for connections of one ``connector`` type
  (``GET /v1/connectors/connections``); each event is keyed
  ``"connection"`` (#328 PR 5).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
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


async def stream_connector_calls(
    httpx_client: httpx.AsyncClient,
    connector: str,
) -> AsyncIterator[SseMessage]:
    """Yield SSE messages from ``GET /v1/connectors/runtime/calls`` (#328 PR 5).

    The caller's runtime token (carried in ``httpx_client``'s default
    headers) scopes the stream to one connector *type*; the SDK fans
    out each emitted call to its per-connection worker by the
    ``connection_id`` field on the JSON payload.

    ``connector`` is passed only for the route's log keys —
    authentication happens via the bearer in the headers.
    """
    del connector  # carried implicitly via the runtime bearer token
    async for msg in _stream_sse(httpx_client, "/v1/connectors/runtime/calls"):
        yield msg


async def stream_connection_discovery(
    httpx_client: httpx.AsyncClient,
    connector: str,
) -> AsyncIterator[SseMessage]:
    """Yield SSE messages from ``GET /v1/connectors/connections`` (#328 PR 5).

    Emits one ``"connection"`` event per backfilled active connection at
    subscribe time, then live ``added`` / ``removed`` events.  See
    :func:`stream_connector_calls` for the auth model.
    """
    del connector  # carried implicitly via the runtime bearer token
    async for msg in _stream_sse(httpx_client, "/v1/connectors/connections"):
        yield msg


async def _stream_sse(httpx_client: httpx.AsyncClient, path: str) -> AsyncIterator[SseMessage]:
    """Open an SSE stream against ``path`` and yield parsed messages."""
    async with httpx_client.stream(
        "GET",
        path,
        headers={"Accept": "text/event-stream"},
        timeout=httpx.Timeout(60.0, read=None),
    ) as response:
        response.raise_for_status()
        async for msg in _aiter_sse(response):
            yield msg


async def _aiter_sse(response: httpx.Response) -> AsyncIterator[SseMessage]:
    """Parse an httpx async streaming response into :class:`SseMessage` values.

    Mirrors :func:`parse_sse_lines` for the async path — same field-by-
    field state machine, same blank-line-flushes-message semantics.
    """
    event: str = "message"
    data_parts: list[str] = []

    async for raw in response.aiter_lines():
        if raw == "":
            if data_parts:
                yield SseMessage(event=event, data="\n".join(data_parts))
            event = "message"
            data_parts = []
            continue
        if raw.startswith(":"):
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
