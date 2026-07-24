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
from aios_sdk.errors import raise_for_response

# Connector discovery / calls / management streams sit idle for long
# stretches (a connector with zero connections sees zero events for
# hours).  The server emits an SSE heartbeat comment (``: ping``) every
# ``SSE_SERVER_PING_SECONDS`` so a healthy idle stream still produces a
# steady trickle of bytes — see ``aios.api.sse.make_sse_response``.
#
# An ``httpx`` read timeout is the gap allowed BETWEEN consecutive bytes,
# so it is reset by each heartbeat on a live stream but fires when a proxy
# or server silently drops the connection without a FIN (Traefik idle
# timeout on a quiet SSE response is the documented culprit — aios#962).
# We set it to a small multiple of the ping interval so two consecutive
# missed heartbeats trip the timeout: the consuming loop then surfaces
# ``httpx.ReadTimeout`` (an ``httpx.HTTPError``) and reconnects with
# backoff, and the reconnect's backfill restores correctness.  Without a
# bounded read timeout a half-open stream blocks forever and connections
# created in the meantime are invisible until the container restarts.
SSE_SERVER_PING_SECONDS = 15.0
CONNECTOR_STREAM_READ_TIMEOUT = 3 * SSE_SERVER_PING_SECONDS


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
        if response.status_code >= 400:
            # Drain the streamed body so the error envelope is available,
            # then decode it the same way every JSON op does — a bare
            # ``raise_for_status`` would leak an ``httpx.HTTPStatusError``
            # that the CLI's ``run_or_die`` does not translate.
            response.read()
            raise_for_response(response)
        yield parse_sse_lines(response.iter_lines())


@contextmanager
def stream_run(
    client: AuthenticatedClient,
    run_id: str,
    *,
    after_seq: int = 0,
) -> Iterator[Iterator[SseMessage]]:
    """Stream a workflow run's journal events as :class:`SseMessage` values.

    The workflow-run analog of :func:`stream_session` (``GET /v1/runs/{id}/stream``).
    Backfills from ``after_seq`` then tails live, ending on the ``done`` event after
    ``run_completed``. Same lazy context-manager + unbounded-read-timeout contract.
    """
    params: dict[str, int] = {"after_seq": after_seq} if after_seq else {}
    httpx_client = client.get_httpx_client()
    with httpx_client.stream(
        "GET",
        f"/v1/runs/{run_id}/stream",
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
    *,
    arm: str | None = "fresh",
    after_change_seq: int | None = None,
) -> AsyncIterator[SseMessage]:
    """Yield SSE messages from ``GET /v1/connectors/connections`` (#328 PR 5).

    Emits one ``"connection"`` event per backfilled active connection at
    subscribe time, then live ``added`` / ``removed`` events.  See
    :func:`stream_connector_calls` for the auth model.
    """
    del connector  # carried implicitly via the runtime bearer token
    params: dict[str, str | int] = {}
    if arm is not None:
        params["arm"] = arm
    if after_change_seq is not None:
        params["after_change_seq"] = after_change_seq
    async for msg in _stream_sse(httpx_client, "/v1/connectors/connections", params=params):
        yield msg


async def stream_management_calls(
    httpx_client: httpx.AsyncClient,
    connector: str,
) -> AsyncIterator[SseMessage]:
    """Yield SSE messages from ``GET /v1/connectors/runtime/management-calls``.

    Sibling of :func:`stream_connector_calls` for operator-initiated
    management operations (e.g. signal-cli ``register`` / ``verify`` /
    ``updateProfile``).  Per-connector-type only — payloads don't carry
    a ``connection_id`` because the call may target an account no
    connection yet exists for.
    """
    del connector  # carried implicitly via the runtime bearer token
    async for msg in _stream_sse(httpx_client, "/v1/connectors/runtime/management-calls"):
        yield msg


async def _stream_sse(
    httpx_client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, str | int] | None = None,
) -> AsyncIterator[SseMessage]:
    """Open an SSE stream against ``path`` and yield parsed messages.

    The first yielded message is a synthetic ``SseMessage(event="_open",
    data="")`` emitted immediately after the underlying HTTP response
    returns 2xx headers — BEFORE any server payload is parsed.  Callers
    that need a deterministic "the runtime is actually listening" signal
    (e.g. ``HttpConnector.wait_ready()``) watch for this event.

    The synthetic event is client-side only so the server-side SSE
    generator stays simple — we don't try to make the server emit a
    "connected" event before its first natural yield, which would push
    sse-starlette into territory where its ASGI body framing can crash
    pre-yield under CI load.  See aios#366 for the failure mode.
    """
    async with httpx_client.stream(
        "GET",
        path,
        params=params,
        headers={"Accept": "text/event-stream"},
        # Bounded read timeout (NOT ``read=None``): a healthy idle stream
        # is kept alive by the server's ``: ping`` heartbeat, so this only
        # fires when the stream has gone silently half-open — see
        # ``CONNECTOR_STREAM_READ_TIMEOUT`` (aios#962).
        timeout=httpx.Timeout(60.0, read=CONNECTOR_STREAM_READ_TIMEOUT),
    ) as response:
        response.raise_for_status()
        # Synthetic open marker so HttpConnector loops can mark themselves
        # ready as soon as the SSE handshake succeeds, without depending
        # on a server-emitted event (see docstring above).  The leading
        # underscore namespaces it away from any real SSE event the
        # server might emit.
        yield SseMessage(event="_open", data="")
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
