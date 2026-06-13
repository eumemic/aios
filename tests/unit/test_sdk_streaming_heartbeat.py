"""SSE half-open detection for the connector SDK streams (aios#962).

The connector discovery / calls / management SSE streams can go silently
half-open: a proxy (Traefik) or the server drops the connection without a
FIN, and a client with no read timeout blocks forever — every connection
created in the meantime is invisible until the container restarts.

The fix is a bounded read timeout on :func:`aios_sdk.streaming._stream_sse`,
sized as a multiple of the server's ``: ping`` heartbeat interval so a
healthy idle stream (kept alive by heartbeats) never trips it, but a
silently dropped stream surfaces ``httpx.ReadTimeout`` (an
``httpx.HTTPError``) — which the connector loops already treat as a
reconnect trigger.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from aios_sdk import streaming
from aios_sdk.streaming import (
    CONNECTOR_STREAM_READ_TIMEOUT,
    SSE_SERVER_PING_SECONDS,
    _stream_sse,
)


def test_read_timeout_tolerates_healthy_idle_heartbeat() -> None:
    """The read timeout must exceed the server ping interval.

    A healthy idle stream produces a ``: ping`` every
    ``SSE_SERVER_PING_SECONDS``; each heartbeat resets httpx's read
    timeout.  If the timeout were <= the ping interval, a healthy idle
    stream would be torn down on every quiet window — exactly the
    flapping the heartbeat exists to prevent.
    """
    assert CONNECTOR_STREAM_READ_TIMEOUT > SSE_SERVER_PING_SECONDS
    # A single missed heartbeat is jitter, not death — require headroom
    # for at least two ping intervals before declaring the stream stale.
    assert CONNECTOR_STREAM_READ_TIMEOUT >= 2 * SSE_SERVER_PING_SECONDS


def test_server_ping_interval_matches_api_heartbeat() -> None:
    """The SDK's assumed ping interval must equal the server's actual one.

    If the API's ``SSE_PING_SECONDS`` drifts above the SDK's assumption,
    a healthy idle stream could miss the read-timeout window and reconnect
    needlessly; if it drifts well below, staleness detection lags.  Pin
    them together so the two halves of the heartbeat contract stay in sync.
    """
    from aios.api.sse import SSE_PING_SECONDS

    assert float(SSE_PING_SECONDS) == SSE_SERVER_PING_SECONDS


class _RecordingStreamClient:
    """Captures the ``timeout`` ``_stream_sse`` hands to ``.stream(...)``.

    httpx only enforces read timeouts inside its real network transport
    (a custom ``AsyncBaseTransport`` bypasses that machinery), so we can't
    provoke a genuine ``ReadTimeout`` deterministically in-process.  We
    instead assert the bounded timeout is wired through to the request —
    httpx's own test suite owns the "this timeout raises ReadTimeout"
    contract.  The body yields nothing, so the stream ends cleanly after
    the synthetic ``_open`` marker.
    """

    def __init__(self) -> None:
        self.captured_timeout: httpx.Timeout | None = None

    @asynccontextmanager
    async def stream(self, method: str, path: str, **kwargs: Any) -> AsyncIterator[httpx.Response]:
        del method, path
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, httpx.Timeout)
        self.captured_timeout = timeout
        response = httpx.Response(200, headers={"content-type": "text/event-stream"})

        async def _empty() -> AsyncIterator[str]:
            return
            yield  # pragma: no cover - makes this an async generator

        def _ok() -> httpx.Response:
            return response

        response.aiter_lines = _empty  # type: ignore[method-assign]
        response.raise_for_status = _ok  # type: ignore[method-assign]
        yield response


async def test_stream_sse_uses_bounded_read_timeout() -> None:
    """``_stream_sse`` opens the stream with a finite read timeout.

    ``read=None`` is the bug (blocks forever on a half-open stream); the
    fix is ``read=CONNECTOR_STREAM_READ_TIMEOUT``.
    """
    client = _RecordingStreamClient()
    events = [msg.event async for msg in _stream_sse(client, "/v1/connectors/connections")]  # type: ignore[arg-type]

    assert events == ["_open"]
    assert client.captured_timeout is not None
    # The bug regressions on a None read timeout; the fix is a finite one.
    assert client.captured_timeout.read == CONNECTOR_STREAM_READ_TIMEOUT
    assert client.captured_timeout.read is not None


def test_read_timeout_is_an_http_error() -> None:
    """``httpx.ReadTimeout`` is an ``httpx.HTTPError``.

    The connector loops catch ``httpx.HTTPError`` to reconnect with
    backoff; this is what turns a tripped read timeout into a recovery
    instead of an unhandled crash.  Pin the relationship so the recovery
    path can't silently break if httpx's exception hierarchy shifts.
    """
    assert issubclass(httpx.ReadTimeout, httpx.HTTPError)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/connectors/connections",
        "/v1/connectors/runtime/calls",
        "/v1/connectors/runtime/management-calls",
    ],
)
async def test_all_connector_streams_bounded(path: str) -> None:
    """Every connector SSE stream goes through the bounded-timeout path.

    Discovery, tool-call, and management-call streams all share
    ``_stream_sse`` (per the spec: the management/tool streams share the
    SSE/long-poll pattern and the same half-open risk), so all three get
    the read timeout.
    """
    client = _RecordingStreamClient()
    _ = [msg async for msg in _stream_sse(client, path)]  # type: ignore[arg-type]
    assert client.captured_timeout is not None
    assert client.captured_timeout.read == CONNECTOR_STREAM_READ_TIMEOUT


def test_streaming_module_constant_default() -> None:
    """Sanity-check the production default is unchanged by monkeypatching."""
    assert streaming.CONNECTOR_STREAM_READ_TIMEOUT == 3 * SSE_SERVER_PING_SECONDS
