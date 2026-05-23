"""Regression: SSE first-open against a cold uvicorn (#377).

Before the preflight LISTEN fix (commit c774a35), the three connector
SSE endpoints set up their ``asyncpg.connect`` + ``add_listener``
INSIDE the ``EventSourceResponse`` generator body. When the SSE GET
was the very first request to a freshly-bound uvicorn — i.e. before
any pool slot had been warmed by a prior request — the LISTEN
connection acquire raced uvicorn's first-request init and the
chunked-response handshake half-aborted. The client saw a
``RemoteProtocolError`` ("peer closed connection without sending
complete message body") while the server logged "ASGI callable
returned without completing response."

The fix moved ``open_listen_for_*`` up into the route handler so it
runs BEFORE ``EventSourceResponse`` is constructed; preflight failure
now surfaces as a clean 503 with an aios error envelope and the
chunked stream itself is only initialised once the LISTEN is live.

These tests lock that in: a fresh uvicorn with pool ``min_size=1``,
``max_size=4``, ``lifespan="off"``, no prior HTTP traffic, and an SSE
GET as the FIRST request must complete the chunked handshake without
the server teardown showing up on the wire as
``RemoteProtocolError`` / ``ReadError``. The :func:`live_aios_server_cold`
fixture polls ``server.started`` rather than ``GET /v1/health`` so the
test's first ``client.stream(...)`` is genuinely uvicorn's first
request — a health-GET would warm the path enough to mask the bug.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from aios.config import get_settings
from aios.db.pool import create_pool
from tests.conftest import needs_docker
from tests.e2e.conftest import live_aios_server_cold
from tests.helpers.connections import mint_runtime_token_via_db

pytestmark = pytest.mark.docker

_SSE_ENDPOINTS: tuple[str, ...] = (
    "/v1/connectors/connections",
    "/v1/connectors/runtime/calls",
    "/v1/connectors/runtime/management-calls",
)


async def _mint_echo_token_via_db() -> str:
    """Open a short-lived pool, mint an ``echo`` runtime token, close it.

    The pool is closed before the test enters :func:`live_aios_server_cold`
    so the uvicorn it spawns starts with a genuinely cold state — no
    lingering asyncpg connections, no warmed pool slots, nothing for
    the SSE preflight to coast on.
    """
    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=2)
    try:
        return await mint_runtime_token_via_db(pool, connector="echo")
    finally:
        await pool.close()


async def _probe_chunked_stream_is_healthy(
    response: httpx.Response, *, deadline_s: float = 2.0
) -> None:
    """Verify the chunked-transfer stream stayed open after the headers.

    The regression we lock in is the server tearing down the chunked
    response right after the 200 OK headers go out — the SSE
    generator dies inside ``open_listen_for_*`` before yielding
    anything, so the very first body read raises
    :class:`httpx.RemoteProtocolError` ("peer closed connection
    without sending complete message body").

    Under the fixed code path, the generator parks waiting for a
    NOTIFY; with an idle DB, the next keepalive ping (default 15 s)
    is the first byte to land. We don't want a 15 s test, so we use
    a short ``deadline_s`` and treat a clean timeout as the healthy
    outcome.

    Three outcomes:

    * a chunk arrives (e.g. backfill event, ping, comment) → healthy.
    * :class:`asyncio.TimeoutError` from :func:`asyncio.wait_for`
      / :class:`httpx.ReadTimeout` → healthy (idle stream, no
      teardown).
    * :class:`httpx.RemoteProtocolError` / :class:`httpx.ReadError` →
      the #377 regression bubbles up to the caller, which fails the
      test with a clear message.
    """

    async def _read_one_chunk() -> None:
        async for chunk in response.aiter_raw():
            if chunk:
                return

    try:
        await asyncio.wait_for(_read_one_chunk(), timeout=deadline_s)
    except (TimeoutError, httpx.ReadTimeout):
        # Idle stream — the chunked handshake completed and the
        # generator is parked waiting for a NOTIFY. Exactly the
        # post-fix steady state, so this is the success path.
        return


@needs_docker
class TestSseFirstOpen:
    @pytest.mark.parametrize("endpoint", _SSE_ENDPOINTS)
    async def test_sse_first_open_succeeds_with_no_warmup_request(
        self,
        aios_env: dict[str, str],
        endpoint: str,
    ) -> None:
        """A single SSE GET as uvicorn's first request must hand off cleanly.

        The chunked-transfer handshake completes (200 + SSE
        content-type) and the server does NOT tear the connection
        down after the headers go out. Any
        ``RemoteProtocolError`` / ``ReadError`` from probing the body
        is the #377 regression — the preflight LISTEN didn't run
        before ``EventSourceResponse`` and the stream died after
        200 OK.
        """
        token = await _mint_echo_token_via_db()

        async with live_aios_server_cold() as base_url:
            try:
                async with (
                    httpx.AsyncClient(
                        base_url=base_url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    ) as client,
                    client.stream(
                        "GET", endpoint, headers={"Accept": "text/event-stream"}
                    ) as response,
                ):
                    assert response.status_code == 200, (
                        f"#377 regression: {endpoint} returned {response.status_code}, expected 200"
                    )
                    content_type = response.headers.get("content-type", "")
                    assert content_type.startswith("text/event-stream"), (
                        f"#377 regression: {endpoint} returned "
                        f"content-type {content_type!r}, expected text/event-stream"
                    )
                    await _probe_chunked_stream_is_healthy(response)
            except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                pytest.fail(
                    f"#377 regression: SSE first-open against cold uvicorn "
                    f"flapped on {endpoint}: {type(exc).__name__}: {exc}"
                )

    async def test_three_concurrent_sse_first_opens_all_succeed(
        self,
        aios_env: dict[str, str],
    ) -> None:
        """All three connector SSE streams opened concurrently against a cold
        uvicorn must each complete the chunked handshake.

        Concurrent first-opens stress the pool harder than the
        parametrized case — three preflight LISTEN acquires racing
        uvicorn's first-request init at once. The preflight fix has
        to hold up under that fan-out too, not just the single-stream
        path.
        """
        token = await _mint_echo_token_via_db()

        async with live_aios_server_cold() as base_url:

            async def _open_one(endpoint: str) -> tuple[int, str]:
                async with (
                    httpx.AsyncClient(
                        base_url=base_url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    ) as client,
                    client.stream(
                        "GET", endpoint, headers={"Accept": "text/event-stream"}
                    ) as response,
                ):
                    # Probe inside the stream context — RemoteProtocolError
                    # surfaces here under the unfixed code path, which
                    # is exactly what we want to detect.
                    await _probe_chunked_stream_is_healthy(response)
                    return (
                        response.status_code,
                        response.headers.get("content-type", ""),
                    )

            try:
                results = await asyncio.gather(
                    *(_open_one(endpoint) for endpoint in _SSE_ENDPOINTS),
                )
            except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                pytest.fail(
                    f"#377 regression: concurrent SSE first-opens flapped: "
                    f"{type(exc).__name__}: {exc}"
                )

            for endpoint, (status, content_type) in zip(_SSE_ENDPOINTS, results, strict=True):
                assert status == 200, f"#377 regression: {endpoint} returned {status}, expected 200"
                assert content_type.startswith("text/event-stream"), (
                    f"#377 regression: {endpoint} returned content-type "
                    f"{content_type!r}, expected text/event-stream"
                )
