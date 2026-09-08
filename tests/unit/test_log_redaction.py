"""Unit tests for ``redact_sensitive_path`` (plan 005).

The per-trigger ingest bearer token (``aios_evt_…``) is a live credential
carried in the URL path — the sole account-key-free auth for
``POST /v1/triggers/ingest/{ingest_token}``. Before this change, the request
log sites logged the raw path verbatim on every ingest call — including
malformed/unknown-token probes. ``redact_sensitive_path`` replaces just the
token segment of that one known route; every other path passes through
unchanged.

The middleware-integration counterpart of these tests lives in
``test_request_logging_middleware.py`` (it reuses that file's ``_find_request_log``
capture helper).

These tests never use a real ``aios_evt_…``-prefixed token, only dummy
literals, so no secret-scanner flags the fixtures.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Scope
from structlog.contextvars import clear_contextvars

from aios.api._log_redaction import redact_sensitive_path
from aios.api.middleware import RequestLoggingMiddleware
from aios.errors import NotFoundError, install_exception_handlers
from aios.logging import configure_logging

_FAKE_TOKEN = "tok-not-a-real-secret"


@pytest.fixture(autouse=True)
def _clear_contextvars() -> object:
    clear_contextvars()
    yield
    clear_contextvars()


@pytest.mark.parametrize(
    "token",
    [
        _FAKE_TOKEN,
        "0123456789abcdef",
        "weird.chars_and-dashes~ok",
    ],
)
def test_ingest_token_redacted_regardless_of_shape(token: str) -> None:
    """The ingest route has nothing after the token segment, so any opaque
    token shape in that position must be redacted."""
    assert redact_sensitive_path(f"/v1/triggers/ingest/{token}") == "/v1/triggers/ingest/<redacted>"


@pytest.mark.parametrize(
    "path",
    [
        # duplicated leading slash (trailing-slash base-URL join, e.g. BASE + '/v1/...')
        f"//v1/triggers/ingest/{_FAKE_TOKEN}",
        # extra separator before the token (template that emits '/v1/triggers/ingest//{token}')
        f"/v1/triggers/ingest//{_FAKE_TOKEN}",
        # both leading and inner slash duplication at once
        f"//v1/triggers/ingest//{_FAKE_TOKEN}",
        # case variant (hardening note: prefix match is case-insensitive)
        f"/V1/triggers/ingest/{_FAKE_TOKEN}",
    ],
)
def test_ingest_token_redacted_regardless_of_prefix_slash_variation(path: str) -> None:
    """uvicorn does not normalize ``scope["path"]``, so an ingest-shaped path
    that carries a live token must still be redacted when the leading slash is
    duplicated (``//v1/...``) or an extra separator sits before the token
    (``/v1/triggers/ingest//<token>``). The redacted form is always the canonical
    single-slash output -- regardless of how many slashes the inbound path carried."""
    assert redact_sensitive_path(path) == "/v1/triggers/ingest/<redacted>"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/sessions/ses_123",
        "/v1/triggers/tr_abc",
        "/health",
        "/v1/triggers",
        "/v1/triggers/ingest",
        # bare ingest route with a trailing slash but no token -- no credential to redact
        "/v1/triggers/ingest/",
        # a sibling route whose path component merely starts with "ingest" -- not the
        # ingest route, must not be mangled
        "/v1/triggers/ingestery/foo",
        # slash duplication on a non-ingest path does not turn it into the ingest route
        "//v1/sessions/ses_123",
        "/v1/triggers//tr_abc",
    ],
)
def test_other_paths_untouched(path: str) -> None:
    assert redact_sensitive_path(path) == path


def test_error_handler_log_redacts_ingest_token(caplog: pytest.LogCaptureFixture) -> None:
    """Integration: the errors.py log site redacts the token on a 4xx/5xx
    raised from the ingest-shaped path (e.g. the real route's uniform 404 on
    an unknown/malformed token probe).

    Captured via ``caplog`` (stdlib layer) with renderer-agnostic substring
    assertions rather than ``capture_logs`` or JSON parsing. Two hazards make
    the obvious approaches flaky here:

    * ``errors.py``'s module logger is exercised across many test files and is
      cached under ``cache_logger_on_first_use=True``; ``configure_logging``
      installs a *fresh* processor-list instance on each call, so a logger
      cached against an older list is NOT intercepted by ``capture_logs``
      (which mutates only the current list) — yielding an order-dependent
      empty capture.
    * the line renders as JSON or ``key=value`` depending on
      ``sys.stderr.isatty()`` (flips under ``pytest -s``), so JSON parsing is
      fragile.

    Reading the redacted path as a plain substring of the emitted stdlib record
    is immune to both: ``caplog`` sees the record regardless of structlog proxy
    caching, and the substring holds under either renderer."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/triggers/ingest/{ingest_token}")
    async def _ingest(ingest_token: str) -> None:
        raise NotFoundError("not found", detail={})

    configure_logging("INFO")
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.WARNING):
        resp = client.post(f"/v1/triggers/ingest/{_FAKE_TOKEN}", json={})
    assert resp.status_code == 404

    messages = [r.getMessage() for r in caplog.records if "api.error" in r.getMessage()]
    assert messages, "no api.error log line captured"
    msg = messages[0]
    assert "/v1/triggers/ingest/<redacted>" in msg
    assert _FAKE_TOKEN not in msg


async def _drive_raw_asgi(app: FastAPI, path: str) -> int:
    """Invoke ``app`` as a raw ASGI app with ``scope['path']`` set verbatim and
    return the response status.

    ``TestClient`` (httpx's ASGI transport) cannot form a request line like
    ``//v1/...``: it parses a leading ``//`` as a network-path reference and the
    request never reaches the app's middleware -- so a TestClient-based
    regression test of these variants would silently mis-test and pass without
    exercising the bug. Driving a raw ASGI scope mirrors exactly what uvicorn
    delivers (it does not collapse/deduplicate slashes in ``scope['path']``).
    """
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    received = False

    async def receive() -> MutableMapping[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
        "scheme": "http",
    }
    await app(scope, receive, send)
    return next(
        (m["status"] for m in sent if m["type"] == "http.response.start"),
        0,
    )


@pytest.mark.parametrize(
    "path",
    [
        f"//v1/triggers/ingest/{_FAKE_TOKEN}",
        f"/v1/triggers/ingest//{_FAKE_TOKEN}",
        f"//v1/triggers/ingest//{_FAKE_TOKEN}",
    ],
)
async def test_noncanonical_ingest_path_redacted_through_raw_asgi(
    path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Integration, raw ASGI: an ingest-shaped path whose prefix differs from
    the canonical ``/v1/triggers/ingest/`` (a duplicated leading slash or an
    extra separator before the token) reaches uvicorn un-normalized, does NOT
    match the canonical route (-> 404), and must NOT leak the live token into
    EITHER the always-emitted ``api.request`` line (``RequestLoggingMiddleware``)
    OR the unmatched-route 404 -> ``api.http_error`` line
    (``http_exception_handler``). Both sites feed the path through
    ``redact_sensitive_path``; this regression pins the previously-bypassing
    variant prefixes to the canonical redacted form -- the one ``TestClient``
    cannot reach because httpx parses a leading ``//`` as a network-path
    reference and 404s at the transport without exercising the middleware.

    The integration harness drives the unmatched-route 404 -> ``api.http_error``
    surface, which the existing suite never exercised for ingest paths (the
    matched-route ``NotFoundError`` -> ``api.error`` path is covered by
    ``test_error_handler_log_redacts_ingest_token``).

    Capture uses ``caplog`` over the per-test structlog config the conftest
    installs (``LoggerFactory`` routed to stdlib + ``cache_logger_on_first_use=
    False``); we deliberately do NOT call ``configure_logging`` here, because
    that re-enables ``cache_logger_on_first_use=True`` and freezes the
    ``aios.api.middleware`` module-level logger proxy to this config -- which
    would then silently defeat a later ``structlog.testing.capture_logs``-based
    middleware test (e.g. ``test_request_log_line_emitted``). Substring
    assertions (event name + redacted path + token absent) are renderer-agnostic
    and hold under either the ConsoleRenderer or JSONRenderer. ``api.request``
    is INFO and ``api.http_error`` is WARNING (4xx), so the level is set at INFO
    to see both."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    install_exception_handlers(app)

    @app.post("/v1/triggers/ingest/{ingest_token}")
    async def _ingest(ingest_token: str) -> None:
        raise NotFoundError("not found", detail={})

    with caplog.at_level(logging.INFO):
        status = await _drive_raw_asgi(app, path)

    assert status == 404, f"expected the variant path to 404, got {status}"

    messages = [r.getMessage() for r in caplog.records]
    assert messages, "no log records captured"

    request_lines = [m for m in messages if "api.request" in m]
    assert request_lines, "no api.request line captured (middleware did not run)"
    http_error_lines = [m for m in messages if "api.http_error" in m]
    assert http_error_lines, (
        "no api.http_error line captured (404 did not reach http_exception_handler)"
    )

    for m in request_lines + http_error_lines:
        assert _FAKE_TOKEN not in m, f"live token leaked in log line: {m}"
        assert "/v1/triggers/ingest/<redacted>" in m, f"token not redacted in log line: {m}"
