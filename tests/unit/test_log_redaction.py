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

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.contextvars import clear_contextvars

from aios.api._log_redaction import redact_sensitive_path
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
        "/v1/sessions/ses_123",
        "/v1/triggers/tr_abc",
        "/health",
        "/v1/triggers",
        "/v1/triggers/ingest",
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
