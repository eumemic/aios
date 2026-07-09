"""Unit tests for ``RequestLoggingMiddleware`` (pure-ASGI request logging).

The middleware emits exactly one ``api.request`` structlog line per http
request with method, path, status and duration_ms. Because it is a pure-ASGI
middleware (same task context as the route), an ``account_id`` bound by an
auth dependency via ``bind_contextvars`` is visible on the request line through
``merge_contextvars`` — the core guarantee these tests pin.

structlog renders the event dict into a single message string before it
reaches stdlib logging (the configured ``LoggerFactory`` + renderer), so the
key/value fields are NOT attributes on a ``caplog`` record. We capture the
structured event dicts directly with ``structlog.testing.capture_logs`` and
read the fields off those.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.testing import capture_logs

from aios.api.middleware import RequestLoggingMiddleware
from aios.logging import configure_logging


@pytest.fixture(autouse=True)
def _clear_contextvars() -> Any:
    """Keep ``account_id`` (and any other) bindings from leaking across tests."""
    clear_contextvars()
    yield
    clear_contextvars()


def _find_request_log(
    entries: list[MutableMapping[str, Any]],
) -> MutableMapping[str, Any] | None:
    """Return the first captured ``api.request`` event dict, or ``None``."""
    for entry in entries:
        if entry.get("event") == "api.request":
            return entry
    return None


def test_request_log_line_emitted() -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    with capture_logs() as entries:
        resp = client.get("/ping")
    assert resp.status_code == 200

    rec = _find_request_log(entries)
    assert rec is not None
    assert rec["method"] == "GET"
    assert rec["path"] == "/ping"
    assert rec["status"] == 200
    assert isinstance(rec["duration_ms"], (int, float))
    assert rec["duration_ms"] >= 0


def test_request_log_redacts_ingest_token() -> None:
    """The per-trigger ingest bearer token (a live credential in the URL path)
    is redacted on the api.request line, so it never persists in request logs.

    ``_FAKE_TOKEN`` is a dummy literal (no real ``aios_evt_`` prefix); the route
    has nothing after the token, so ``…/ingest/<anything>`` → ``…/ingest/<redacted>``.
    """
    fake_token = "tok-not-a-real-secret"
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/v1/triggers/ingest/{ingest_token}")
    async def _ingest(ingest_token: str) -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    with capture_logs() as entries:
        resp = client.post(f"/v1/triggers/ingest/{fake_token}", json={})
    assert resp.status_code == 200

    rec = _find_request_log(entries)
    assert rec is not None
    assert rec["path"] == "/v1/triggers/ingest/<redacted>"


def test_status_captured_for_404() -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    with capture_logs() as entries:
        resp = client.get("/nonexistent")
    assert resp.status_code == 404

    rec = _find_request_log(entries)
    assert rec is not None
    assert rec["status"] == 404
    assert rec["path"] == "/nonexistent"


async def test_non_http_scope_passthrough() -> None:
    """A non-http scope is delegated straight through; no api.request line."""
    delegated: list[MutableMapping[str, Any]] = []

    async def dummy_app(
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        delegated.append(scope)

    async def recv() -> MutableMapping[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(_message: MutableMapping[str, Any]) -> None:
        return None

    mw = RequestLoggingMiddleware(dummy_app)
    with capture_logs() as entries:
        await mw({"type": "lifespan"}, recv, send)

    assert delegated == [{"type": "lifespan"}]
    assert _find_request_log(entries) is None


def test_account_id_present_when_bound(caplog: pytest.LogCaptureFixture) -> None:
    """An account_id bound in a route dep is visible on the api.request line.

    Proves the same-task contextvar visibility that the pure-ASGI shape buys:
    ``bind_contextvars`` in the dep and ``merge_contextvars`` in the processor
    chain put account_id on the middleware's own log line.

    ``capture_logs`` can't see this — it swaps in a minimal processor chain
    without ``merge_contextvars`` — so we exercise the real chain via
    ``configure_logging`` and read the rendered JSON line off ``caplog``.
    """

    async def _bind_account() -> None:
        bind_contextvars(account_id="acc_x")

    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping", dependencies=[Depends(_bind_account)])
    async def _ping() -> dict[str, bool]:
        return {"ok": True}

    configure_logging("INFO")
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        resp = client.get("/ping")
    assert resp.status_code == 200

    payloads = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("{") and '"api.request"' in r.getMessage()
    ]
    assert payloads, "no api.request JSON log line captured"
    rec = payloads[0]
    assert rec["event"] == "api.request"
    assert rec["account_id"] == "acc_x"
