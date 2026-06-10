"""Unit tests that the three exception handlers emit structured log lines.

Before this change the handlers returned JSON and logged nothing, so a 500
(or a silent 401) left no trace. Now:

* ``AiosError`` < 500 → ``api.error`` at WARNING; >= 500 → at ERROR with
  ``exc_info`` (``log.exception``).
* ``StarletteHTTPException`` → ``api.http_error`` (warn < 500 / exception >= 500).
* request validation error → ``api.validation_error`` WARNING, status 422.
* ``account_id`` (bound via the auth dep contextvar) appears on the line.

structlog renders the event dict to a single message string via the configured
chain, so we read the JSON line off ``caplog`` rather than expecting kv
attributes on the record.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog.contextvars import bind_contextvars, clear_contextvars

from aios.errors import (
    CryptoDecryptError,
    NotFoundError,
    install_exception_handlers,
)
from aios.logging import configure_logging


@pytest.fixture(autouse=True)
def _logging_and_contextvars() -> object:
    configure_logging("INFO")
    clear_contextvars()
    yield
    clear_contextvars()


def _find(
    caplog: pytest.LogCaptureFixture, event: str
) -> tuple[dict[str, object], logging.LogRecord]:
    for r in caplog.records:
        msg = r.getMessage()
        if not msg.startswith("{"):
            continue
        try:
            payload = json.loads(msg)
        except ValueError:
            continue
        if payload.get("event") == event:
            return payload, r
    raise AssertionError(f"no {event!r} log line captured")


def test_aios_error_4xx_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    async def _boom() -> None:
        raise NotFoundError("nope")

    with caplog.at_level(logging.WARNING):
        TestClient(app, raise_server_exceptions=False).get("/boom")

    payload, record = _find(caplog, "api.error")
    assert payload["status"] == 404
    assert payload["path"] == "/boom"
    assert record.levelno == logging.WARNING


def test_aios_error_5xx_logs_exception(caplog: pytest.LogCaptureFixture) -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    async def _boom() -> None:
        raise CryptoDecryptError("decrypt failed")

    with caplog.at_level(logging.INFO):
        TestClient(app, raise_server_exceptions=False).get("/boom")

    payload, record = _find(caplog, "api.error")
    assert payload["status"] == 500
    assert payload["path"] == "/boom"
    assert record.levelno == logging.ERROR
    # log.exception → structlog's format_exc_info renders the traceback into
    # the line (an ``exception`` key), which is the captured-traceback signal.
    assert "exception" in payload
    assert "CryptoDecryptError" in str(payload["exception"])


def test_http_exception_logs_http_error(caplog: pytest.LogCaptureFixture) -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/forbidden")
    async def _forbidden() -> None:
        raise StarletteHTTPException(status_code=403, detail="no")

    with caplog.at_level(logging.WARNING):
        TestClient(app, raise_server_exceptions=False).get("/forbidden")

    payload, record = _find(caplog, "api.http_error")
    assert payload["status"] == 403
    assert payload["path"] == "/forbidden"
    assert record.levelno == logging.WARNING


class _Model(BaseModel):
    x: str

    @field_validator("x")
    @classmethod
    def _v(cls, v: str) -> str:
        if v == "bad":
            raise ValueError("x must not be 'bad'")
        return v


def test_validation_error_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/echo")
    async def _echo(body: _Model) -> dict[str, str]:
        return {"ok": "true"}

    with caplog.at_level(logging.WARNING):
        TestClient(app, raise_server_exceptions=False).post("/echo", json={"x": "bad"})

    payload, record = _find(caplog, "api.validation_error")
    assert payload["status"] == 422
    assert payload["path"] == "/echo"
    assert record.levelno == logging.WARNING


def test_account_id_present_on_error_line(caplog: pytest.LogCaptureFixture) -> None:
    """When the auth dep has bound account_id, the error line carries it."""
    app = FastAPI()
    install_exception_handlers(app)

    async def _bind() -> None:
        bind_contextvars(account_id="acc_y")

    from fastapi import Depends

    @app.get("/boom", dependencies=[Depends(_bind)])
    async def _boom() -> None:
        raise NotFoundError("nope")

    with caplog.at_level(logging.WARNING):
        TestClient(app, raise_server_exceptions=False).get("/boom")

    payload, _ = _find(caplog, "api.error")
    assert payload["account_id"] == "acc_y"
