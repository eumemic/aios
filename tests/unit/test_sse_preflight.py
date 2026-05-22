"""Unit tests for SSE preflight error handling (issue #376).

The four SSE generators in ``aios.api.sse`` used to do their own
``asyncpg.connect`` + ``add_listener`` setup INSIDE the
``EventSourceResponse`` body.  Any failure there happened AFTER 200 OK
headers had already been written, so the client saw a half-open
chunked stream and a noisy "ASGI callable returned without completing
response" error on the server.

The fix moves the ``open_listen_for_*`` call into the route handler,
before constructing ``EventSourceResponse``.  Preflight failure now
surfaces as a clean 503 with an aios error envelope; headers are not
written until the LISTEN connection is established.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import (
    get_account_id,
    get_db_url,
    get_pool,
    require_runtime_auth,
)
from aios.api.routers import connectors as connectors_router
from aios.api.routers import sessions as sessions_router
from aios.errors import install_exception_handlers


def _build_app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(connectors_router.router)
    app.include_router(sessions_router.router)
    app.state.pool = MagicMock()
    app.state.db_url = "postgresql://stub/aios"
    app.state.crypto_box = MagicMock()
    app.state.procrastinate = MagicMock()

    async def _fake_pool() -> Any:
        return MagicMock()

    async def _fake_db_url() -> str:
        return "postgresql://stub/aios"

    async def _fake_account_id() -> str:
        return "acct_test"

    async def _fake_runtime_auth() -> tuple[str, str, str, list[str] | None]:
        return ("tok_test", "telegram", "acct_test", None)

    app.dependency_overrides[get_pool] = _fake_pool
    app.dependency_overrides[get_db_url] = _fake_db_url
    app.dependency_overrides[get_account_id] = _fake_account_id
    app.dependency_overrides[require_runtime_auth] = _fake_runtime_auth
    return app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = _build_app()
    with TestClient(app) as c:
        yield c


def _assert_sse_preflight_503(response: Any, *, stream: str) -> None:
    """Both the envelope error_type and the detail.stream must be set."""
    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["type"] == "sse_preflight_failed"
    assert body["error"]["detail"]["stream"] == stream


def test_runtime_calls_503_when_open_listen_raises(client: TestClient) -> None:
    with patch(
        "aios.api.routers.connectors.open_listen_for_connector_calls_by_type",
        AsyncMock(side_effect=asyncpg.CannotConnectNowError("startup")),
    ):
        response = client.get(
            "/v1/connectors/runtime/calls",
            headers={"Authorization": "Bearer fake"},
        )
    _assert_sse_preflight_503(response, stream="runtime_calls")


def test_management_calls_503_when_open_listen_raises(client: TestClient) -> None:
    with patch(
        "aios.api.routers.connectors.open_listen_for_management_calls",
        AsyncMock(side_effect=asyncpg.CannotConnectNowError("startup")),
    ):
        response = client.get(
            "/v1/connectors/runtime/management-calls",
            headers={"Authorization": "Bearer fake"},
        )
    _assert_sse_preflight_503(response, stream="management_calls")


def test_connection_discovery_503_when_open_listen_raises(client: TestClient) -> None:
    with patch(
        "aios.api.routers.connectors.open_listen_for_connection_discovery",
        AsyncMock(side_effect=asyncpg.CannotConnectNowError("startup")),
    ):
        response = client.get(
            "/v1/connectors/connections",
            headers={"Authorization": "Bearer fake"},
        )
    _assert_sse_preflight_503(response, stream="connection_discovery")


def test_session_stream_503_when_open_listen_raises(client: TestClient) -> None:
    """The session-existence check must pass before the preflight runs;
    stub it so the preflight gets a chance to raise."""
    with (
        patch(
            "aios.api.routers.sessions.service.get_session_basic",
            AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "aios.api.routers.sessions.open_listen_for_events",
            AsyncMock(side_effect=asyncpg.CannotConnectNowError("startup")),
        ),
    ):
        response = client.get(
            "/v1/sessions/sess_abc/stream",
            headers={"Authorization": "Bearer fake"},
        )
    _assert_sse_preflight_503(response, stream="session_events")


def test_runtime_calls_503_logged_with_diagnostic(client: TestClient) -> None:
    """Preflight failure emits a structured warning with connector + error fields.

    We patch the module-level :data:`aios.api.routers.connectors.log` object
    directly rather than asserting against ``caplog``/``capsys``.  structlog
    is configured with ``cache_logger_on_first_use=True``, so the bound
    logger captured during the first import freezes whatever level/processors
    were active then — making suite-wide log-capture assertions
    order-dependent.  Asserting on the call to ``log.warning`` itself targets
    the actual diagnostic contract (call site + kwargs) and is stable across
    the suite.
    """
    with (
        patch(
            "aios.api.routers.connectors.open_listen_for_connector_calls_by_type",
            AsyncMock(side_effect=asyncpg.CannotConnectNowError("startup blip")),
        ),
        patch("aios.api.routers.connectors.log") as log_mock,
    ):
        response = client.get(
            "/v1/connectors/runtime/calls",
            headers={"Authorization": "Bearer fake"},
        )
    assert response.status_code == 503
    log_mock.warning.assert_called_once()
    call = log_mock.warning.call_args
    assert call.args == ("sse.runtime_calls.preflight_failed",)
    assert call.kwargs["connector"] == "telegram"
    assert call.kwargs["error"] == "startup blip"
    assert call.kwargs["error_type"] == "CannotConnectNowError"
