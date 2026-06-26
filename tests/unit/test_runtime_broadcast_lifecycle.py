"""Unit tests for ``POST /v1/connectors/runtime/lifecycle`` (#1538).

The broadcast lifecycle fan-out route. It appends a ``kind=lifecycle`` event
onto *every* session bound to a connection. Unlike the single-target session-/
chat-lifecycle siblings (which let ``append_event`` surface uncaught), the
broadcast route catches the **one** benign per-session failure — a session
archived between the binding snapshot and its append, surfaced by
``append_event`` as a typed ``NotFoundError`` — and reports it under
``skipped_session_ids``. Every *other* append failure (serialization, statement
timeout, pool exhaustion, a broken connection) is a real writer fault and must
propagate uncaught (→ 500), never be buried in a 201 body.

Driven by calling the router handler ``post_runtime_lifecycle`` directly: the
runtime-auth dependency resolves to a plain ``(token_id, connector, account_id,
connection_ids)`` tuple, so the handler is callable without an HTTP layer.
``queries.get_connection`` / ``queries.list_session_ids_for_connection`` are
patched at the router import site (returning two bound session ids), and
``sessions_service.append_event`` is patched to script per-session outcomes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from aios.api.routers.connectors import (
    RuntimeLifecycleRequest,
    post_runtime_lifecycle,
)
from aios.errors import NotFoundError

_CONNECTOR = "whatsapp"
_ACCOUNT = "acc_bc"
_CONNECTION = "conn_bc"


def _auth() -> tuple[str, str, str, None]:
    """Shape of a resolved unscoped ``RuntimeAuthDep`` tuple:
    ``(token_id, connector, account_id, connection_ids)``."""
    return ("rt_test", _CONNECTOR, _ACCOUNT, None)


class _FakePool:
    """Minimal stand-in for an asyncpg pool: ``async with pool.acquire()``
    yields a sentinel connection (the patched queries ignore it)."""

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield object()


@pytest.fixture
def patched_queries() -> Any:
    """Patch the router's ``queries`` import site so ``get_connection`` returns
    a connection on ``_CONNECTOR`` and ``list_session_ids_for_connection``
    returns two bound session ids (``A``, ``B``)."""
    connection = mock.Mock()
    connection.connector = _CONNECTOR
    with (
        mock.patch(
            "aios.api.routers.connectors.queries.get_connection",
            new=AsyncMock(return_value=connection),
        ),
        mock.patch(
            "aios.api.routers.connectors.queries.list_session_ids_for_connection",
            new=AsyncMock(return_value=["A", "B"]),
        ),
    ):
        yield


def _body() -> RuntimeLifecycleRequest:
    return RuntimeLifecycleRequest(
        connection_id=_CONNECTION,
        event="whatsapp.connection.lost",
        reason="daemon_crashed",
    )


async def test_broadcast_skips_archived_session(patched_queries: None) -> None:
    """Session B archived mid-broadcast (``NotFoundError``) is skipped + reported;
    session A is appended. No exception escapes."""

    async def _append(pool: Any, sess_id: str, *args: Any, **kwargs: Any) -> None:
        if sess_id == "B":
            raise NotFoundError("session archived")

    with mock.patch(
        "aios.api.routers.connectors.sessions_service.append_event",
        new=AsyncMock(side_effect=_append),
    ):
        result = await post_runtime_lifecycle(_body(), _FakePool(), _auth())

    assert result == {"appended_session_ids": ["A"], "skipped_session_ids": ["B"]}


async def test_broadcast_raises_on_writer_failure(patched_queries: None) -> None:
    """A non-``NotFoundError`` append fault (statement timeout) propagates
    uncaught — fail hard — rather than being folded into a 201 body."""

    async def _append(pool: Any, sess_id: str, *args: Any, **kwargs: Any) -> None:
        if sess_id == "B":
            raise RuntimeError("statement timeout")

    with (
        mock.patch(
            "aios.api.routers.connectors.sessions_service.append_event",
            new=AsyncMock(side_effect=_append),
        ),
        pytest.raises(RuntimeError, match="statement timeout"),
    ):
        await post_runtime_lifecycle(_body(), _FakePool(), _auth())


async def test_broadcast_all_succeed(patched_queries: None) -> None:
    """Both sessions append cleanly: result carries only ``appended_session_ids``,
    with no ``skipped_session_ids`` (and certainly no ``failed_session_ids``)."""
    with mock.patch(
        "aios.api.routers.connectors.sessions_service.append_event",
        new=AsyncMock(return_value=None),
    ):
        result = await post_runtime_lifecycle(_body(), _FakePool(), _auth())

    assert result == {"appended_session_ids": ["A", "B"]}
    assert "skipped_session_ids" not in result
    assert "failed_session_ids" not in result
