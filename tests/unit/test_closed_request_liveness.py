"""Unit tests for the #1773 defect-1 liveness-first fix: `return`/`error` against
an ALREADY-CLOSED request must short-circuit with one clear terminal stop message
— BEFORE schema validation ever runs — instead of leaving the child bouncing
forever on an ``output_schema_violation`` that never mentions the request is dead.

DB-backed behavior (``queries.get_closed_request`` itself) is covered in
tests/integration; these pin the two pure/near-pure pieces: the message
wording and the handler-level short-circuit ordering, with the pool/queries
calls mocked out (matching the pattern in test_unify_obligations_return_close.py
and test_workflow_output_schema.py).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

from aios.db import queries
from aios.harness import runtime
from aios.models.sessions import Err, Ok
from aios.tools import workflow_completion
from aios.tools.registry import ToolResult
from aios.tools.workflow_completion import (
    _closed_request_error,
    _closed_request_message,
    error_handler,
    return_handler,
)

_CLOSED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class TestClosedRequestMessage:
    def test_close_without_event_detail_keeps_terminal_framing(self) -> None:
        msg = _closed_request_message()
        assert msg == (
            "this request was already answered; do not call return again — end your turn."
        )

    def test_timeout_close_names_deadline_timeout(self) -> None:
        msg = _closed_request_message(Err(error={"kind": "timeout"}), _CLOSED_AT)
        assert "deadline timeout" in msg
        assert _CLOSED_AT.isoformat() in msg
        assert "do not call return again" in msg

    def test_non_timeout_close_is_truthful_not_a_fake_timeout(self) -> None:
        # A duplicate self-answer (Ok close) must NOT claim "deadline timeout" —
        # that would be false; it just needs to be equally terminal.
        msg = _closed_request_message(Ok(result="already answered"), _CLOSED_AT)
        assert "deadline timeout" not in msg
        assert "already answered" in msg
        assert "do not call return again" in msg

    def test_no_return_close_is_also_worded_terminal(self) -> None:
        msg = _closed_request_message(Err(error={"kind": "no_return"}), _CLOSED_AT)
        assert "deadline timeout" not in msg
        assert "do not call return again" in msg


class TestClosedRequestErrorShortCircuit:
    async def test_returns_none_when_request_still_open(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(queries, "get_closed_request", AsyncMock(return_value=None))
        pool = mock.MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock.MagicMock())
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(runtime, "require_pool", lambda: pool)

        result = await _closed_request_error("ses_1", "req_1")
        assert result is None

    async def test_returns_none_for_non_string_request_id(self, monkeypatch: Any) -> None:
        # No request_id (or a malformed one) is the existing unknown_request path's
        # job downstream — this short-circuit must not swallow it.
        get_closed = AsyncMock()
        monkeypatch.setattr(queries, "get_closed_request", get_closed)
        result = await _closed_request_error("ses_1", None)
        assert result is None
        get_closed.assert_not_called()

    async def test_returns_terminal_error_when_closed(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            queries,
            "get_closed_request",
            AsyncMock(return_value=(Err(error={"kind": "timeout"}), _CLOSED_AT)),
        )
        pool = mock.MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock.MagicMock())
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(runtime, "require_pool", lambda: pool)

        result = await _closed_request_error("ses_1", "req_1")
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "deadline timeout" in result.content


class TestHandlerLivenessFirst:
    """The handlers must check liveness BEFORE schema validation — a schema-invalid
    value against a CLOSED request gets the closed-request message, not a schema
    bounce."""

    async def test_return_handler_short_circuits_before_schema_check(
        self, monkeypatch: Any
    ) -> None:
        closed_result = ToolResult(content="this request was already answered", is_error=True)
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=closed_result)
        )
        enforce_schema = AsyncMock(return_value="should never be reached")
        monkeypatch.setattr(workflow_completion, "_enforce_output_schema", enforce_schema)
        finish = AsyncMock()
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await return_handler("ses_1", {"request_id": "req_1", "value": 123})

        assert result is closed_result
        enforce_schema.assert_not_called()
        finish.assert_not_called()

    async def test_return_handler_proceeds_when_request_open(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            workflow_completion, "_enforce_output_schema", AsyncMock(return_value=None)
        )
        finish = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await return_handler("ses_1", {"request_id": "req_1", "value": {"answer": "x"}})

        assert result == {"ok": True}
        finish.assert_awaited_once()
        assert finish.await_args is not None
        assert finish.await_args.kwargs["request_id"] == "req_1"
        assert finish.await_args.kwargs["outcome"] == Ok(result={"answer": "x"})

    async def test_error_handler_short_circuits_on_closed_request(self, monkeypatch: Any) -> None:
        closed_result = ToolResult(content="this request was already answered", is_error=True)
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=closed_result)
        )
        finish = AsyncMock()
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await error_handler("ses_1", {"request_id": "req_1", "message": "nope"})

        assert result is closed_result
        finish.assert_not_called()

    async def test_error_handler_proceeds_when_request_open(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=None)
        )
        finish = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await error_handler("ses_1", {"request_id": "req_1", "message": "nope"})

        assert result == {"ok": True}
        finish.assert_awaited_once()
        assert finish.await_args is not None
        assert finish.await_args.kwargs["outcome"] == Err(error={"message": "nope"})
