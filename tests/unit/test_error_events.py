"""Unit tests for error event emission and ?error_only filter (issue #372).

Covers:
- ``_derive_is_error`` correctly extracts is_error for any kind (not just message).
- ``_handle_step_timeout`` emits a span with ``is_error: True``.
- Unhandled exceptions in ``run_session_step`` emit a ``harness_error`` span.
- ``read_events`` ``error_only`` flag adds the correct SQL filter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db.queries import _derive_is_error


class TestDeriveIsError:
    def test_message_with_is_error_true_returns_true(self) -> None:
        assert _derive_is_error("message", {"is_error": True}) is True

    def test_message_without_is_error_returns_none(self) -> None:
        assert _derive_is_error("message", {"role": "user", "content": "hi"}) is None

    def test_span_with_is_error_true_returns_true(self) -> None:
        # KEY regression test: span events should have is_error written to the
        # physical column. Previously guarded by `if kind != "message": return None`.
        assert _derive_is_error("span", {"event": "model_request_end", "is_error": True}) is True

    def test_span_with_is_error_false_returns_false(self) -> None:
        assert _derive_is_error("span", {"event": "model_request_end", "is_error": False}) is False

    def test_span_without_is_error_returns_none(self) -> None:
        assert _derive_is_error("span", {"event": "step_start"}) is None

    def test_lifecycle_without_is_error_returns_none(self) -> None:
        assert _derive_is_error("lifecycle", {"event": "turn_ended"}) is None


class TestStepTimeoutSpanHasIsError:
    async def test_step_timeout_span_has_is_error_true(self) -> None:
        from aios.harness.loop import _handle_step_timeout

        mock_append = AsyncMock()
        mock_retry = AsyncMock(return_value=2.0)
        pool = MagicMock()

        with (
            patch("aios.harness.loop.sessions_service.append_event", mock_append),
            patch("aios.harness.loop._apply_retry_or_failure", mock_retry),
        ):
            await _handle_step_timeout(pool=pool, session_id="sess_x")

        # Find the span call (there may be others from _apply_retry_or_failure path)
        span_calls = [
            call
            for call in mock_append.await_args_list
            if len(call.args) >= 4 and call.args[2] == "span"
        ]
        assert len(span_calls) >= 1
        step_timeout_calls = [
            call
            for call in span_calls
            if isinstance(call.args[3], dict) and call.args[3].get("event") == "step_timeout"
        ]
        assert len(step_timeout_calls) == 1
        data = step_timeout_calls[0].args[3]
        assert data.get("is_error") is True


class TestHarnessErrorSpan:
    async def test_unhandled_exception_emits_harness_error_span(self) -> None:
        from aios.harness.loop import run_session_step

        mock_append = AsyncMock(return_value=SimpleNamespace(id="ev_start"))
        mock_retry = AsyncMock(return_value=2.0)
        mock_task_registry = MagicMock()
        mock_task_registry.register_step = MagicMock()
        mock_task_registry.unregister_step = MagicMock()

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=mock_task_registry,
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                mock_append,
            ),
            patch(
                "aios.harness.loop._apply_retry_or_failure",
                mock_retry,
            ),
            patch(
                "aios.harness.loop._run_session_step_body",
                AsyncMock(side_effect=ValueError("boom")),
            ),
            patch("aios.harness.loop.defer_wake", AsyncMock()),
        ):
            await run_session_step("sess_x")

        # Find the harness_error span
        harness_error_calls = [
            call
            for call in mock_append.await_args_list
            if len(call.args) >= 4
            and call.args[2] == "span"
            and isinstance(call.args[3], dict)
            and call.args[3].get("event") == "harness_error"
        ]
        assert len(harness_error_calls) == 1
        data = harness_error_calls[0].args[3]
        assert data.get("is_error") is True

    async def test_unhandled_exception_budget_exhausted_raises(self) -> None:
        from aios.harness.loop import run_session_step

        mock_append = AsyncMock(return_value=SimpleNamespace(id="ev_start"))
        mock_retry = AsyncMock(return_value=None)  # budget exhausted
        mock_task_registry = MagicMock()
        mock_task_registry.register_step = MagicMock()
        mock_task_registry.unregister_step = MagicMock()

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=mock_task_registry,
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                mock_append,
            ),
            patch(
                "aios.harness.loop._apply_retry_or_failure",
                mock_retry,
            ),
            patch(
                "aios.harness.loop._run_session_step_body",
                AsyncMock(side_effect=ValueError("boom")),
            ),
            pytest.raises(ValueError, match="boom"),
        ):
            await run_session_step("sess_x")

        # harness_error span must still be emitted even when budget is exhausted
        harness_error_calls = [
            call
            for call in mock_append.await_args_list
            if len(call.args) >= 4
            and call.args[2] == "span"
            and isinstance(call.args[3], dict)
            and call.args[3].get("event") == "harness_error"
        ]
        assert len(harness_error_calls) == 1
        data = harness_error_calls[0].args[3]
        assert data.get("is_error") is True


class TestReadEventsErrorOnlyFilter:
    async def test_error_only_true_adds_is_error_filter(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", error_only=True)

        query = conn.fetch.call_args[0][0]
        assert "is_error IS TRUE" in query

    async def test_error_only_false_omits_is_error_filter(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", error_only=False)

        query = conn.fetch.call_args[0][0]
        assert "is_error IS TRUE" not in query

    async def test_error_only_combined_with_kind_filter(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", kind="span", error_only=True)

        query = conn.fetch.call_args[0][0]
        assert "kind" in query
        assert "is_error IS TRUE" in query
