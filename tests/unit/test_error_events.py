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
            await _handle_step_timeout(pool=pool, session_id="sess_x", account_id="acc_test_stub")

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
        mock_inflight_tool_registry = MagicMock()
        mock_inflight_tool_registry.register_step = MagicMock()
        mock_inflight_tool_registry.unregister_step = MagicMock()

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=mock_inflight_tool_registry,
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
        mock_inflight_tool_registry = MagicMock()
        mock_inflight_tool_registry.register_step = MagicMock()
        mock_inflight_tool_registry.unregister_step = MagicMock()

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=mock_inflight_tool_registry,
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

        await read_events(conn, "sess_x", error_only=True, account_id="acc_test_stub")

        query = conn.fetch.call_args[0][0]
        assert "is_error IS TRUE" in query

    async def test_error_only_false_omits_is_error_filter(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", error_only=False, account_id="acc_test_stub")

        query = conn.fetch.call_args[0][0]
        assert "is_error IS TRUE" not in query

    async def test_error_only_combined_with_kind_filter(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", kind="span", error_only=True, account_id="acc_test_stub")

        query = conn.fetch.call_args[0][0]
        assert "kind" in query
        assert "is_error IS TRUE" in query


class TestReadEventsCursorPlaceholders:
    """Pin the ``$N`` placeholder/param alignment in ``read_events``.

    The WHERE clause is built by appending to ``params`` and numbering each
    placeholder from ``len(params)``, so a future reorder that desynced them
    would silently bind the wrong value to the wrong column. The e2e suite
    proves this against real Postgres; these are the fast unit-tier guards.
    """

    async def test_forward_after_seq_with_kind_aligns_placeholders(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", after_seq=10, kind="message", limit=50, account_id="acc")

        query, *params = conn.fetch.call_args[0]
        assert params == ["sess_x", "acc", 10, "message", 50]
        assert "seq > $3" in query and "kind = $4" in query
        assert "ORDER BY seq ASC" in query and "LIMIT $5" in query

    async def test_backward_before_with_kind_aligns_placeholders(self) -> None:
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", before=20, kind="message", limit=50, account_id="acc")

        query, *params = conn.fetch.call_args[0]
        assert params == ["sess_x", "acc", 20, "message", 50]
        assert "seq < $3" in query and "kind = $4" in query
        assert "ORDER BY seq DESC" in query and "LIMIT $5" in query

    async def test_after_seq_zero_omits_lower_bound_clause(self) -> None:
        # after_seq=0 is the "no lower bound" default: the seq clause is skipped
        # entirely (equivalent to ``seq > 0`` since seq is gapless from 1).
        from aios.db.queries import read_events

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[])

        await read_events(conn, "sess_x", after_seq=0, limit=50, account_id="acc")

        query, *params = conn.fetch.call_args[0]
        assert params == ["sess_x", "acc", 50]
        assert "seq >" not in query
        assert "LIMIT $3" in query
