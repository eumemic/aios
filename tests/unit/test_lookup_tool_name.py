"""Unit tests for lookup_tool_name_by_call_id and submit_tool_result name injection."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.db.queries import (
    lookup_tool_name_by_call_id,
)

# ─── lookup_tool_name_by_call_id ─────────────────────────────────────────────


class TestLookupToolNameByCallId:
    """Unit tests for the lookup_tool_name_by_call_id query helper."""

    async def test_returns_name_when_found(self) -> None:
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value="get_weather")
        result = await lookup_tool_name_by_call_id(conn, "sess_01", "call_abc")
        assert result == "get_weather"

    async def test_returns_none_when_not_found(self) -> None:
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=None)
        result = await lookup_tool_name_by_call_id(conn, "sess_01", "call_missing")
        assert result is None

    async def test_passes_correct_params(self) -> None:
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value="bash")
        await lookup_tool_name_by_call_id(conn, "sess_XYZ", "call_123")
        conn.fetchval.assert_called_once()
        call_args = conn.fetchval.call_args
        # Second positional arg should be session_id, third should be tool_call_id
        assert call_args.args[1] == "sess_XYZ"
        assert call_args.args[2] == "call_123"

    async def test_query_targets_assistant_rows_with_tool_calls(self) -> None:
        """The SQL must use predicates matching the partial index."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=None)
        await lookup_tool_name_by_call_id(conn, "sess_01", "call_1")
        sql: str = conn.fetchval.call_args.args[0]
        assert "tool_calls" in sql
        assert "assistant" in sql
        assert "jsonb_array_elements" in sql
        assert "function" in sql


# ─── submit_tool_result name injection ───────────────────────────────────────


class TestSubmitToolResultNameInjection:
    """Tests that submit_tool_result injects 'name' into the event data
    when lookup_tool_name_by_call_id finds a matching tool call."""

    def _make_pool(self, conn: Any) -> Any:
        """Build a mock pool whose acquire() yields conn."""
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool.acquire.return_value = cm
        return pool

    async def test_name_injected_when_lookup_succeeds(self) -> None:
        """When lookup finds a name, data['name'] is present in append_event call."""
        from aios.api.routers.sessions import submit_tool_result
        from aios.models.sessions import ToolResultRequest

        conn = AsyncMock()
        pool = self._make_pool(conn)

        fake_event = MagicMock()

        with (
            patch(
                "aios.api.routers.sessions.db_queries.lookup_tool_name_by_call_id",
                new_callable=AsyncMock,
                return_value="get_weather",
            ) as mock_lookup,
            patch(
                "aios.api.routers.sessions.db_queries.append_event",
                new_callable=AsyncMock,
                return_value=fake_event,
            ) as mock_append,
            patch(
                "aios.api.routers.sessions.defer_wake",
                new_callable=AsyncMock,
            ),
        ):
            body = ToolResultRequest(
                tool_call_id="call_abc",
                content="72°F and sunny",
            )
            await submit_tool_result("sess_01", body, pool, _auth=None)

        mock_lookup.assert_called_once()
        # db_queries.append_event(conn, session_id=..., kind=..., data=...) — data is a kwarg
        call_args = mock_append.call_args
        data = call_args.kwargs["data"]
        assert data["name"] == "get_weather"

    async def test_raises_404_when_lookup_returns_none(self) -> None:
        """When lookup returns None, submit_tool_result must raise HTTP 404."""
        import pytest
        from fastapi import HTTPException

        from aios.api.routers.sessions import submit_tool_result
        from aios.models.sessions import ToolResultRequest

        conn = AsyncMock()
        pool = self._make_pool(conn)

        with (
            patch(
                "aios.api.routers.sessions.db_queries.lookup_tool_name_by_call_id",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            body = ToolResultRequest(
                tool_call_id="call_missing",
                content="result",
            )
            with pytest.raises(HTTPException) as exc_info:
                await submit_tool_result("sess_01", body, pool, _auth=None)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "tool_call_id not found"

    async def test_is_error_still_injected(self) -> None:
        """is_error flag survives alongside name injection."""
        from aios.api.routers.sessions import submit_tool_result
        from aios.models.sessions import ToolResultRequest

        conn = AsyncMock()
        pool = self._make_pool(conn)

        fake_event = MagicMock()

        with (
            patch(
                "aios.api.routers.sessions.db_queries.lookup_tool_name_by_call_id",
                new_callable=AsyncMock,
                return_value="bash",
            ),
            patch(
                "aios.api.routers.sessions.db_queries.append_event",
                new_callable=AsyncMock,
                return_value=fake_event,
            ) as mock_append,
            patch(
                "aios.api.routers.sessions.defer_wake",
                new_callable=AsyncMock,
            ),
        ):
            body = ToolResultRequest(
                tool_call_id="call_err",
                content='{"error": "nope"}',
                is_error=True,
            )
            await submit_tool_result("sess_01", body, pool, _auth=None)

        call_args = mock_append.call_args
        data = call_args.kwargs["data"]
        assert data["name"] == "bash"
        assert data["is_error"] is True
