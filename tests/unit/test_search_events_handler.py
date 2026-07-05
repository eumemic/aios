"""Unit tests for the search_events tool handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.tools.invoke import ToolBail
from aios.tools.search_events import (
    MAX_ROWS,
    _format_results,
    _validate_sql,
    search_events_handler,
)

# ─── SQL Validation ──────────────────────────────────────────────────────────


class TestValidateSql:
    """Tests for _validate_sql pure function."""

    def test_valid_select(self) -> None:
        assert _validate_sql("SELECT * FROM events_search") is None

    def test_valid_select_count(self) -> None:
        assert _validate_sql("SELECT COUNT(*) FROM events_search") is None

    def test_valid_select_with_where(self) -> None:
        assert _validate_sql("SELECT id FROM events_search WHERE role = 'user' LIMIT 10") is None

    def test_valid_select_with_subquery(self) -> None:
        assert _validate_sql("SELECT * FROM (SELECT id FROM events_search) sub") is None

    def test_valid_select_ilike(self) -> None:
        assert (
            _validate_sql("SELECT * FROM events_search WHERE content_text ILIKE '%hello%'") is None
        )

    def test_valid_select_case_insensitive_keyword(self) -> None:
        assert _validate_sql("select * from events_search") is None

    def test_rejects_insert(self) -> None:
        err = _validate_sql("INSERT INTO foo VALUES (1)")
        assert err is not None
        assert "SELECT" in err

    def test_rejects_update(self) -> None:
        err = _validate_sql("UPDATE foo SET x = 1")
        assert err is not None

    def test_rejects_delete(self) -> None:
        err = _validate_sql("DELETE FROM foo")
        assert err is not None

    def test_rejects_drop_in_subquery(self) -> None:
        err = _validate_sql("SELECT * FROM events_search WHERE id IN (DROP TABLE events)")
        assert err is not None

    def test_rejects_create(self) -> None:
        err = _validate_sql("SELECT * FROM events_search; CREATE TABLE evil (id int)")
        assert err is not None
        # Caught by semicolon check first
        assert "semicolon" in err.lower() or "multiple" in err.lower()

    def test_rejects_truncate(self) -> None:
        err = _validate_sql("SELECT * FROM events_search WHERE TRUNCATE = 1")
        assert err is not None
        assert "TRUNCATE" in err

    def test_rejects_semicolon(self) -> None:
        err = _validate_sql("SELECT 1; DROP TABLE events")
        assert err is not None
        assert "semicolon" in err.lower() or "multiple" in err.lower()

    def test_rejects_semicolon_case_insensitive(self) -> None:
        err = _validate_sql("select * from events_search; delete from x")
        assert err is not None

    def test_rejects_non_select_start(self) -> None:
        err = _validate_sql("WITH cte AS (DELETE FROM events) SELECT 1")
        assert err is not None
        assert "SELECT" in err

    def test_rejects_direct_events_table_query(self) -> None:
        """The agent's SQL surface is the per-session ``events_search`` view.

        Allowing direct access to the ``events`` table bypasses the view's
        ``session_id = current_setting('app.session_id', true)`` filter and
        returns events from any session in the database (cross-tenant leak,
        since the table holds rows for every account).
        """
        err = _validate_sql("SELECT data FROM events")
        assert err is not None

    def test_rejects_query_against_other_table(self) -> None:
        """Tables other than ``events_search`` are out of scope for the agent.

        A read-only SELECT against ``vault_credentials`` / ``accounts`` /
        ``sessions`` / etc. still leaks cross-tenant rows; the view scoping
        only works when the agent is forced to use the view.
        """
        for table in ("vault_credentials", "accounts", "sessions", "connections"):
            err = _validate_sql(f"SELECT * FROM {table}")
            assert err is not None, f"validator allowed access to {table!r}"

    def test_rejects_events_table_via_cte(self) -> None:
        """CTEs that reference ``events`` directly are the same defect class."""
        err = _validate_sql(
            "WITH peek AS (SELECT data FROM events WHERE session_id = 'sess_OTHER') "
            "SELECT * FROM peek"
        )
        assert err is not None

    def test_rejects_events_via_quoted_identifier(self) -> None:
        """Postgres double-quoted identifiers must not bypass the check."""
        err = _validate_sql("SELECT * FROM \"events\" WHERE session_id = 'sess_OTHER'")
        assert err is not None

    def test_rejects_pg_catalog_introspection(self) -> None:
        """Postgres catalog tables enumerate the whole schema and must be blocked."""
        for q in (
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT * FROM pg_class",
            "SELECT * FROM information_schema.tables",
        ):
            err = _validate_sql(q)
            assert err is not None, f"validator allowed {q!r}"

    def test_allows_ilike_substring_with_table_name(self) -> None:
        """ILIKE patterns containing a forbidden token must not false-positive.

        ``content_text ILIKE '%events%'`` is a legitimate substring search
        on the agent's own messages; the literal-stripping pass strips
        the quoted string so the structural regex doesn't see it.
        """
        assert (
            _validate_sql("SELECT * FROM events_search WHERE content_text ILIKE '%events%'") is None
        )
        assert (
            _validate_sql("SELECT * FROM events_search WHERE content_text ILIKE '%sessions%'")
            is None
        )

    def test_rejects_events_via_dollar_quoted_evasion(self) -> None:
        """Dollar-quoted strings (Postgres-specific) must be stripped before the check.

        Without dollar-quote stripping, a model could write
        ``'%' || $$evil events$$ || '%'`` to confuse a naïve scanner; we want
        to make sure the dollar-quoted body doesn't seed false positives
        (i.e. that legitimate dollar-quoted strings containing forbidden
        words still validate cleanly).
        """
        # Dollar-quoted string with forbidden word inside the literal — legit.
        assert (
            _validate_sql("SELECT * FROM events_search WHERE content_text ILIKE $$%events table%$$")
            is None
        )
        # Dollar-quoted around a real FROM events bypass attempt — must reject.
        err = _validate_sql("SELECT * FROM events_search UNION SELECT * FROM events")
        assert err is not None

    def test_rejects_empty_string(self) -> None:
        err = _validate_sql("")
        assert err is not None

    def test_rejects_whitespace_only(self) -> None:
        err = _validate_sql("   ")
        assert err is not None

    def test_allows_leading_whitespace(self) -> None:
        assert _validate_sql("  SELECT 1") is None


# ─── Format Results ──────────────────────────────────────────────────────────


class _FakeRecord:
    """Minimal stand-in for asyncpg.Record for unit tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def values(self) -> Any:
        return self._data.values()

    def keys(self) -> Any:
        return self._data.keys()


class TestFormatResults:
    """Tests for _format_results pure function."""

    def test_empty_rows(self) -> None:
        result = _format_results([], False)
        assert result == "No results."

    def test_single_row(self) -> None:
        rows = [_FakeRecord({"id": "evt_01", "role": "assistant", "content_text": "Hello"})]
        result = _format_results(rows, False)
        assert "id" in result
        assert "role" in result
        assert "assistant" in result
        assert "Hello" in result

    def test_multiple_rows(self) -> None:
        rows = [
            _FakeRecord({"seq": 1, "role": "user"}),
            _FakeRecord({"seq": 2, "role": "assistant"}),
        ]
        result = _format_results(rows, False)
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + divider + 2 data rows

    def test_truncated_notice(self) -> None:
        rows = [_FakeRecord({"id": "evt_01"})]
        result = _format_results(rows, truncated=True)
        assert "truncat" in result.lower()
        assert str(MAX_ROWS) in result

    def test_not_truncated_no_notice(self) -> None:
        rows = [_FakeRecord({"id": "evt_01"})]
        result = _format_results(rows, truncated=False)
        assert "truncat" not in result.lower()

    def test_null_values(self) -> None:
        rows = [_FakeRecord({"id": "evt_01", "role": None, "content_text": "text"})]
        result = _format_results(rows, False)
        assert "NULL" in result

    def test_header_divider_format(self) -> None:
        rows = [_FakeRecord({"a": 1, "b": 2})]
        result = _format_results(rows, False)
        lines = result.split("\n")
        assert lines[0] == "a | b"
        assert lines[1] == "-" * len("a | b")


# ─── Handler Behavior ────────────────────────────────────────────────────────


def _mock_execute(
    return_value: Any = ([], False),
    side_effect: Any = None,
) -> Any:
    """Context manager that mocks both _execute_query and runtime.require_pool."""
    return patch(
        "aios.tools.search_events._execute_query",
        new_callable=AsyncMock,
        return_value=return_value,
        side_effect=side_effect,
    )


def _mock_pool() -> Any:
    return patch("aios.tools.search_events.runtime.require_pool")


class TestSearchEventsHandler:
    """Tests for the handler with mocked _execute_query."""

    async def test_success_returns_formatted_results(self) -> None:
        rows = [_FakeRecord({"role": "assistant", "content_text": "Hello"})]
        with _mock_execute(return_value=(rows, False)), _mock_pool():
            result = await search_events_handler(
                "sess_01TEST", {"query": "SELECT * FROM events_search"}
            )
        assert "result" in result
        assert "assistant" in result["result"]
        assert "Hello" in result["result"]

    async def test_sql_validation_error(self) -> None:
        # Post-#1680: an expected failure raises ``ToolBail`` (one typed failure
        # channel) rather than returning a bare ``{"error": ...}`` dict.
        with _mock_execute() as mock_exec:
            with pytest.raises(ToolBail) as excinfo:
                await search_events_handler("sess_01TEST", {"query": "INSERT INTO foo VALUES (1)"})
            mock_exec.assert_not_called()
        assert "SELECT" in excinfo.value.message

    async def test_missing_query(self) -> None:
        with pytest.raises(ToolBail) as excinfo:
            await search_events_handler("sess_01TEST", {})
        assert "query" in excinfo.value.message.lower()

    async def test_empty_query(self) -> None:
        with pytest.raises(ToolBail):
            await search_events_handler("sess_01TEST", {"query": ""})

    async def test_db_error_returns_error(self) -> None:
        with (
            _mock_execute(side_effect=Exception("connection refused")),
            _mock_pool(),
            pytest.raises(ToolBail) as excinfo,
        ):
            await search_events_handler("sess_01TEST", {"query": "SELECT * FROM events_search"})
        assert "failed" in excinfo.value.message.lower()

    async def test_timeout_returns_error(self) -> None:
        import asyncpg.exceptions

        with (
            _mock_execute(side_effect=asyncpg.exceptions.QueryCanceledError()),
            _mock_pool(),
            pytest.raises(ToolBail) as excinfo,
        ):
            await search_events_handler("sess_01TEST", {"query": "SELECT * FROM events_search"})
        assert "timed out" in excinfo.value.message.lower()

    async def test_row_limit_exact_max_no_truncation(self) -> None:
        rows = [_FakeRecord({"id": f"evt_{i}"}) for i in range(MAX_ROWS)]
        with _mock_execute(return_value=(rows, False)), _mock_pool():
            result = await search_events_handler(
                "sess_01TEST", {"query": "SELECT * FROM events_search"}
            )
        assert "truncat" not in result["result"].lower()

    async def test_row_limit_truncation(self) -> None:
        rows = [_FakeRecord({"id": f"evt_{i}"}) for i in range(MAX_ROWS)]
        with _mock_execute(return_value=(rows, True)), _mock_pool():
            result = await search_events_handler(
                "sess_01TEST", {"query": "SELECT * FROM events_search"}
            )
        assert "truncat" in result["result"].lower()

    async def test_no_results(self) -> None:
        with _mock_execute(return_value=([], False)), _mock_pool():
            result = await search_events_handler(
                "sess_01TEST", {"query": "SELECT * FROM events_search"}
            )
        assert result["result"] == "No results."
