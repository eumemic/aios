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
    """Tests for ``_validate_sql`` — the sqlglot AST walk that gates the tool.

    The guarantee: the model's SQL surface is exactly the per-session
    ``events_search`` view. Validation parses the query and walks the AST,
    so it enforces the boundary structurally rather than by pattern-matching
    text — closing the regex-era bypasses (comma cross-join, in-string comment
    tricks, ``UNION TABLE``, ``pg_*`` in SELECT position).
    """

    # ── Legitimate queries pass ──────────────────────────────────────────────

    def test_valid_select(self) -> None:
        assert _validate_sql("SELECT * FROM events_search") is None

    def test_valid_select_count(self) -> None:
        assert _validate_sql("SELECT COUNT(*) FROM events_search") is None

    def test_valid_select_with_where(self) -> None:
        assert _validate_sql("SELECT id FROM events_search WHERE role = 'user' LIMIT 10") is None

    def test_valid_select_case_insensitive(self) -> None:
        assert _validate_sql("select * from events_search") is None

    def test_allows_leading_whitespace(self) -> None:
        assert _validate_sql("  SELECT 1") is None

    def test_allows_no_from(self) -> None:
        """A query with no relation target touches no table and is allowed."""
        assert _validate_sql("SELECT 1") is None
        assert _validate_sql("SELECT now()") is None

    def test_allows_schema_qualified_public(self) -> None:
        """Explicit ``public.`` qualification is fine — that's where the view
        lives on the default search path. (The regex era rejected any schema
        qualification; the AST distinguishes ``public`` from ``pg_catalog``.)"""
        assert _validate_sql("SELECT * FROM public.events_search") is None

    def test_allows_cte(self) -> None:
        """CTEs are genuinely supported now: the CTE name is a local relation,
        and its body is walked so a CTE reading a forbidden table is still
        caught (see ``test_rejects_forbidden_table_inside_cte_body``)."""
        assert _validate_sql("WITH x AS (SELECT * FROM events_search) SELECT * FROM x") is None

    def test_allows_subquery(self) -> None:
        assert _validate_sql("SELECT * FROM (SELECT * FROM events_search) s") is None

    def test_allows_window_function(self) -> None:
        assert (
            _validate_sql("SELECT seq, row_number() OVER (ORDER BY seq) FROM events_search") is None
        )

    def test_allows_from_keyword_inside_string_literal(self) -> None:
        """A ``FROM`` inside a string literal is not a relation target — the
        parser sees it as a string, so this legitimate query passes."""
        assert (
            _validate_sql("SELECT count(*) FROM events_search WHERE body ILIKE '%FROM secret%'")
            is None
        )

    def test_allows_current_setting(self) -> None:
        """``current_setting`` is not ``pg_*`` / ``set_config`` — it only reads
        the (already session-scoped) GUC and is harmless."""
        assert _validate_sql("SELECT current_setting('app.session_id')") is None

    # ── The four confirmed regex-era cross-tenant bypasses now REJECT ────────

    def test_rejects_comma_cross_join(self) -> None:
        """``FROM events_search, events`` — a comma cross-join. The regex only
        scanned the token right after FROM/JOIN; the AST surfaces every table."""
        err = _validate_sql("SELECT * FROM events_search, events")
        assert err is not None
        assert "events" in err

    def test_rejects_in_string_comment_union_tricks(self) -> None:
        """A comment opened inside a string literal used to fool the regex's
        strip-comments-before-strings ordering, hiding a trailing ``UNION``.
        The parser reads the literal correctly and surfaces the ``events`` arm."""
        for q in (
            "SELECT * FROM events_search WHERE x='a -- ' UNION SELECT * FROM events",
            "SELECT * FROM events_search WHERE x='a /* ' UNION SELECT * FROM events",
        ):
            err = _validate_sql(q)
            assert err is not None, f"in-string comment bypass allowed: {q!r}"
            assert "events" in err

    def test_rejects_union_table_primary(self) -> None:
        """``UNION TABLE accounts`` — the ``TABLE`` primary form reaches a
        relation without a FROM/JOIN keyword. sqlglot rejects it as unparseable
        for our read path, so it fails closed."""
        err = _validate_sql("SELECT * FROM events_search UNION TABLE accounts")
        assert err is not None

    def test_rejects_pg_function_in_select_position(self) -> None:
        """``pg_read_file`` reads host files with no table reference at all —
        the regex only looked at FROM/JOIN. The function scan blocks all
        ``pg_*`` functions wherever they appear."""
        err = _validate_sql("SELECT pg_read_file('/etc/passwd')")
        assert err is not None
        assert "pg_read_file" in err

    # ── Forbidden relations ──────────────────────────────────────────────────

    def test_rejects_direct_events_table(self) -> None:
        """The underlying ``events`` table holds every account's rows; reading
        it directly bypasses the view's ``app.session_id`` scoping."""
        err = _validate_sql("SELECT data FROM events")
        assert err is not None
        assert "events" in err

    def test_rejects_other_tables(self) -> None:
        for table in ("vault_credentials", "accounts", "sessions", "connections"):
            err = _validate_sql(f"SELECT * FROM {table}")
            assert err is not None, f"validator allowed access to {table!r}"

    def test_rejects_forgotten_tables_the_old_denylist_missed(self) -> None:
        """The regression this fix closes: the old hand-maintained denylist
        never listed these, so ``SELECT * FROM <them>`` ran on the privileged
        pool and returned every account's rows. An allowlist blocks them by
        default (fail-closed)."""
        for table in ("runtime_tokens", "oauth_flows", "wf_run_events", "chat_sessions"):
            err = _validate_sql(f"SELECT * FROM {table}")
            assert err is not None, f"validator allowed access to {table!r}"
            assert "not allowed" in err

    def test_rejects_memories_search_view(self) -> None:
        """Deliberate narrowing: ``memories_search`` is owned by the separate
        ``memory_search`` tool (a fixed parameterised query), not part of
        search_events' raw-SQL surface, so it is not allowlisted."""
        err = _validate_sql("SELECT * FROM memories_search")
        assert err is not None
        assert "memories_search" in err
        assert "events_search" in err

    def test_rejects_quoted_forbidden_identifier(self) -> None:
        err = _validate_sql("SELECT * FROM \"events\" WHERE session_id = 'sess_OTHER'")
        assert err is not None

    def test_rejects_forbidden_table_inside_cte_body(self) -> None:
        """A CTE whose body reads a forbidden table is caught on that table,
        not waved through because the outer query only names the CTE."""
        err = _validate_sql("WITH x AS (SELECT * FROM accounts) SELECT * FROM x")
        assert err is not None
        assert "accounts" in err

    def test_rejects_forbidden_table_in_join(self) -> None:
        err = _validate_sql("SELECT * FROM events_search e JOIN accounts a ON e.id = a.id")
        assert err is not None
        assert "accounts" in err

    def test_rejects_table_function_fail_closed(self) -> None:
        """A table-valued function in FROM (``generate_series``) parses to a
        relation with an off name that isn't allowlisted — rejected."""
        err = _validate_sql("SELECT * FROM generate_series(1, 10)")
        assert err is not None

    def test_rejects_schema_introspection(self) -> None:
        """``pg_catalog`` / ``information_schema`` enumerate the whole catalogue
        and are rejected as inaccessible schemas."""
        for q in (
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT * FROM information_schema.tables",
        ):
            err = _validate_sql(q)
            assert err is not None, f"validator allowed {q!r}"

    def test_rejection_message_names_the_allowed_relation(self) -> None:
        err = _validate_sql("SELECT * FROM runtime_tokens")
        assert err is not None
        assert "runtime_tokens" in err
        assert "search_events may only read events_search" in err

    # ── Read-only enforcement ────────────────────────────────────────────────

    def test_rejects_write_and_ddl_statements(self) -> None:
        """DML/DDL and command statements (``COPY``/``CALL`` parse to
        ``exp.Command``) are all rejected — none is a read."""
        for q in (
            "INSERT INTO foo VALUES (1)",
            "UPDATE foo SET x = 1",
            "DELETE FROM foo",
            "DROP TABLE events",
            "COPY events TO '/tmp/x'",
            "CALL do_thing()",
        ):
            err = _validate_sql(q)
            assert err is not None, f"validator allowed non-read statement: {q!r}"

    def test_rejects_set_config(self) -> None:
        """``set_config`` could rewrite ``app.session_id`` (widening the view to
        another session) even inside a read-only transaction, so it is blocked
        wherever it appears."""
        err = _validate_sql("SELECT set_config('app.session_id', 'sess_OTHER', true)")
        assert err is not None
        assert "set_config" in err

    def test_rejects_data_modifying_cte(self) -> None:
        """A data-modifying CTE (``WITH x AS (DELETE ... RETURNING ...)``) is a
        write hiding behind a SELECT root — the write-node walk rejects it."""
        err = _validate_sql("WITH x AS (DELETE FROM events RETURNING id) SELECT * FROM x")
        assert err is not None

    # ── Malformed / multi-statement ──────────────────────────────────────────

    def test_rejects_semicolon(self) -> None:
        err = _validate_sql("SELECT 1; DROP TABLE events")
        assert err is not None
        assert "semicolon" in err.lower() or "multiple" in err.lower()

    def test_rejects_empty_string(self) -> None:
        assert _validate_sql("") is not None

    def test_rejects_whitespace_only(self) -> None:
        assert _validate_sql("   ") is not None

    def test_rejects_unparseable(self) -> None:
        assert _validate_sql("SELECT FROM WHERE ORDER") is not None


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
