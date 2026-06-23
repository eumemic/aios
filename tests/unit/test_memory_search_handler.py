"""Unit tests for the memory_search tool handler (pure/no-DB surface).

The DB-backed behaviour (scoping, ts_rank ordering, soft-delete exclusion,
read-only enforcement) lives in
``tests/integration/test_migrations_0119_memories_fts.py``. These cover the
parts that don't need Postgres: the empty-query guard and result formatting.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.tools.memory_search import (
    _PREVIEW_CHARS,
    MAX_ROWS,
    _format_results,
    memory_search_handler,
)


class _FakeRecord(dict):
    """asyncpg.Record-like: indexable by column name, the only access
    ``_format_results`` uses."""


def _row(store: str, path: str, size: int, content: str, rank: float) -> _FakeRecord:
    return _FakeRecord(
        store=store,
        path=path,
        content_size_bytes=size,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        rank=rank,
        content=content,
    )


class TestFormatResults:
    def test_empty(self) -> None:
        assert _format_results([], truncated=False) == "No results."

    def test_renders_store_path_rank_and_preview(self) -> None:
        rows = [_row("notes", "/a.md", 12, "hello world", 0.5)]
        out = _format_results(rows, truncated=False)
        assert "store=notes" in out
        assert "path=/a.md" in out
        assert "rank=0.5000" in out
        assert "hello world" in out

    def test_preview_truncated_for_long_content(self) -> None:
        long = "x" * (_PREVIEW_CHARS + 100)
        out = _format_results([_row("s", "/p", len(long), long, 0.1)], truncated=False)
        assert "…" in out
        # The full body is not dumped — only the preview slice.
        assert ("x" * (_PREVIEW_CHARS + 1)) not in out

    def test_newlines_flattened_in_preview(self) -> None:
        out = _format_results([_row("s", "/p", 3, "a\nb\nc", 0.1)], truncated=False)
        # Preview lines are joined on a single line (no raw newline inside body).
        body_line = out.splitlines()[-1]
        assert "a b c" in body_line

    def test_truncation_notice(self) -> None:
        out = _format_results([_row("s", "/p", 1, "y", 0.1)], truncated=True)
        assert f"truncated to {MAX_ROWS}" in out.lower()


class TestHandlerGuards:
    async def test_rejects_empty_query(self) -> None:
        result = await memory_search_handler("sess_x", {"query": "   "})
        assert "error" in result
        assert "query" in result["error"]

    async def test_rejects_missing_query(self) -> None:
        result = await memory_search_handler("sess_x", {})
        assert "error" in result

    async def test_rejects_non_string_query(self) -> None:
        result = await memory_search_handler("sess_x", {"query": 123})
        assert "error" in result
