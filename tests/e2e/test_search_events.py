"""E2E tests for the search_events built-in tool.

Exercises the full stack: real Postgres (testcontainer), the events_search
view created by migration 0010, and the search_events tool handler querying
against real event data. Verifies session scoping, result formatting, SQL
validation, and read-only enforcement.
"""

from __future__ import annotations

import pytest

from aios.tools.search_events import search_events_handler
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant


@needs_docker
class TestSearchEvents:
    """search_events tool against real Postgres with real event data."""

    async def _setup_session(self, harness: Harness) -> str:
        """Create a session with a user message and an assistant reply."""
        harness.script_model([assistant("I can help with Docker!")])
        session = await harness.start(
            "Tell me about Docker",
            tools=["search_events"],
        )
        await harness.run_until_idle(session.id)
        return session.id

    async def test_basic_keyword_search(self, harness: Harness) -> None:
        """Search for a keyword that appears in the conversation."""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT seq, role, substr(content_text, 1, 100) AS preview "
                    "FROM events_search "
                    "WHERE content_text ILIKE '%docker%' "
                    "ORDER BY seq"
                ),
            },
        )

        assert "result" in result
        assert "docker" in result["result"].lower()

    async def test_count_by_role(self, harness: Harness) -> None:
        """Aggregate query: count events by role."""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT role, count(*) AS n "
                    "FROM events_search "
                    "WHERE kind = 'message' AND role IS NOT NULL "
                    "GROUP BY role "
                    "ORDER BY role"
                ),
            },
        )

        assert "result" in result
        text = result["result"]
        # Should have at least user and assistant rows
        assert "user" in text
        assert "assistant" in text

    async def test_session_scoping(self, harness: Harness) -> None:
        """Events from one session must not appear in another session's query."""
        # Create two independent sessions
        session_id_a = await self._setup_session(harness)

        harness.script_model([assistant("Kubernetes is great!")])
        session_b = await harness.start(
            "Tell me about Kubernetes",
            tools=["search_events"],
        )
        await harness.run_until_idle(session_b.id)

        # Session A should NOT see "Kubernetes"
        result_a = await search_events_handler(
            session_id_a,
            {
                "query": ("SELECT * FROM events_search WHERE content_text ILIKE '%kubernetes%'"),
            },
        )
        assert result_a["result"] == "No results."

        # Session B should NOT see "Docker"
        result_b = await search_events_handler(
            session_b.id,
            {
                "query": ("SELECT * FROM events_search WHERE content_text ILIKE '%docker%'"),
            },
        )
        assert result_b["result"] == "No results."

    async def test_select_star(self, harness: Harness) -> None:
        """SELECT * returns all expected columns."""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT * FROM events_search WHERE kind = 'message' ORDER BY seq LIMIT 1"
                ),
            },
        )

        text = result["result"]
        # Header line should contain all view columns
        header = text.split("\n")[0]
        for col in ("id", "seq", "kind", "role", "created_at", "content_text"):
            assert col in header

    async def test_sql_validation_rejects_insert(self, harness: Harness) -> None:
        """DML queries are rejected before hitting the database."""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": "INSERT INTO events (id) VALUES ('evil')",
            },
        )

        assert "error" in result
        assert "SELECT" in result["error"]

    async def test_read_only_enforcement(self, harness: Harness) -> None:
        """The READ ONLY transaction used by _execute_query blocks writes.

        Verifies the defense-in-depth layer: even if SQL validation were
        bypassed, Postgres itself rejects mutations.
        """
        import asyncpg.exceptions

        from aios.harness import runtime

        session_id = await self._setup_session(harness)
        pool = runtime.require_pool()

        # Simulate what _execute_query does: BEGIN READ ONLY, then attempt
        # a write — Postgres must reject it.
        async with pool.acquire() as conn:
            await conn.execute("BEGIN READ ONLY")
            try:
                with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
                    await conn.execute(
                        "INSERT INTO events (id, session_id, seq, kind, data) "
                        "VALUES ('evil', $1, 999999, 'message', '{}'::jsonb)",
                        session_id,
                    )
            finally:
                await conn.execute("ROLLBACK")

    async def test_no_results_query(self, harness: Harness) -> None:
        """Query that matches nothing returns 'No results.'"""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT * FROM events_search WHERE content_text ILIKE '%zzz_nonexistent_zzz%'"
                ),
            },
        )

        assert result["result"] == "No results."
