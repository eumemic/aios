"""E2E tests for the search_events built-in tool.

Exercises the full stack: real Postgres (testcontainer), the events_search
view (created by migration 0010, restricted to messages by 0013, widened
with promoted paradigm columns by 0022), and the search_events tool handler
querying against real event data. Verifies session scoping, result
formatting, SQL validation, read-only enforcement, and that the promoted
columns (channel, tool_name, is_error, sender_name) are stamped correctly.
"""

from __future__ import annotations

import pytest

from aios.harness import runtime
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
                    "WHERE role IS NOT NULL "
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
        """SELECT * returns all widened-view columns (migration 0022)."""
        session_id = await self._setup_session(harness)

        result = await search_events_handler(
            session_id,
            {
                "query": ("SELECT * FROM events_search ORDER BY seq LIMIT 1"),
            },
        )

        text = result["result"]
        header = text.split("\n")[0]
        for col in (
            "id",
            "seq",
            "role",
            "channel",
            "tool_name",
            "is_error",
            "sender_name",
            "created_at",
            "content_text",
        ):
            assert col in header, f"missing column {col!r} in header: {header}"
        # kind column should NOT be present (view is still messages-only;
        # span exposure deferred — see issue #117 follow-up).
        assert "kind" not in header

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

    async def test_view_excludes_non_message_events(self, harness: Harness) -> None:
        """The events_search view must only expose message events.

        Lifecycle, span, and interrupt events are harness internals — they
        should not leak into the agent's search results.  After migration
        0013 the view filters to ``kind = 'message'`` and drops the ``kind``
        column entirely.  Migration 0022 keeps this filter in place — span
        exposure is deferred pending a per-agent tool-access-control design.
        """
        session_id = await self._setup_session(harness)

        # A normal step produces lifecycle + span events alongside messages.
        # Verify the raw event log has non-message events…
        all_events = await harness.all_events(session_id)
        non_message_kinds = {e.kind for e in all_events if e.kind != "message"}
        assert non_message_kinds, "test setup: expected lifecycle/span events in the log"

        # …but the search view must not expose them.  Count total rows in the
        # view vs message events in the raw log — they must match.
        msg_count = sum(1 for e in all_events if e.kind == "message")

        result = await search_events_handler(
            session_id,
            {"query": "SELECT count(*) AS n FROM events_search"},
        )

        text = result["result"]
        assert str(msg_count) in text, f"expected {msg_count} rows in view, got: {text}"


@needs_docker
class TestPromotedColumns:
    """Migration 0022: role, channel, tool_name, is_error, sender_name.

    Append synthetic events covering each shape `append_event` needs to
    handle (user with metadata, assistant with tool_calls, tool result
    success, tool result failure) and verify each column is stamped and
    queryable via events_search.
    """

    async def _bare_session(self, harness: Harness) -> str:
        """Create a session with no scripted model calls — we only need
        the session row so we can hand-append events against it."""
        harness.script_model([])
        session = await harness.start("seed", tools=["search_events"])
        return session.id

    async def test_channel_column_stamped_and_queryable(self, harness: Harness) -> None:
        """User events carry the metadata.channel through to events.channel."""
        from aios.services import sessions as sessions_service

        session_id = await self._bare_session(harness)
        pool = runtime.require_pool()
        await sessions_service.append_user_message(
            pool,
            session_id,
            "hello on A",
            metadata={"channel": "slack:CHANA", "sender_name": "alice"},
        )
        await sessions_service.append_user_message(
            pool,
            session_id,
            "hello on B",
            metadata={"channel": "slack:CHANB", "sender_name": "bob"},
        )

        # Direct column read via the pool to sidestep the
        # session-scoping machinery on events_search.
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT channel, sender_name FROM events "
                "WHERE session_id = $1 AND kind = 'message' "
                "AND data->>'role' = 'user' ORDER BY seq",
                session_id,
            )
        channels = [r["channel"] for r in rows]
        senders = [r["sender_name"] for r in rows]
        # The seed message from `start()` has no channel metadata.
        assert channels[-2:] == ["slack:CHANA", "slack:CHANB"]
        assert senders[-2:] == ["alice", "bob"]

        # And queryable via the view.
        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT channel, sender_name, content_text "
                    "FROM events_search WHERE channel = 'slack:CHANA'"
                ),
            },
        )
        assert "slack:CHANA" in result["result"]
        assert "alice" in result["result"]
        assert "hello on A" in result["result"]
        assert "CHANB" not in result["result"]

    async def test_tool_name_column_for_assistant_and_tool_rows(self, harness: Harness) -> None:
        """Assistant turns with tool_calls promote the first name;
        tool-result rows promote data.name."""
        from aios.db import queries

        session_id = await self._bare_session(harness)
        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                    ],
                },
            )
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "bash",
                    "content": "hello",
                },
            )

        result = await search_events_handler(
            session_id,
            {
                "query": (
                    "SELECT role, tool_name FROM events_search "
                    "WHERE tool_name = 'bash' ORDER BY seq"
                ),
            },
        )
        text = result["result"]
        # Both the assistant row (first tool_call was 'bash') and the tool
        # row (name='bash') should match.
        assert text.count("bash") >= 2
        assert "assistant" in text
        assert "tool" in text
        # The second tool_call's name is NOT promoted — multi-tool turns
        # expose only the first — so this query returns no rows.
        result2 = await search_events_handler(
            session_id,
            {"query": "SELECT 1 FROM events_search WHERE tool_name = 'read'"},
        )
        assert result2["result"] == "No results."

    async def test_is_error_column_nullable_true_only(self, harness: Harness) -> None:
        """is_error is TRUE on failures, NULL on success — never FALSE."""
        from aios.db import queries

        session_id = await self._bare_session(harness)
        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "call_ok",
                    "name": "bash",
                    "content": "ok",
                },
            )
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "call_boom",
                    "name": "bash",
                    "content": '{"error": "nope"}',
                    "is_error": True,
                },
            )

        # Direct column read — the view formatter stringifies NULLs to
        # 'NULL' which conflates with FALSE otherwise.
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data->>'tool_call_id' AS tcid, is_error FROM events "
                "WHERE session_id = $1 AND kind = 'message' "
                "AND data->>'role' = 'tool' ORDER BY seq",
                session_id,
            )
        by_tcid = {r["tcid"]: r["is_error"] for r in rows}
        assert by_tcid["call_ok"] is None
        assert by_tcid["call_boom"] is True

        # The `WHERE is_error` predicate implicitly excludes NULL.
        result = await search_events_handler(
            session_id,
            {"query": "SELECT count(*) AS n FROM events_search WHERE is_error"},
        )
        assert "1" in result["result"]
