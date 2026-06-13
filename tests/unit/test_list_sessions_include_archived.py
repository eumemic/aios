"""Unit test: ``list_sessions`` drops the ``archived_at IS NULL`` clause exactly
when enumerating a workflow run's children (``parent_run_id``) or asking for the
terminal ``status="archived"`` query (#831), and keeps it otherwise.

Exercised against a fake asyncpg connection that captures the emitted SQL, so it
runs without a database. The status-derivation correctness (archived dominating
active/idle) is covered by the integration test.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.db.queries import sessions as session_queries


class _CapturingConn:
    """Captures the SQL the query emits; returns no rows."""

    def __init__(self) -> None:
        self.sql: str | None = None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.sql = sql
        return []


class TestListSessionsIncludeArchived:
    @pytest.mark.parametrize(
        ("kwargs", "expect_archived_filter"),
        [
            ({}, True),  # default listing stays archive-blind
            ({"status": "idle"}, True),  # a non-archived status filter stays blind
            ({"agent_id": "agt_x"}, True),  # plain agent filter stays blind
            ({"parent_run_id": "wfr_x"}, False),  # run children: archived visible
            ({"status": "archived"}, False),  # terminal-status query: archived visible
            # parent_run_id wins even with a live-status filter (still drops clause;
            # the status filter then narrows the visible set).
            ({"parent_run_id": "wfr_x", "status": "idle"}, False),
        ],
    )
    async def test_archived_clause_applied_only_when_blind(
        self, kwargs: dict[str, Any], expect_archived_filter: bool
    ) -> None:
        conn = _CapturingConn()
        await session_queries.list_sessions(conn, account_id="acc_x", **kwargs)
        assert conn.sql is not None
        has_clause = "archived_at IS NULL" in conn.sql
        assert has_clause is expect_archived_filter
