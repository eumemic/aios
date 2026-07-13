"""Regression tests for the per-agent inbound-budget index migration."""

from __future__ import annotations

from pathlib import Path

MIGRATION = (
    Path(__file__).parents[2]
    / "migrations"
    / "versions"
    / "0149_events_agent_inbound_budget_index.py"
)


def test_agent_inbound_budget_index_matches_session_window_query() -> None:
    ddl = MIGRATION.read_text()

    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_agent_inbound_budget_idx" in ddl
    assert "ON events (session_id, created_at)" in ddl
    assert "kind = 'message' AND data->>'role' = 'user'" in ddl
    assert "kind = 'lifecycle' AND (data->>'wake')::boolean IS TRUE" in ddl


def test_agent_inbound_budget_index_is_reversibly_dropped_concurrently() -> None:
    ddl = MIGRATION.read_text()

    assert "DROP INDEX CONCURRENTLY IF EXISTS events_agent_inbound_budget_idx" in ddl
    assert ddl.count("autocommit_block()") == 2
