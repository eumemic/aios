"""Regression tests for the online awaiting-marker index (#2393)."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

_MIGRATION = (
    Path(__file__).parents[2] / "migrations" / "versions" / "0195_events_tool_marker_idx.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_migration_0195", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(operation: str) -> tuple[list[str], Mock]:
    migration = _load()
    context = Mock()
    context.autocommit_block.return_value = nullcontext()
    migration.op.get_context = Mock(return_value=context)
    statements: list[str] = []
    migration.op.execute = statements.append

    getattr(migration, operation)()
    return statements, context


def test_upgrade_replaces_remnant_then_builds_index_concurrently() -> None:
    statements, context = _capture("upgrade")

    assert statements == [
        "SET lock_timeout = '5s'",
        "DROP INDEX CONCURRENTLY IF EXISTS events_tool_marker_idx",
        "CREATE INDEX CONCURRENTLY events_tool_marker_idx "
        "ON events (session_id, (data->>'tool_call_id')) "
        "WHERE kind = 'lifecycle' "
        "AND data->>'event' IN ('tool_confirmed', 'tool_requested')",
        "SET lock_timeout = DEFAULT",
    ]
    context.autocommit_block.assert_called_once_with()


def test_upgrade_resets_lock_timeout_when_the_build_fails() -> None:
    """A timed-out build must not leave the session-scoped bound armed."""
    migration = _load()
    context = Mock()
    context.autocommit_block.return_value = nullcontext()
    migration.op.get_context = Mock(return_value=context)
    statements: list[str] = []

    def execute(sql: str) -> None:
        statements.append(sql)
        if sql.startswith("CREATE INDEX"):
            raise RuntimeError("canceling statement due to lock timeout")

    migration.op.execute = execute
    try:
        migration.upgrade()
    except RuntimeError:
        pass
    else:  # pragma: no cover - the stub always raises
        raise AssertionError("upgrade swallowed the build failure")
    assert statements[0] == "SET lock_timeout = '5s'"
    assert statements[-1] == "SET lock_timeout = DEFAULT"


def test_downgrade_removes_index_concurrently() -> None:
    statements, context = _capture("downgrade")

    assert statements == [
        "SET lock_timeout = '5s'",
        "DROP INDEX CONCURRENTLY IF EXISTS events_tool_marker_idx",
        "SET lock_timeout = DEFAULT",
    ]
    context.autocommit_block.assert_called_once_with()


def test_migration_is_parented_on_masters_head() -> None:
    migration = _load()
    assert migration.revision == "0195"
    assert migration.down_revision == "0194"
