"""Regression tests for the online recent errored-session index."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

_MIGRATION = (
    Path(__file__).parents[2] / "migrations" / "versions" / "0181_sessions_recent_error_idx.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_migration_0181", _MIGRATION)
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
        "DROP INDEX CONCURRENTLY IF EXISTS sessions_recent_error_idx",
        "CREATE INDEX CONCURRENTLY sessions_recent_error_idx "
        "ON sessions (account_id, updated_at DESC) "
        "WHERE archived_at IS NULL AND stop_reason->>'type' = 'error'",
    ]
    context.autocommit_block.assert_called_once_with()


def test_downgrade_removes_index_concurrently() -> None:
    statements, context = _capture("downgrade")

    assert statements == [
        "DROP INDEX CONCURRENTLY IF EXISTS sessions_recent_error_idx",
    ]
    context.autocommit_block.assert_called_once_with()
