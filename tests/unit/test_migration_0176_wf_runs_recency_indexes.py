"""Regression tests for the online workflow-run recency indexes."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

_MIGRATION = (
    Path(__file__).parents[2] / "migrations" / "versions" / "0176_wf_runs_recency_indexes.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_migration_0176", _MIGRATION)
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


def test_upgrade_builds_both_indexes_concurrently_outside_transaction() -> None:
    statements, context = _capture("upgrade")

    creates = [statement for statement in statements if statement.startswith("CREATE INDEX")]
    drops = [statement for statement in statements if statement.startswith("DROP INDEX")]
    assert len(creates) == 2
    assert all(statement.startswith("CREATE INDEX CONCURRENTLY ") for statement in creates)
    assert len(drops) == 2
    assert all(statement.startswith("DROP INDEX CONCURRENTLY IF EXISTS ") for statement in drops)
    context.autocommit_block.assert_called_once_with()


def test_downgrade_removes_both_indexes_concurrently_outside_transaction() -> None:
    statements, context = _capture("downgrade")

    assert len(statements) == 2
    assert all(statement.startswith("DROP INDEX CONCURRENTLY IF EXISTS ") for statement in statements)
    context.autocommit_block.assert_called_once_with()
