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

    assert statements == [
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_account_recency_idx",
        "CREATE INDEX CONCURRENTLY wf_runs_account_recency_idx "
        "ON wf_runs (account_id, created_at DESC, id DESC) WHERE archived_at IS NULL",
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_account_workflow_recency_idx",
        "CREATE INDEX CONCURRENTLY wf_runs_account_workflow_recency_idx "
        "ON wf_runs (account_id, workflow_id, created_at DESC, id DESC) "
        "WHERE archived_at IS NULL",
    ]
    context.autocommit_block.assert_called_once_with()


def test_downgrade_removes_both_indexes_concurrently_outside_transaction() -> None:
    statements, context = _capture("downgrade")

    assert statements == [
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_account_workflow_recency_idx",
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_account_recency_idx",
    ]
    context.autocommit_block.assert_called_once_with()
