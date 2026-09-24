"""Statement-shape regression tests for migration 0182 (#2446 d).

The column add must stay nullable with no default (metadata-only on a populated
``wf_runs``) and the index must build online, replacing any invalid remnant of
an interrupted concurrent build. The live upgrade/downgrade round trip is in
``tests/integration/test_migrations_0182_wf_runs_trigger_id.py``.
"""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

_MIGRATION = Path(__file__).parents[2] / "migrations" / "versions" / "0182_wf_runs_trigger_id.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_migration_0182", _MIGRATION)
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


def test_revision_chain() -> None:
    migration = _load()
    assert (migration.revision, migration.down_revision) == ("0182", "0181")


def test_upgrade_adds_nullable_column_then_builds_index_concurrently() -> None:
    statements, context = _capture("upgrade")
    assert statements == [
        "ALTER TABLE wf_runs ADD COLUMN IF NOT EXISTS trigger_id text",
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_trigger_active_idx",
        "CREATE INDEX CONCURRENTLY wf_runs_trigger_active_idx ON wf_runs (trigger_id) "
        "WHERE trigger_id IS NOT NULL AND archived_at IS NULL "
        "AND status IN ('pending','running','suspended')",
    ]
    context.autocommit_block.assert_called_once_with()


def test_downgrade_drops_index_then_column() -> None:
    statements, context = _capture("downgrade")
    assert statements == [
        "DROP INDEX CONCURRENTLY IF EXISTS wf_runs_trigger_active_idx",
        "ALTER TABLE wf_runs DROP COLUMN IF EXISTS trigger_id",
    ]
    context.autocommit_block.assert_called_once_with()
