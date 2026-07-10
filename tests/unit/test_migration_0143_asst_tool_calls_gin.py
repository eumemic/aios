"""Guard migration 0143's assistant tool-call containment GIN index (#1737)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"


def _load() -> ModuleType:
    path = next(_VERSIONS_DIR.glob("0143_*.py"))
    spec = importlib.util.spec_from_file_location("_mig_0143", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_extends_the_single_head() -> None:
    migration = _load()
    assert migration.revision == "0143"
    assert migration.down_revision == "0142"


def test_index_supports_the_unmodified_containment_probe() -> None:
    ddl = _load().CREATE_INDEX
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_asst_tool_calls_gin_idx" in ddl
    assert "ON events USING gin ((data->'tool_calls') jsonb_path_ops)" in ddl
    assert "WHERE kind = 'message' AND role = 'assistant'" in ddl
    # This term prevents PostgreSQL from proving the unmodified query implies
    # the partial predicate, making the index unusable.
    assert "data ? 'tool_calls'" not in ddl


def test_upgrade_and_downgrade_use_autocommit() -> None:
    source = next(_VERSIONS_DIR.glob("0143_*.py")).read_text()
    assert source.count("autocommit_block()") >= 2
    assert "DROP INDEX CONCURRENTLY IF EXISTS events_asst_tool_calls_gin_idx" in source
