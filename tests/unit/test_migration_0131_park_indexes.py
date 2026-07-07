"""Guard migration 0131's park crash-recovery partial indexes (#1707).

``find_unharvested_model_dispatch_parks`` seq-scanned ``events`` on every 30s
cross-session sweep because its ``kind='span' AND data->>'event'=
'model_workflow_park'`` predicate had no supporting index. 0131 adds the required
park index plus the two harvest anti-join companions, all as partial
``CREATE INDEX CONCURRENTLY``.

These assertions pin the DDL that makes the EXPLAIN flip from ``Seq Scan`` to an
index scan: the partial predicates (so the index is scoped to exactly the sweep's
rows), the leading key columns (so the scoped/anti-join lookups can seek), and the
``CONCURRENTLY`` + ``autocommit_block`` build discipline (so applying the
migration on a live ``events`` table never takes an ACCESS EXCLUSIVE lock).

Pure-Python: loads the migration module off disk; no DB, no Docker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"


def _load(name: str) -> ModuleType:
    path = next(_VERSIONS_DIR.glob(f"{name}_*.py"))
    spec = importlib.util.spec_from_file_location(f"_mig_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_pointers() -> None:
    m = _load("0131")
    assert m.revision == "0131"
    assert m.down_revision == "0130"


def test_park_index_is_the_required_partial_index() -> None:
    """The required deliverable: a partial index on the park predicate keyed for
    the ``DISTINCT ON (session_id) … ORDER BY session_id, seq DESC`` scan."""
    ddl = _load("0131").PARK_INDEX
    assert "CREATE INDEX CONCURRENTLY" in ddl
    assert "events_model_workflow_park_idx" in ddl
    assert "ON events (session_id, seq DESC)" in ddl
    assert "WHERE kind = 'span'" in ddl
    assert "data->>'event' = 'model_workflow_park'" in ddl


def test_harvest_companion_indexes_cover_the_antijoin_probes() -> None:
    """The two ``NOT EXISTS`` anti-joins probe by
    ``(session_id, account_id, data->>'run_id')`` on the harvest_end / harvest
    span predicates; each needs its own partial index or it still seq-scans."""
    m = _load("0131")
    harvest_end = (
        m.HARVEST_END_INDEX,
        "events_model_workflow_harvest_end_idx",
        "model_workflow_harvest_end",
    )
    harvest = (m.HARVEST_INDEX, "events_model_workflow_harvest_idx", "model_workflow_harvest")
    for ddl, name, event in (harvest_end, harvest):
        assert "CREATE INDEX CONCURRENTLY" in ddl
        assert name in ddl
        assert "ON events (session_id, account_id, (data->>'run_id'))" in ddl
        assert "WHERE kind = 'span'" in ddl
        assert f"data->>'event' = '{event}'" in ddl


def test_harvest_companions_are_distinct_predicates() -> None:
    """``model_workflow_harvest`` is a prefix of ``model_workflow_harvest_end``;
    the two indexes must target the two distinct events, not collide."""
    m = _load("0131")
    assert "'model_workflow_harvest_end'" in m.HARVEST_END_INDEX
    assert "'model_workflow_harvest'" in m.HARVEST_INDEX
    assert "'model_workflow_harvest_end'" not in m.HARVEST_INDEX


def test_all_three_indexes_applied_in_upgrade_concurrently() -> None:
    """upgrade()/downgrade() run inside an autocommit_block (required for
    CONCURRENTLY) and every DROP is guarded IF EXISTS + CONCURRENTLY."""
    src = (next(_VERSIONS_DIR.glob("0131_*.py"))).read_text()
    assert "autocommit_block()" in src
    for name in (
        "events_model_workflow_park_idx",
        "events_model_workflow_harvest_end_idx",
        "events_model_workflow_harvest_idx",
    ):
        assert f"DROP INDEX CONCURRENTLY IF EXISTS {name}" in src
