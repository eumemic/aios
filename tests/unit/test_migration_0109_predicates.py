"""Byte-identity / vocabulary guard for migration 0109's residue_events CHECKs
and append-only trigger (#1328).

Same discipline as ``test_migration_0108_predicates``: pin the migration's
load-bearing CHECK predicate text and the append-only trigger so a silent
vocabulary drift (a typo'd finder, an axis collapsed to a single value, an
accidentally-relaxed append-only guard) is caught WITHOUT a DB.

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


def test_revision_chains_onto_0108() -> None:
    m = _load("0109")
    assert m.revision == "0109"
    assert m.down_revision == "0108"


def test_axis_check_is_exactly_one_and_two() -> None:
    m = _load("0109")
    # The segregation is a column-level invariant: axis is 1 or 2 and NOTHING
    # else. A drift that admitted a third axis would let a cross-axis sum hide.
    assert m.AXIS_CHECK == "axis IN (1, 2)"


def test_finder_check_freezes_the_four_finders() -> None:
    m = _load("0109")
    pred = m.FINDER_CHECK
    for finder in (
        "internal-armed-check",
        "external-world",
        "chairman",
        "seat-incidental",
    ):
        assert "'%s'" % finder in pred, finder
    # A new finder must be a deliberate migration, not a typo: exactly four.
    assert pred.count("'") == 8


def test_kind_source_check_freezes_the_three_sources() -> None:
    m = _load("0109")
    pred = m.KIND_SOURCE_CHECK
    for src in ("gate-resolve-payload", "observer", "manual"):
        assert "'%s'" % src in pred, src
    assert pred.count("'") == 6


def test_append_only_trigger_raises_on_update_and_delete() -> None:
    m = _load("0109")
    src = Path(next(_VERSIONS_DIR.glob("0109_*.py"))).read_text()
    # The trigger must fire on BOTH UPDATE and DELETE and RAISE — append-only is
    # structural, not convention.
    assert "BEFORE UPDATE OR DELETE ON residue_events" in src
    assert "RAISE EXCEPTION" in src
    assert m.APPEND_ONLY_TRIGGER in src
    assert m.APPEND_ONLY_FN in src


def test_residue_kind_is_not_check_constrained() -> None:
    """The residue_kind enum is deliberately OPEN (``other``-bucket growth is the
    render-side alarm, not a constraint), so the migration must NOT add a CHECK on
    residue_kind."""
    src = Path(next(_VERSIONS_DIR.glob("0109_*.py"))).read_text()
    assert "residue_kind        text NOT NULL," in src
    assert "residue_kind text NOT NULL CHECK" not in src.replace("  ", " ")


def test_idempotency_unique_is_account_scoped() -> None:
    src = Path(next(_VERSIONS_DIR.glob("0109_*.py"))).read_text()
    assert "UNIQUE (account_id, idempotency_key)" in src
