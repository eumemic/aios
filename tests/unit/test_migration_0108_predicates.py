"""Byte-identity guard for migration 0108's CHECK predicate extension (#1281).

Same discipline as ``test_migration_0086_predicates``: 0108's
``SOURCE_SPEC_PREDICATE`` must be 0086's text with exactly one new
``external_event`` ``WHEN`` branch spliced before ``ELSE false`` — every prior
branch byte-identical — and the ``*_0086`` constant 0108 embeds for
``downgrade()`` must equal the live 0086 constant byte-for-byte (migrations load
under synthetic module names and cannot import each other).

Pure-Python: loads the migration modules off disk; no DB, no Docker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"

_TAIL = "        ELSE false\n    END\n), false)"


def _load(name: str) -> ModuleType:
    path = next(_VERSIONS_DIR.glob(f"{name}_*.py"))
    spec = importlib.util.spec_from_file_location(f"_mig_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_spec_predicate_extends_0086_byte_identically() -> None:
    m86, m108 = _load("0086"), _load("0108")
    old, new = m86.SOURCE_SPEC_PREDICATE, m108.SOURCE_SPEC_PREDICATE
    assert old.endswith(_TAIL)
    assert new.endswith(_TAIL)
    old_prefix = old.removesuffix(_TAIL)
    assert new.startswith(old_prefix), "existing CASE branches were modified"
    inserted = new.removesuffix(_TAIL).removeprefix(old_prefix)
    assert inserted.lstrip().startswith("WHEN 'external_event' THEN")


def test_embedded_downgrade_constant_matches_live_0086() -> None:
    m86, m108 = _load("0086"), _load("0108")
    assert m108.SOURCE_SPEC_PREDICATE_0086 == m86.SOURCE_SPEC_PREDICATE


def test_reactive_no_next_fire_covers_both_reactive_kinds() -> None:
    m108 = _load("0108")
    pred = m108.REACTIVE_NO_NEXT_FIRE_PREDICATE
    assert "run_completion" in pred and "external_event" in pred
    assert "next_fire IS NULL" in pred


def test_ingest_iff_predicate_is_external_event_iff_hash() -> None:
    m108 = _load("0108")
    pred = m108.INGEST_TOKEN_IFF_EXTERNAL_EVENT_PREDICATE
    assert "source = 'external_event'" in pred
    assert "ingest_token_hash IS NOT NULL" in pred


def test_revision_chains_onto_0107() -> None:
    m108 = _load("0108")
    assert m108.revision == "0108"
    assert m108.down_revision == "0107"
