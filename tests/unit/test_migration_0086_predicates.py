"""Byte-identity guard for migration 0086's CHECK predicate extension.

The slice-2 contract mandates that 0086's predicate constants extend 0083's by
APPENDING new CASE branches only — every existing branch byte-identical, so the
constraint swap can never silently re-shape what slice-1 rows must satisfy.
These tests enforce that structurally: each 0086 predicate must be exactly
0083's text with one new ``WHEN`` branch spliced in before ``ELSE false``, and
the ``*_0083`` constants 0086 embeds for ``downgrade()`` must equal the live
0083 constants byte-for-byte (migrations load under synthetic module names and
cannot import each other, so the embedding is the only way — and the only way
it can drift).

Pure-Python: loads the two migration modules off disk; no DB, no Docker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "migrations" / "versions"

# Both predicates close with this exact tail; the new branch is spliced
# immediately before it.
_TAIL = "        ELSE false\n    END\n), false)"


def _load(name: str) -> ModuleType:
    path = next(_VERSIONS_DIR.glob(f"{name}_*.py"))
    spec = importlib.util.spec_from_file_location(f"_mig_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_appended_branch_only(old: str, new: str, *, added_branch: str) -> None:
    """``new`` == ``old`` with exactly one new ``WHEN`` branch before the tail."""
    assert old.endswith(_TAIL)
    assert new.endswith(_TAIL)
    old_prefix = old.removesuffix(_TAIL)
    # The old branches survive byte-identically as the new predicate's prefix…
    assert new.startswith(old_prefix), "existing CASE branches were modified"
    # …and the only insertion is the named new branch.
    inserted = new.removesuffix(_TAIL).removeprefix(old_prefix)
    assert inserted.lstrip().startswith(f"WHEN '{added_branch}' THEN")


def test_source_spec_predicate_extends_0083_byte_identically() -> None:
    m83, m86 = _load("0083"), _load("0086")
    _assert_appended_branch_only(
        m83.SOURCE_SPEC_PREDICATE, m86.SOURCE_SPEC_PREDICATE, added_branch="run_completion"
    )


def test_action_predicate_extends_0083_byte_identically() -> None:
    m83, m86 = _load("0083"), _load("0086")
    _assert_appended_branch_only(
        m83.ACTION_PREDICATE, m86.ACTION_PREDICATE, added_branch="workflow"
    )


def test_embedded_downgrade_constants_match_live_0083() -> None:
    m83, m86 = _load("0083"), _load("0086")
    assert m86.SOURCE_SPEC_PREDICATE_0083 == m83.SOURCE_SPEC_PREDICATE
    assert m86.ACTION_PREDICATE_0083 == m83.ACTION_PREDICATE


# ─── 0107: wake_session action kind (#1280) — same byte-identity discipline ──


def test_0107_action_predicate_extends_0086_byte_identically() -> None:
    """0107's ACTION_PREDICATE is 0086's with exactly one new ``wake_session``
    branch spliced before the tail — every prior branch byte-identical, so the
    constraint swap can never silently re-shape what slice-2 rows must satisfy.
    """
    m86, m107 = _load("0086"), _load("0107")
    _assert_appended_branch_only(
        m86.ACTION_PREDICATE, m107.ACTION_PREDICATE, added_branch="wake_session"
    )


def test_0107_embedded_downgrade_constant_matches_live_0086() -> None:
    m86, m107 = _load("0086"), _load("0107")
    assert m107.ACTION_PREDICATE_0086 == m86.ACTION_PREDICATE
