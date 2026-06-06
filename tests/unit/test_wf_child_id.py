"""B2.A — deterministic child-session id (folds the full call_key)."""

from __future__ import annotations

from aios.ids import split_id
from aios.workflows.child_id import child_session_id


def test_deterministic_and_parseable() -> None:
    a = child_session_id("wfr_run", "sha:abc#0")
    b = child_session_id("wfr_run", "sha:abc#0")
    assert a == b  # same (run_id, call_key) -> same id (replay reproduces it)
    prefix, ulid_part = split_id(a)
    assert prefix == "sess" and len(ulid_part) == 26


def test_identical_siblings_get_distinct_ids() -> None:
    # The §3.1 pinned invariant: parallel([agent('ping')]*3) -> keys #0/#1/#2 ->
    # THREE distinct children. Folding the content hash alone would collapse them.
    ids = {child_session_id("wfr_run", f"sha:abc#{n}") for n in range(3)}
    assert len(ids) == 3


def test_distinct_runs_get_distinct_ids() -> None:
    assert child_session_id("wfr_a", "sha:abc#0") != child_session_id("wfr_b", "sha:abc#0")
