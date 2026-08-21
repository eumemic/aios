"""Tests for the PR deletion guard used by GitHub Actions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_pr_deletions.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pr_deletions", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmentioned_removed_file_is_flagged_with_actionable_message() -> None:
    guard = _module()
    suspicious = guard.find_suspicious_deletions(
        files=[{"filename": "tests/unit/test_guard.py", "status": "removed", "sha": "old"}],
        description="Improve the runtime",
        base_paths={"tests/unit/test_guard.py"},
    )

    assert suspicious == ["tests/unit/test_guard.py"]
    message = guard.failure_message(suspicious)
    assert "tests/unit/test_guard.py" in message
    assert "branch is STALE (forked before those files landed)" in message
    assert "Rebase onto master" in message
    assert "say so in the PR body" in message


def test_filename_in_title_or_body_documents_intent() -> None:
    guard = _module()
    files = [{"filename": "src/legacy.py", "status": "removed", "sha": "old"}]

    assert guard.find_suspicious_deletions(files, "Remove `src/legacy.py`", {"src/legacy.py"}) == []


def test_same_content_move_is_not_flagged() -> None:
    guard = _module()
    files = [
        {"filename": "src/old.py", "status": "removed", "sha": "same-blob"},
        {"filename": "src/new.py", "status": "added", "sha": "same-blob"},
    ]

    assert guard.find_suspicious_deletions(files, "Move module", {"src/old.py"}) == []


def test_stale_branch_without_real_api_deletions_does_not_fire() -> None:
    """Regression for #2063: tip-diff looked deleted, GitHub reported only additions."""
    guard = _module()
    github_files = [{"filename": "src/new_lane.py", "status": "added", "sha": "new"}]

    assert (
        guard.find_suspicious_deletions(github_files, "Activate lane", {"tests/new_on_base.py"})
        == []
    )


def test_mutation_real_deletion_is_red_then_rebased_file_list_is_green() -> None:
    """Pin the verdict transition to GitHub's merge-base-correct PR file list."""
    guard = _module()
    base_paths = {"tests/new_on_base.py"}
    stale_deletion = [{"filename": "tests/new_on_base.py", "status": "removed", "sha": "base-blob"}]

    assert guard.find_suspicious_deletions(stale_deletion, "Unrelated work", base_paths)
    assert guard.find_suspicious_deletions([], "Unrelated work", base_paths) == []


def test_removed_path_absent_from_current_base_is_not_flagged() -> None:
    guard = _module()
    files = [{"filename": "already-gone.py", "status": "removed", "sha": "old"}]

    assert guard.find_suspicious_deletions(files, "Change", set()) == []
