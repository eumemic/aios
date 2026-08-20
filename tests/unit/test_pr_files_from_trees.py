"""Integration fixture for comparing an unchanged PR after its base advances."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_only_file_is_a_deletion_from_unchanged_pr_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR is green at B; adding a base file makes a fresh evaluation red."""
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "existing.py").write_text("existing = True\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base B"], cwd=tmp_path, check=True)
    pr_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()

    comparer = _load("pr_files_from_trees", _ROOT / "scripts" / "pr_files_from_trees.py")
    guard = _load("check_pr_deletions", _ROOT / "scripts" / "check_pr_deletions.py")
    monkeypatch.chdir(tmp_path)
    assert (
        guard.find_suspicious_deletions(comparer.compare(pr_head, pr_head), "Work", {"existing.py"})
        == []
    )

    (tmp_path / "new_on_base.py").write_text("landed = True\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "advance base"], cwd=tmp_path, check=True)
    new_base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    files = comparer.compare(new_base, pr_head)
    assert guard.find_suspicious_deletions(
        files, "Unrelated PR", {"existing.py", "new_on_base.py"}
    ) == ["new_on_base.py"]
