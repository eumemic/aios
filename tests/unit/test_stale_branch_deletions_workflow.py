"""Load-bearing checks for the stale-branch deletion workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "stale-branch-deletions.yml"
)


def _doc() -> dict[Any, Any]:
    doc: dict[Any, Any] = yaml.safe_load(_WORKFLOW.read_text())
    return doc


def test_runs_for_pr_changes_including_description_edits() -> None:
    triggers = _doc().get("on", _doc().get(True))
    assert isinstance(triggers, dict)
    assert triggers["pull_request"]["types"] == ["opened", "synchronize", "reopened", "edited"]


def test_uses_github_pr_files_api_instead_of_local_diff() -> None:
    steps = _doc()["jobs"]["check"]["steps"]
    run = next(step["run"] for step in steps if "run" in step)

    assert "/pulls/${PR_NUMBER}/files" in run
    assert "--paginate" in run
    assert "git diff" not in run
    assert "scripts/check_pr_deletions.py" in run


def test_permissions_are_read_only() -> None:
    assert _doc()["permissions"] == {"contents": "read", "pull-requests": "read"}
