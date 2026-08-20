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


def _triggers() -> dict[str, Any]:
    triggers = _doc().get("on", _doc().get(True))
    assert isinstance(triggers, dict)
    return triggers


def test_runs_for_pr_changes_including_description_edits() -> None:
    assert _triggers()["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "edited",
    ]


def test_base_push_rechecks_open_prs_and_updates_their_head_checks() -> None:
    """A green check from base B must be superseded after master advances."""
    assert _triggers()["push"]["branches"] == ["master"]
    job = _doc()["jobs"]["recheck-open-prs"]
    run = next(step["run"] for step in job["steps"] if "run" in step)

    assert "pulls?state=open&base=${BASE_REF}" in run
    assert 'git ls-tree -r --name-only "$BASE_SHA"' in run
    assert 'head_sha="$head_sha"' in run
    assert 'scripts/pr_files_from_trees.py "$BASE_SHA" "$head_sha"' in run
    assert "scripts/check_pr_deletions.py" in run
    assert "check-runs/${check_id}" in run
    assert "conclusion=failure" in run


def test_every_pr_event_compares_the_current_base_and_head_trees() -> None:
    """A synchronize/reopen must not turn a stale-deletion check green."""
    steps = _doc()["jobs"]["check"]["steps"]
    run = next(step["run"] for step in steps if "run" in step)

    assert 'git ls-tree -r --name-only "$BASE_SHA"' in run
    assert 'scripts/pr_files_from_trees.py "$BASE_SHA" "$HEAD_SHA"' in run
    assert "/pulls/${PR_NUMBER}/files" not in run
    assert "scripts/check_pr_deletions.py" in run


def test_permissions_are_least_privilege_for_publishing_fresh_checks() -> None:
    assert _doc()["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
        "checks": "write",
    }
