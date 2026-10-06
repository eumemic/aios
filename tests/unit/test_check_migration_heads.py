from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.check_migration_heads import (
    MigrationHistoryError,
    check_against_base,
    check_history,
    load_revisions,
)


def _migration(
    path: Path, revision: str, down_revision: str | None, *, annotated: bool = False
) -> None:
    annotation = ": str" if annotated else ""
    parent = "None" if down_revision is None else repr(down_revision)
    path.write_text(
        f"revision{annotation} = {revision!r}\ndown_revision: str | None = {parent}\n",
        encoding="utf-8",
    )


def test_parser_handles_annotated_and_unannotated_revisions(tmp_path: Path) -> None:
    _migration(tmp_path / "0158_base.py", "0158", None, annotated=True)
    _migration(tmp_path / "0159_tip.py", "0159", "0158")

    assert load_revisions(tmp_path) == {"0158": None, "0159": "0158"}
    assert check_history(load_revisions(tmp_path)) == "0159"


@pytest.mark.parametrize("include_valid_root", [False, True])
def test_rejects_revision_whose_parent_is_missing(
    tmp_path: Path, *, include_valid_root: bool
) -> None:
    if include_valid_root:
        _migration(tmp_path / "0158_base.py", "0158", None)
    _migration(tmp_path / "0161_disconnected.py", "0161", "DOES_NOT_EXIST")

    with pytest.raises(MigrationHistoryError, match="unknown down_revision") as exc_info:
        check_history(load_revisions(tmp_path))

    assert "0161" in str(exc_info.value)
    assert "DOES_NOT_EXIST" in str(exc_info.value)


def test_mutation_detects_stale_parent_then_passes_when_reparented(tmp_path: Path) -> None:
    _migration(tmp_path / "0158_base.py", "0158", None)
    _migration(tmp_path / "0159_current_tip.py", "0159", "0158")
    stale = tmp_path / "0161_pr_migration.py"
    _migration(stale, "0161", "0158", annotated=True)

    with pytest.raises(MigrationHistoryError) as exc_info:
        check_history(load_revisions(tmp_path), current_tip="0159")

    assert str(exc_info.value) == (
        'branched alembic history: 0159 and 0161 both declare down_revision="0158"\n'
        "  -> re-parent your migration onto the current tip (0159)\n"
        "  -> NOTE: a git rebase moves the file but does NOT re-parent it"
    )

    _migration(stale, "0161", "0159", annotated=True)
    assert check_history(load_revisions(tmp_path), current_tip="0159") == "0161"


def test_workflow_runs_on_every_push_and_checks_live_master() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "migration-head-check.yml"
    )
    doc: dict[Any, Any] = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict)

    assert triggers["push"] is None
    steps = doc["jobs"]["migration-head"]["steps"]
    live_base_checkout = next(step for step in steps if step.get("name") == "Check out live master")
    assert live_base_checkout["with"]["ref"] == "master"
    check_command = next(
        step["run"] for step in steps if step.get("name", "").startswith("Detect branched")
    )
    assert "--base-versions-dir _base/migrations/versions" in check_command


def test_live_base_mutation_rejects_stale_parent_then_accepts_current_tip() -> None:
    base = {"0158": None, "0159": "0158"}
    stale_branch = {"0158": None, "0161": "0158"}

    with pytest.raises(MigrationHistoryError) as exc_info:
        check_against_base(stale_branch, base)

    assert str(exc_info.value) == (
        "migration branch does not extend the current base head (0159): "
        "combined heads are 0159 and 0161\n"
        "  -> re-parent your migration onto the current base head (0159)"
    )

    current_branch = {"0158": None, "0161": "0159"}
    assert check_against_base(current_branch, base) == "0161"


def test_accepts_linear_stack_rooted_on_live_base_tip() -> None:
    # Expand/backfill/contract in one PR: several new migrations chained on
    # each other, the chain's root parented on the live tip. Nothing is stale.
    base: dict[str, str | None] = {"0158": None, "0159": "0158"}
    stacked = {**base, "0160": "0159", "0161": "0160", "0162": "0161"}

    assert check_against_base(stacked, base) == "0162"


def test_rejects_linear_stack_rooted_below_live_base_tip() -> None:
    base: dict[str, str | None] = {"0158": None, "0159": "0158"}
    stale_stack = {"0158": None, "0160": "0158", "0161": "0160"}

    with pytest.raises(MigrationHistoryError) as exc_info:
        check_against_base(stale_stack, base)

    assert str(exc_info.value) == (
        "migration branch does not extend the current base head (0159): "
        "combined heads are 0159 and 0161\n"
        "  -> re-parent your migration onto the current base head (0159)"
    )


@pytest.mark.parametrize(
    ("branch_only", "expected"),
    [
        # Two new revisions forking off a new revision.
        (
            {"0160": "0159", "0161": "0160", "0162": "0160"},
            'branch migrations fork: 0161 and 0162 both declare down_revision="0160"',
        ),
        # Two independent new chains, both rooted on the live tip.
        (
            {"0160": "0159", "0161": "0159"},
            'branch migrations fork: 0160 and 0161 both declare down_revision="0159"',
        ),
    ],
)
def test_rejects_forked_branch_migrations(branch_only: dict[str, str], expected: str) -> None:
    base: dict[str, str | None] = {"0158": None, "0159": "0158"}

    with pytest.raises(MigrationHistoryError) as exc_info:
        check_against_base({**base, **branch_only}, base)

    assert str(exc_info.value).startswith(expected)


def test_rejects_branch_migration_whose_parent_exists_nowhere() -> None:
    base: dict[str, str | None] = {"0158": None, "0159": "0158"}
    branch = {**base, "0160": "0159", "0161": "DOES_NOT_EXIST"}

    with pytest.raises(MigrationHistoryError) as exc_info:
        check_against_base(branch, base)

    assert str(exc_info.value) == (
        'revision 0161 declares down_revision="DOES_NOT_EXIST", but that parent '
        "is not present on the live base or this branch"
    )


def test_known_good_repository_has_expected_revision_count_and_one_head() -> None:
    versions = Path(__file__).resolve().parents[2] / "migrations" / "versions"
    revisions = load_revisions(versions)

    # Bump this when adding a migration — the count pins accidental
    # deletions/duplications, not a maximum.
    #
    # Resolved on rebase to the ACTUAL count on this tree (161), not to either
    # side of the conflict: master said 160 and the branch said 159, and both
    # were stale. This branch adds no migration; master has gained two since the
    # branch was cut. A count pin resolved by PICKING A SIDE re-asserts a number
    # nobody re-measured -- which is how a pin meant to catch accidental
    # deletions becomes the thing that fails CI.
    assert len(revisions) == 177
    assert check_history(revisions) in revisions


@pytest.mark.parametrize(
    "branch_only",
    [
        {"0160": "0160"},
        {"0160": "0161", "0161": "0160"},
        {"0160": "0159", "0161": "0162", "0162": "0161"},
    ],
)
def test_rejects_cyclic_or_disconnected_branch_migrations(branch_only: dict[str, str]) -> None:
    base: dict[str, str | None] = {"0158": None, "0159": "0158"}

    with pytest.raises(MigrationHistoryError):
        check_against_base({**base, **branch_only}, base)


def test_rejects_cycle_in_plain_history() -> None:
    with pytest.raises(MigrationHistoryError):
        check_history({"0158": None, "0159": "0158", "0160": "0161", "0161": "0160"})


def test_cli_rejects_self_parented_branch_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import check_migration_heads

    base_dir = tmp_path / "base"
    branch_dir = tmp_path / "branch"
    base_dir.mkdir()
    branch_dir.mkdir()
    for directory in (base_dir, branch_dir):
        _migration(directory / "0158.py", "0158", None)
        _migration(directory / "0159.py", "0159", "0158")
    _migration(branch_dir / "0160.py", "0160", "0160")
    monkeypatch.setattr(
        "sys.argv",
        ["check_migration_heads", str(branch_dir), "--base-versions-dir", str(base_dir)],
    )

    assert check_migration_heads.main() == 1
