from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_migration_heads import MigrationHistoryError, check_history, load_revisions


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


def test_known_good_repository_has_expected_revision_count_and_one_head() -> None:
    versions = Path(__file__).resolve().parents[2] / "migrations" / "versions"
    revisions = load_revisions(versions)

    # Bump this when adding a migration — the count pins accidental
    # deletions/duplications, not a maximum.
    assert len(revisions) == 159
    assert check_history(revisions) in revisions
