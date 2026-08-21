#!/usr/bin/env python3
"""Fail with an actionable error when Alembic's migration history is branched."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path


class MigrationHistoryError(ValueError):
    """The migration files do not form a single-headed history."""


def _value(module: ast.Module, name: str, path: Path) -> str | None:
    for statement in module.body:
        value: ast.expr | None = None
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name for target in statement.targets
            )
        ) or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            value = statement.value
        if value is not None:
            parsed = ast.literal_eval(value)
            if parsed is None or isinstance(parsed, str):
                return parsed
            raise MigrationHistoryError(f"{path}: {name} must be a string or None")
    raise MigrationHistoryError(f"{path}: missing {name} declaration")


def load_revisions(versions_dir: Path) -> dict[str, str | None]:
    revisions: dict[str, str | None] = {}
    for path in sorted(versions_dir.glob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        revision = _value(module, "revision", path)
        if revision is None:
            raise MigrationHistoryError(f"{path}: revision must be a string")
        if revision in revisions:
            raise MigrationHistoryError(f"duplicate alembic revision: {revision}")
        revisions[revision] = _value(module, "down_revision", path)
    return revisions


def _heads(revisions: dict[str, str | None]) -> list[str]:
    return sorted(set(revisions) - {parent for parent in revisions.values() if parent is not None})


def check_against_base(
    branch_revisions: dict[str, str | None], base_revisions: dict[str, str | None]
) -> str:
    """Check branch migrations individually against the live base history."""
    base_tip = check_history(base_revisions)

    # A branch-only revision cannot legitimize another branch-only revision. Each
    # new migration must have been authored directly against a parent that is
    # independently present on the live base. Checking this before constructing
    # the union prevents a copied/stacked parent from masking the violation.
    for revision in sorted(set(branch_revisions) - set(base_revisions)):
        parent = branch_revisions[revision]
        if parent not in base_revisions:
            raise MigrationHistoryError(
                f'revision {revision} declares down_revision="{parent}", but that parent '
                "is not present on the live base"
            )

    combined = dict(base_revisions)
    for revision, parent in branch_revisions.items():
        if revision in combined and combined[revision] != parent:
            raise MigrationHistoryError(
                f"revision {revision} has different parents on the branch and live base"
            )
        combined[revision] = parent

    heads = _heads(combined)
    if len(heads) != 1:
        raise MigrationHistoryError(
            f"migration branch does not extend the current base head ({base_tip}): "
            f"combined heads are {' and '.join(heads)}\n"
            f"  -> re-parent your migration onto the current base head ({base_tip})"
        )
    return check_history(combined, current_tip=base_tip)


def check_history(revisions: dict[str, str | None], *, current_tip: str | None = None) -> str:
    missing_parents = sorted(
        (revision, parent)
        for revision, parent in revisions.items()
        if parent is not None and parent not in revisions
    )
    if missing_parents:
        revision, parent = missing_parents[0]
        raise MigrationHistoryError(
            f'unknown down_revision: {revision} declares down_revision="{parent}", '
            "but that revision is not present"
        )

    children: dict[str, list[str]] = defaultdict(list)
    # NB: distinct names from the `revision, parent` unpacked above — reusing them
    # rebinds `parent` from `str` to `str | None` and mypy --strict rejects it.
    for child_revision, child_parent in revisions.items():
        if child_parent is not None:
            children[child_parent].append(child_revision)

    collisions = [(parent, sorted(nodes)) for parent, nodes in children.items() if len(nodes) > 1]
    heads = _heads(revisions)
    if len(heads) == 1 and not collisions:
        return heads[0]

    if collisions:
        parent, nodes = collisions[0]
        tip = current_tip if current_tip in nodes else nodes[0]
        others = [node for node in nodes if node != tip]
        named = " and ".join([tip, *others])
        raise MigrationHistoryError(
            f'branched alembic history: {named} both declare down_revision="{parent}"\n'
            f"  -> re-parent your migration onto the current tip ({tip})\n"
            "  -> NOTE: a git rebase moves the file but does NOT re-parent it"
        )

    raise MigrationHistoryError(f"branched alembic history: expected one head, found {heads}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("versions_dir", type=Path, nargs="?", default=Path("migrations/versions"))
    parser.add_argument("--current-tip")
    parser.add_argument(
        "--base-versions-dir",
        type=Path,
        help="live base versions directory to union with the branch before checking",
    )
    args = parser.parse_args()
    try:
        revisions = load_revisions(args.versions_dir)
        if args.base_versions_dir is None:
            head = check_history(revisions, current_tip=args.current_tip)
        else:
            head = check_against_base(revisions, load_revisions(args.base_versions_dir))
    except (MigrationHistoryError, SyntaxError, ValueError) as error:
        print(error)
        return 1
    print(f"alembic history has one head: {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
