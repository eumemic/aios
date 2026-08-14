"""Reparent root's stranded pre-existing children beneath the Eumemic child.

Migration 0154 created the Eumemic child under the platform root and moved
every root-owned account-scoped row (including ``model_providers``) onto it,
but it did **not** reparent the root's already-existing child accounts.  On
every database that applied 0154 *as originally written* those children are
now siblings of Eumemic rather than descendants of it.  Provider resolution
is a nearest-ancestor walk (``resolve_model_provider``), so they walk up to a
root that owns zero providers and fail closed with
``model_provider_not_configured`` (aios#2060, aios#1969).

0154 has already been applied in production, and alembic records applied
revisions by id -- it will never re-run.  Correcting 0154 in place therefore
fixes fresh databases only.  **This forward migration is what repairs an
already-migrated database.**

Safety properties:

* **Idempotent.**  The reparenting ``UPDATE`` is defined by the predicate it
  eliminates (``parent_account_id = root AND id <> child``), so a second run
  matches zero rows.
* **Harmless where the defect never existed.**  A database migrated by the
  corrected 0154 -- or any database with no root, or no migration-owned
  Eumemic child -- finds nothing to move and returns without writing.
* **Resolution can only improve.**  Eumemic's own parent is the root, so
  moving an account from ``root`` to ``child`` strictly *grows* its ancestor
  set (it gains Eumemic and keeps root).  No account can lose a provider it
  could previously resolve; the migration asserts this from the live tree
  rather than trusting the argument.

Revision ID: 0160
Revises: 0159
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0160"
down_revision: str = "0159"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.migration.0160")

_CHILD_NAME = "Eumemic"
# 0154 tags the account it creates with this revoked bootstrap key.  Matching
# on it (not on the display name alone) means an operator-created account that
# merely happens to be called "Eumemic" is never mistaken for the migration's.
_MARKER_LABEL = "0154 bootstrap (revoked)"

_RESOLVED_ACCOUNTS = """
    WITH RECURSIVE ancestry(account_id, ancestor_id) AS (
        SELECT id, id FROM accounts WHERE archived_at IS NULL
        UNION ALL
        SELECT ancestry.account_id, accounts.parent_account_id
        FROM ancestry
        JOIN accounts ON accounts.id = ancestry.ancestor_id
        WHERE accounts.parent_account_id IS NOT NULL
    )
    SELECT DISTINCT ancestry.account_id
    FROM ancestry
    JOIN model_providers ON model_providers.account_id = ancestry.ancestor_id
    WHERE model_providers.archived_at IS NULL
"""


def _root_and_marked_child() -> tuple[str | None, str | None]:
    """Locate the live root and the 0154-created Eumemic child, if both exist."""
    bind = op.get_bind()
    root = bind.execute(
        sa.text(
            "SELECT id FROM accounts WHERE parent_account_id IS NULL AND archived_at IS NULL "
            "FOR UPDATE"
        )
    ).scalar_one_or_none()
    if root is None:
        # Fresh database (tests, new installs): 0154 was a structural no-op.
        return None, None
    child = bind.execute(
        sa.text(
            "SELECT accounts.id FROM accounts "
            "JOIN account_keys ON account_keys.account_id = accounts.id "
            "WHERE accounts.parent_account_id = :root "
            "  AND accounts.display_name = :name "
            "  AND accounts.archived_at IS NULL "
            "  AND account_keys.label = :label "
            "  AND account_keys.revoked_at IS NOT NULL "
            "FOR UPDATE OF accounts"
        ),
        {"root": root, "name": _CHILD_NAME, "label": _MARKER_LABEL},
    ).scalar_one_or_none()
    return root, child


def _assert_no_name_collisions(root: str, child: str) -> None:
    """Refuse rather than trip ``accounts_sibling_name_uniq`` mid-move.

    Accounts minted under Eumemic since the cutover could share a display name
    with a stranded sibling still under root.  The partial unique index would
    abort the transaction with a constraint error; failing here names the
    conflict instead.
    """
    conflicts = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT stranded.id || ' (' || stranded.display_name || ')' "
                "FROM accounts stranded "
                "JOIN accounts existing "
                "  ON existing.parent_account_id = :child "
                " AND existing.archived_at IS NULL "
                " AND existing.display_name = stranded.display_name "
                "WHERE stranded.parent_account_id = :root "
                "  AND stranded.id <> :child "
                "  AND stranded.archived_at IS NULL "
                "ORDER BY 1"
            ),
            {"root": root, "child": child},
        )
        .scalars()
        .all()
    )
    if conflicts:
        raise RuntimeError(
            "migration 0160 cannot reparent: display-name collision under "
            f"{child} for: " + ", ".join(conflicts) + " -- rename the conflicting "
            "account(s) and re-run"
        )


def _snapshot_resolution() -> None:
    op.get_bind().execute(
        sa.text("CREATE TEMP TABLE migration_0160_resolved ON COMMIT DROP AS " + _RESOLVED_ACCOUNTS)
    )


def _assert_resolution_not_regressed() -> None:
    """Derive the invariant from the live tree, not from a fixture list."""
    stranded = (
        op.get_bind()
        .execute(
            sa.text(
                "WITH resolved AS (" + _RESOLVED_ACCOUNTS + ") "
                "SELECT snapshot.account_id "
                "FROM migration_0160_resolved snapshot "
                "LEFT JOIN resolved USING (account_id) "
                "WHERE resolved.account_id IS NULL "
                "ORDER BY snapshot.account_id"
            )
        )
        .scalars()
        .all()
    )
    if stranded:
        raise RuntimeError(
            "migration 0160 would strand provider resolution for accounts: " + ", ".join(stranded)
        )


def upgrade() -> None:
    root, child = _root_and_marked_child()
    if root is None or child is None:
        # No root, or no migration-owned Eumemic child: nothing 0154 could
        # have stranded on this database.
        return
    _assert_no_name_collisions(root, child)
    _snapshot_resolution()
    moved = (
        op.get_bind()
        .execute(
            sa.text(
                "UPDATE accounts SET parent_account_id = :child "
                "WHERE parent_account_id = :root AND id <> :child "
                "RETURNING id"
            ),
            {"root": root, "child": child},
        )
        .scalars()
        .all()
    )
    _assert_resolution_not_regressed()
    if moved:
        logger.warning(
            "migration 0160 reparented %d account(s) stranded under the platform root "
            "by migration 0154 beneath %s: %s",
            len(moved),
            child,
            ", ".join(sorted(moved)),
        )
    else:
        logger.info("migration 0160: no accounts stranded under the platform root; no-op")


def downgrade() -> None:
    """Intentionally a no-op.

    0160 corrects data that 0154 should have written; it does not own a
    schema change to reverse.  The set of accounts it moved is not recoverable
    after the fact (accounts legitimately minted under Eumemic since the
    cutover are indistinguishable from reparented ones), so blindly moving
    every Eumemic child back to the root would corrupt the tree rather than
    restore it.  0154's own ``downgrade`` already returns the whole fleet --
    including this topology -- to the root.
    """
    return
