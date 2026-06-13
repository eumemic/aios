"""Drop the orphaned legacy ``credentials`` table.

``credentials`` (migration 0001) was the pre-multi-tenancy, pre-vaults
credential store. Migration 0005 introduced ``vaults``/``vault_credentials``
and dropped the ``credential_id`` foreign-key columns from ``agents`` and
``agent_versions``, leaving ``credentials`` with no incoming references. No
code reads or writes it — it has been dead schema since 0005. Drop it.

This is window-safe despite being a destructive ``DROP TABLE``: the running
code (old or new) never touches ``credentials``, so the new-code/old-schema
window during a post-deploy migrate has nothing to break. (Operators should
confirm no un-migrated rows remain in ``credentials`` before deploying —
this migration discards the table and any rows.)

Revision ID: 0099
Revises: 0098
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0099"
down_revision: str = "0098"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS credentials")


def downgrade() -> None:
    # Recreate the orphaned table shape from migration 0001 (data-lossy: the
    # dropped rows are not reconstituted, matching the one-way posture of the
    # other legacy-table drops in this ladder). Nothing references it.
    op.execute(
        """
        CREATE TABLE credentials (
            id           text PRIMARY KEY,
            name         text NOT NULL,
            provider     text NOT NULL,
            ciphertext   bytea NOT NULL,
            nonce        bytea NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now(),
            archived_at  timestamptz
        )
        """
    )
