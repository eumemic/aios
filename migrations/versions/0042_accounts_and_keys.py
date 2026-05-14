"""Add ``accounts`` + ``account_keys`` tables.

``accounts`` is the hierarchical tenant. Each row points to its parent
via ``parent_account_id`` (NULL for the singular root). ``can_mint_children``
gates the create-child management verbs. Soft-archived via ``archived_at``.

``account_keys`` holds bearer API keys, hashed at rest. Plaintext is
returned exactly once at mint time; multiple active keys per account
are allowed so rotation is mint-new + switch-callers + revoke-old.

Revision ID: 0042
Revises: 0041
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0042"
down_revision: str = "0041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE accounts (
            id                  text PRIMARY KEY,
            parent_account_id   text REFERENCES accounts(id) ON DELETE RESTRICT,
            can_mint_children   boolean     NOT NULL DEFAULT FALSE,
            display_name        text        NOT NULL,
            metadata            jsonb       NOT NULL DEFAULT '{}'::jsonb,
            archived_at         timestamptz,
            created_at          timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX accounts_sibling_name_uniq
        ON accounts (parent_account_id, display_name)
        WHERE archived_at IS NULL
        """
    )

    # Regular UNIQUE treats NULLs as distinct, so a plain
    # ``UNIQUE (parent_account_id, display_name)`` would let two roots
    # share a name (both rows have ``parent_account_id IS NULL``).
    # Partial unique on display_name where the parent is NULL fixes it.
    op.execute(
        """
        CREATE UNIQUE INDEX accounts_root_name_uniq
        ON accounts (display_name)
        WHERE parent_account_id IS NULL AND archived_at IS NULL
        """
    )

    # ``((TRUE))`` makes every matching row share the same index key,
    # so a second active root violates uniqueness — cleaner than a
    # CHECK + trigger for "at most one active root."
    op.execute(
        """
        CREATE UNIQUE INDEX accounts_one_active_root
        ON accounts ((TRUE))
        WHERE parent_account_id IS NULL AND archived_at IS NULL
        """
    )

    op.execute(
        """
        CREATE INDEX accounts_parent_idx
        ON accounts (parent_account_id)
        WHERE archived_at IS NULL
        """
    )

    op.execute(
        """
        CREATE TABLE account_keys (
            key_id          text PRIMARY KEY,
            account_id      text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            hash            bytea       NOT NULL,
            label           text        NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now(),
            last_used_at    timestamptz,
            revoked_at      timestamptz,
            UNIQUE (hash)
        )
        """
    )

    op.execute(
        """
        CREATE INDEX account_keys_account_active_idx
        ON account_keys (account_id)
        WHERE revoked_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS account_keys")
    op.execute("DROP TABLE IF EXISTS accounts")
