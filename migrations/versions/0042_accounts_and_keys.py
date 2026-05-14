"""Multi-tenancy v1 PR 1: ``accounts`` + ``account_keys`` tables.

First migration of the multi-tenancy stack (issue #367). Adds the two
tables that everything else hangs off:

* ``accounts`` — the hierarchical tenant. Tree via ``parent_account_id``;
  one capability flag (``can_mint_children``); soft-archived via
  ``archived_at``. Three partial-unique indexes enforce: (a) sibling
  display names are unique within a parent, (b) root display names are
  unique among roots (because ``NULL != NULL`` in regular UNIQUE), and
  (c) at most one active root can exist at a time.
* ``account_keys`` — bearer API keys. Plaintext returned once at mint;
  stored as ``sha256`` hash. Multiple active keys per account allowed
  (zero-downtime rotation). ``ON DELETE CASCADE`` so a hard ``purge``
  of an account sweeps its keys with it.

No resource table gets touched here — PR 4 in the stack will add the
``account_id`` columns and FKs. PR 2 changes the auth dep to return
``(account_id, key_id, can_mint_children)``; PR 1 just lays the
foundation.

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

    # Sibling-unique display names: two active accounts under the same
    # parent can't share a name.  Partial on ``archived_at IS NULL`` so a
    # soft-deleted name is immediately reusable.
    op.execute(
        """
        CREATE UNIQUE INDEX accounts_sibling_name_uniq
        ON accounts (parent_account_id, display_name)
        WHERE archived_at IS NULL
        """
    )

    # Root display names: regular UNIQUE treats NULLs as distinct, so a
    # plain ``UNIQUE (parent_account_id, display_name)`` would let two
    # roots share a name (both rows have ``parent_account_id IS NULL``).
    # Partial unique on display_name where the parent is NULL fixes it.
    op.execute(
        """
        CREATE UNIQUE INDEX accounts_root_name_uniq
        ON accounts (display_name)
        WHERE parent_account_id IS NULL AND archived_at IS NULL
        """
    )

    # At most one active root: an expression index on a constant value
    # makes every matching row share the same key, so a second active
    # root violates uniqueness.  Cleaner than a CHECK + trigger.
    op.execute(
        """
        CREATE UNIQUE INDEX accounts_one_active_root
        ON accounts ((TRUE))
        WHERE parent_account_id IS NULL AND archived_at IS NULL
        """
    )

    # Fast lookup of a parent's direct children (the management
    # endpoint that lists children, the cascade-archive walk).
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

    # The auth dep looks up tokens by hash; the UNIQUE constraint above
    # already gives a B-tree on hash.  We also need a fast "active keys
    # for this account" lookup for the list-keys endpoint.
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
