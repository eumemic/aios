"""Add ``account_id`` to ``pending_management_calls`` (the table missed by the
multi-tenancy roll-out).

The table was created in 0041, before the ``user_id → account_id`` rename
in 0043 and the corresponding backfill + ``SET NOT NULL`` in 0044. Both
0043 and 0044 omit it from their ``_TABLES`` lists, so the schema has
never carried a tenancy column for it — but ``get_management_call`` and
``mark_management_call_resolved`` were retrofitted with ``WHERE
account_id = $N`` predicates regardless, raising
``UndefinedColumnError`` at runtime on every connector RPC.

This migration closes the gap: add the column, backfill any pre-existing
rows to the root account (mirrors 0044's pattern), ``SET NOT NULL``, add
the FK to ``accounts(id) ON DELETE RESTRICT``, and replace the partial
``(connector, created_at)`` index with the account-aware
``(connector, account_id, created_at)`` form so the SSE-backfill list
query (filtered by ``connector`` and ``account_id``) stays index-served.

Empty-database guard (same shape as 0044): tests run ``alembic upgrade
head`` before any bootstrap. The backfill ``UPDATE`` matches zero rows,
the ``SELECT`` subquery is never evaluated, and ``SET NOT NULL``
succeeds on an empty table. On a populated production database, the
operator must have already bootstrapped a root account before running
this migration — otherwise the backfill leaves ``NULL`` rows and ``SET
NOT NULL`` fails loudly, which is the right behavior.

Revision ID: 0049
Revises: 0048
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0049"
down_revision: str = "0048"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_management_calls ADD COLUMN account_id text")
    # Backfill: any pre-existing rows (none in production yet, but tests
    # and dev DBs may carry some) belong to the root tenant. Same shape
    # as 0044's tenancy backfill.
    op.execute(
        """
        UPDATE pending_management_calls
           SET account_id = (
             SELECT id FROM accounts
              WHERE parent_account_id IS NULL AND archived_at IS NULL
              LIMIT 1
           )
         WHERE account_id IS NULL
        """
    )
    op.execute(
        "ALTER TABLE pending_management_calls ALTER COLUMN account_id SET NOT NULL"
    )
    op.execute(
        """
        ALTER TABLE pending_management_calls
          ADD CONSTRAINT pending_management_calls_account_id_fk
          FOREIGN KEY (account_id) REFERENCES accounts(id)
          ON DELETE RESTRICT
        """
    )
    # Replace the connector-only partial index with the account-aware
    # form so ``list_pending_management_calls_for_connector`` (now
    # filtered by ``WHERE connector = $1 AND account_id = $2``) stays
    # index-served.
    op.execute("DROP INDEX IF EXISTS pending_management_calls_connector_pending_idx")
    op.execute(
        """
        CREATE INDEX pending_management_calls_connector_account_pending_idx
            ON pending_management_calls (connector, account_id, created_at)
            WHERE status = 'pending'
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS pending_management_calls_connector_account_pending_idx"
    )
    op.execute(
        """
        CREATE INDEX pending_management_calls_connector_pending_idx
            ON pending_management_calls (connector, created_at)
            WHERE status = 'pending'
        """
    )
    op.execute(
        "ALTER TABLE pending_management_calls "
        "DROP CONSTRAINT pending_management_calls_account_id_fk"
    )
    op.execute("ALTER TABLE pending_management_calls DROP COLUMN account_id")
