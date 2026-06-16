"""Rename ``connections.account`` (and the sibling dedup-ledger column) to
``external_account_id`` and reindex the partial unique constraint under a
clearer name.

Predates multi-tenancy; after ``account_id`` arrived as the tenant FK,
``account`` no longer reads as the globally-exclusive external identity
it actually is. The same column lives on ``connector_inbound_acks`` as
part of the dedup-ledger PK — renamed in lock-step so call sites that
pass ``connection.external_account_id`` keep their parameter name
aligned with the column they write to.

Wire-format change with no deprecation shim. Connector runtime
containers must redeploy in lock-step (the SDK's focal-kwarg injection
now passes ``external_account_id`` rather than ``account``).

Revision ID: 0050
Revises: 0049
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0050"
down_revision: str = "0049"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Rename the column on ``connections``. ``RENAME COLUMN`` is a
    # metadata-only catalog update — no row rewrite, indexes referencing
    # the column track it by attnum.
    op.execute("ALTER TABLE connections RENAME COLUMN account TO external_account_id")
    # The partial unique index still functions (Postgres tracks columns
    # by attnum, not name), but its name embeds the old column word.
    # Drop+recreate under the new name so ``\d connections`` and grep
    # both surface the right relationship.
    op.execute("DROP INDEX IF EXISTS connections_active_account_uniq")
    op.execute(
        "CREATE UNIQUE INDEX connections_active_external_account_uniq "
        "ON connections (connector, external_account_id) WHERE archived_at IS NULL"
    )

    # 2. Same rename on the dedup ledger. The PK ``(connector, account,
    # event_id)`` becomes ``(connector, external_account_id, event_id)``
    # — the PK constraint name (``connector_inbound_acks_pkey``) is
    # column-list-bound, not column-name-bound, so no rename needed.
    op.execute("ALTER TABLE connector_inbound_acks RENAME COLUMN account TO external_account_id")


def downgrade() -> None:
    # Rename the connections column first, then swap the index — keeping
    # *some* partial-unique index on the renamed column for as much of
    # the migration window as the engine allows. Postgres tracks index
    # columns by attnum so the existing index keeps working through the
    # rename; DROP+CREATE under the new name closes the cycle.
    op.execute("ALTER TABLE connections RENAME COLUMN external_account_id TO account")
    op.execute("DROP INDEX IF EXISTS connections_active_external_account_uniq")
    op.execute(
        "CREATE UNIQUE INDEX connections_active_account_uniq "
        "ON connections (connector, account) WHERE archived_at IS NULL"
    )
    op.execute("ALTER TABLE connector_inbound_acks RENAME COLUMN external_account_id TO account")
