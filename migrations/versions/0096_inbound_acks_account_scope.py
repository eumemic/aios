"""Tenant-scope the connector_inbound_acks dedup ledger.

The ledger's PK ``(connector, external_account_id, event_id)`` was
tenant-unique only while migration 0050's global active-unique made
``(connector, external_account_id)`` universally exclusive. Migration 0060
relaxed connections uniqueness to per-account (``account_id, connector,
external_account_id``) to support the #694 reparent primitive, so two
tenants can now hold the same external identity. ``event_id`` is a
deterministic function of the chat namespace (the Telegram connector emits
``telegram-{chat_id}-{message_id}``), not a global ULID — so the two
tenants' dedup keyspaces collide and one tenant's *first-ever* delivery of
an event_id the other already acked is silently swallowed. Add ``account_id``
to the ledger key to restore per-tenant dedup.

Revision ID: 0096
Revises: 0095
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0096"
down_revision: str = "0095"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE connector_inbound_acks ADD COLUMN account_id text")
    # Backfill: attribute each row to the account that holds its identity, but
    # only when exactly one account does. The ledger is a short-lived dedup
    # marker (no retention implemented; it only guards a connector re-emitting
    # the same event_id within a crash-before-ack window), so rows whose
    # identity is claimed by more than one account's connection (archived or
    # active count alike — the join is deliberately archived_at-blind) post-0060,
    # or no longer mapped to any connection, are dropped rather than guessed.
    # Worst case is a single re-delivered inbound re-appending, which the new
    # per-tenant key then dedups going forward.
    op.execute(
        """
        UPDATE connector_inbound_acks a
           SET account_id = c.account_id
          FROM connections c
         WHERE c.connector = a.connector
           AND c.external_account_id = a.external_account_id
           AND NOT EXISTS (
                SELECT 1 FROM connections c2
                 WHERE c2.connector = a.connector
                   AND c2.external_account_id = a.external_account_id
                   AND c2.account_id <> c.account_id
           )
        """
    )
    op.execute("DELETE FROM connector_inbound_acks WHERE account_id IS NULL")
    op.execute("ALTER TABLE connector_inbound_acks ALTER COLUMN account_id SET NOT NULL")
    op.execute("ALTER TABLE connector_inbound_acks DROP CONSTRAINT connector_inbound_acks_pkey")
    op.execute(
        "ALTER TABLE connector_inbound_acks "
        "ADD PRIMARY KEY (account_id, connector, external_account_id, event_id)"
    )


def downgrade() -> None:
    # Lossy by nature: the per-account ledger may hold rows that collide under
    # the old 3-tuple key (two tenants, same identity + event_id). Keep one
    # arbitrary row per old key before restoring the narrower PK.
    op.execute(
        """
        DELETE FROM connector_inbound_acks a
         USING connector_inbound_acks b
         WHERE a.ctid < b.ctid
           AND a.connector = b.connector
           AND a.external_account_id = b.external_account_id
           AND a.event_id = b.event_id
        """
    )
    op.execute("ALTER TABLE connector_inbound_acks DROP CONSTRAINT connector_inbound_acks_pkey")
    op.execute(
        "ALTER TABLE connector_inbound_acks "
        "ADD PRIMARY KEY (connector, external_account_id, event_id)"
    )
    op.execute("ALTER TABLE connector_inbound_acks DROP COLUMN account_id")
