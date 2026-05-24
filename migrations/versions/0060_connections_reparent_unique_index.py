"""Relax the connections active-row UNIQUE from globally-exclusive to per-account.

Pre-this-migration, ``connections_active_external_account_uniq`` was a
partial unique index on ``(connector, external_account_id) WHERE
archived_at IS NULL`` — globally exclusive across tenants, since
real-world messaging identities (Signal phone numbers, Telegram bot
tokens) are universally unique by construction.

jarbot v2 introduces an explicit transfer primitive: moving an
``ExternalIdentity`` (Signal phone, Telegram bot, WhatsApp account)
from one ``Organization`` to another is an atomic API operation, not a
delete+recreate. The new ``POST /v1/connections/{id}/reparent`` endpoint
updates ``connections.account_id`` in place, preserving the
``connection.id`` so the connector daemon's accumulated state
(signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``, telegram
webhook config, etc.) — all keyed by ``connection.id`` — carries over.

The reparent UPDATE is structurally impossible under the global UNIQUE
because every reparent transit would briefly violate it (the source row
and a hypothetical destination row both holding the active identity).
Relaxing to per-account makes the constraint match the new semantic:
*one active connection per ``(account, connector, external_account_id)``*.

Existing data is already unique under the relaxed key (it was unique
under the stricter global key), so the data churn is zero — only the
index swaps.

Revision ID: 0060
Revises: 0059
Create Date: 2026-05-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0060"
down_revision: str = "0059"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS connections_active_external_account_uniq")
    op.execute(
        "CREATE UNIQUE INDEX connections_active_account_external_uniq "
        "ON connections (account_id, connector, external_account_id) "
        "WHERE archived_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS connections_active_account_external_uniq")
    op.execute(
        "CREATE UNIQUE INDEX connections_active_external_account_uniq "
        "ON connections (connector, external_account_id) "
        "WHERE archived_at IS NULL"
    )
