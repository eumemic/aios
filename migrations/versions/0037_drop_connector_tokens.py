"""Drop the ``connector_tokens`` table (#328 PR 7).

The legacy per-connection bearer-token table backed ``ConnectorAuthDep``
and the ``/v1/connectors/{inbound,tool-results,calls,secrets,tools}``
route family.  PR 5 introduced ``runtime_tokens`` as the per-connector-type
successor; PR 7 deleted the legacy routes + auth dep + ``connector_tokens``
service / model / CLI.  This migration deletes the now-readerless table.

Revision ID: 0037
Revises: 0036
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0037"
down_revision: str = "0036"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS connector_tokens")


def downgrade() -> None:
    # Reconstituting an empty table is fine; the hashes inside are lost
    # but no production data depends on them being recovered (runtime
    # tokens succeeded them in PR 5).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS connector_tokens (
            id             text PRIMARY KEY,
            connection_id  text NOT NULL REFERENCES connections(id),
            label          text,
            token_hash     text NOT NULL UNIQUE,
            created_at     timestamptz NOT NULL DEFAULT now(),
            last_used_at   timestamptz,
            revoked_at     timestamptz
        )
        """
    )
