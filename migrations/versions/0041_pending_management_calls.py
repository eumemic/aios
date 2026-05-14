"""Pending management calls — operator→connector RPC plane.

Resolved rows are kept for audit; a TTL cleanup is a follow-up.

Revision ID: 0041
Revises: 0040
Create Date: 2026-05-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0041"
down_revision: str = "0040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE pending_management_calls (
            id           text PRIMARY KEY,
            connector    text NOT NULL,
            method       text NOT NULL,
            params       jsonb NOT NULL,
            status       text NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'succeeded', 'failed')),
            result       jsonb,
            is_error     boolean NOT NULL DEFAULT false,
            created_at   timestamptz NOT NULL DEFAULT now(),
            expires_at   timestamptz NOT NULL,
            resolved_at  timestamptz
        )
        """
    )
    # Partial index supports list_pending_management_calls_for_connector
    # (SSE backfill on connector reconnect): the connector type is a
    # high-cardinality filter, and only 'pending' rows ever need
    # listing — resolved rows stay for audit but are never re-dispatched.
    op.execute(
        """
        CREATE INDEX pending_management_calls_connector_pending_idx
            ON pending_management_calls (connector, created_at)
            WHERE status = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS pending_management_calls_connector_pending_idx")
    op.execute("DROP TABLE IF EXISTS pending_management_calls")
