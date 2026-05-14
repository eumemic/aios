"""Pending management calls — operator→connector RPC plane (#348).

The api process posts a row here when an operator hits one of the
``/v1/connectors/<connector>/<method>`` routes (register, verify,
updateProfile for signal); the runtime container subscribes via
``GET /v1/connectors/runtime/management-calls`` SSE, dispatches the
call, and POSTs the result back to
``/v1/connectors/runtime/management-call-results``.  The api process
LISTENs on ``connector_result_<call_id>`` (existing primitive in
``db/listen.py``) for the wakeup.

Rows are NOT deleted on resolve — a small audit trail for botched
registrations.  A cleanup job for ``resolved_at IS NOT NULL AND
resolved_at < now() - interval '30 days'`` is a follow-up.

Revision ID: 0040
Revises: 0039
Create Date: 2026-05-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0040"
down_revision: str = "0039"
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
