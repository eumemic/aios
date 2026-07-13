"""Add session-led partial index for the per-agent inbound budget.

Revision ID: 0148
Revises: 0147
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0148"
down_revision: str = "0147"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_agent_inbound_budget_idx "
            "ON events (session_id, created_at) "
            "WHERE (kind = 'message' AND data->>'role' = 'user') "
            "OR (kind = 'lifecycle' AND (data->>'wake')::boolean IS TRUE)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_agent_inbound_budget_idx")
