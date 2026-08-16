"""Persist worker-observed session egress interception state.

Revision ID: 0160
Revises: 0159
"""

from alembic import op

revision = "0160"
down_revision = "0159"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE session_egress_states (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            hosts JSONB NOT NULL,
            provisioned_at TIMESTAMPTZ NOT NULL,
            sandbox_generation BIGINT NOT NULL
        )
    """)


def downgrade() -> None:
    op.drop_table("session_egress_states")
