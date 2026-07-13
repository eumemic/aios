"""Index rolling per-session outbound tool quota counts.

Revision ID: 0148
Revises: 0147
"""

from alembic import op

revision = "0148"
down_revision = "0147"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX events_session_tool_name_created_at_idx "
        "ON events (session_id, tool_name, created_at) "
        "WHERE tool_name IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX events_session_tool_name_created_at_idx")
