"""Bound context lifecycle reads to model-visible event kinds.

The broad lifecycle index also accumulates quiet, never-pruned trigger receipts.
This expression/partial index keeps both windowed and full-load (drop=None)
context reads proportional to the static model-visible allowlist instead.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0145"
down_revision: str | None = "0144"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must remain identical to MODEL_VISIBLE_LIFECYCLE_EVENTS.  SQL migrations
# intentionally cannot import application code.
_MODEL_VISIBLE = (
    "connector_delivery_failed",
    "connector_message_delivered",
    "connector_message_edited",
    "sandbox_fs_expired",
    "sandbox_fs_over_limit",
    "sandbox_fs_reset",
)


def upgrade() -> None:
    values = ", ".join(f"'{value}'" for value in _MODEL_VISIBLE)
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "events_session_model_visible_lifecycle_seq_idx "
            "ON events (session_id, seq) "
            "WHERE kind = 'lifecycle' AND data->>'event' IN (" + values + ")"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS events_session_model_visible_lifecycle_seq_idx"
        )
