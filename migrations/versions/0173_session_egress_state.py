"""Persist worker-observed session egress interception state.

Revision ID: 0173
Revises: 0172
"""

from alembic import op

revision = "0173"
down_revision = "0172"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ``hosts`` is deliberately NULLABLE. NULL is the INVALIDATED tombstone:
    # "this session's intercept set could not be observed", which reads back
    # through the same ``NotFoundError`` contract as a missing row. It is NOT
    # the same as ``[]`` — an empty array is an affirmative "nothing is
    # intercepted", the false all-clear this endpoint exists to prevent, and
    # may only ever be published from a real observation.
    #
    # Invalidation tombstones rather than DELETEs so ``sandbox_generation``
    # stays MONOTONIC: a DELETE would restart the counter at 1 on the next
    # stamp, making a diagnostic field on a diagnostic endpoint lie.
    op.execute("""
        CREATE TABLE session_egress_states (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            hosts JSONB,
            provisioned_at TIMESTAMPTZ NOT NULL,
            sandbox_generation BIGINT NOT NULL
        )
    """)


def downgrade() -> None:
    op.drop_table("session_egress_states")
