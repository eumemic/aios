"""Add cumulative_tokens column for O(1) context windowing.

The ``cumulative_tokens`` column stores the running total of approximate
token counts through each message event (in seq order within a session).
This lets the read path compute the chunked-window boundary from a single
index seek and load only the windowed tail, eliminating the O(N) full-scan
that previously loaded every message event.

Non-message events (lifecycle, span, interrupt) get ``NULL``.

Revision ID: 0012
Revises: 0011
"""

from alembic import op

revision = "0012"
down_revision = "0011"


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN cumulative_tokens bigint;")
    op.execute(
        "CREATE INDEX events_session_cumtokens_idx "
        "ON events (session_id, cumulative_tokens) "
        "WHERE kind = 'message' AND cumulative_tokens IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS events_session_cumtokens_idx;")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS cumulative_tokens;")
