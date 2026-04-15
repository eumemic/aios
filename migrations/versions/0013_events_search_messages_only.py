"""Restrict events_search view to message events only.

The view previously exposed all event kinds (lifecycle, span, interrupt)
which are harness internals the agent should never see.  Filter to
``kind = 'message'`` so the agent only searches its own conversation
history.

The ``kind`` column is dropped from the view since it is now constant.

Revision ID: 0013
Revises: 0012
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP VIEW IF EXISTS events_search")
    op.execute("""
        CREATE VIEW events_search AS
        SELECT
            id,
            seq,
            data->>'role' AS role,
            created_at,
            COALESCE(data->>'content', data::text) AS content_text
        FROM events
        WHERE session_id = current_setting('app.session_id', true)
          AND kind = 'message'
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS events_search")
    op.execute("""
        CREATE VIEW events_search AS
        SELECT
            id,
            seq,
            kind,
            data->>'role' AS role,
            created_at,
            COALESCE(data->>'content', data::text) AS content_text
        FROM events
        WHERE session_id = current_setting('app.session_id', true)
    """)
