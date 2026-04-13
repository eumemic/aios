"""Events search view for the search_events built-in tool.

Creates a ``events_search`` view that exposes the event log in a
model-friendly shape, scoped to the current session via
``current_setting('app.session_id')``. The tool handler sets
``SET LOCAL app.session_id = <id>`` inside a read-only transaction
so the agent can only see its own events.

Columns:
- ``id`` (text): event ID
- ``seq`` (bigint): gapless sequence number within the session
- ``kind`` (text): event kind (message, lifecycle, span, interrupt)
- ``role`` (text): message role (user, assistant, tool) — NULL for
  non-message events
- ``created_at`` (timestamptz): when the event was appended
- ``content_text`` (text): human-readable content; extracts
  ``data->>'content'`` for messages, falls back to ``data::text``
  for other event kinds

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE VIEW events_search AS
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


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS events_search")
