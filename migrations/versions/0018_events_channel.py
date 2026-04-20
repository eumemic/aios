"""Derived ``events.channel`` column — "which channel does this event
belong to?" — stamped at append time.

For user events, ``channel = orig_channel`` (where the inbound came from).
For assistant events, ``channel = focal_channel_at_arrival`` (where the
agent was focused when it spoke).  For tool events, ``channel`` is the
parent assistant's ``focal_channel_at_arrival`` — NOT the live focal
when the tool result arrived, because a tool call started in A and
completed after a switch to B conceptually belongs to A.

Collapses the role-specific predicate branches in the recap renderer
(and anywhere else that filters events by channel) into a single
``event.channel == target`` comparison.  ``focal_channel_at_arrival`` is
retained — it's still load-bearing for ``render_user_event``'s
full-vs-notification decision and for ``derive_last_seen`` /
``derive_unread_counts``.

Revision ID: 0018
Revises: 0017
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0018"
down_revision: str = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN channel text")
    # Backfill existing rows.  Tool events look up their parent assistant
    # via JSONB containment on tool_calls[].id — one correlated subquery
    # per tool row, scoped to the same session, ordered by seq DESC so we
    # pick the most recent parent if a tool_call_id somehow repeats.
    op.execute(
        """
        UPDATE events e SET channel =
          CASE
            WHEN e.kind = 'message' AND e.data->>'role' = 'user'
              THEN e.orig_channel
            WHEN e.kind = 'message' AND e.data->>'role' = 'assistant'
              THEN e.focal_channel_at_arrival
            WHEN e.kind = 'message' AND e.data->>'role' = 'tool'
              THEN (
                SELECT a.focal_channel_at_arrival FROM events a
                WHERE a.session_id = e.session_id
                  AND a.kind = 'message'
                  AND a.data->'tool_calls' @> jsonb_build_array(
                    jsonb_build_object('id', e.data->>'tool_call_id'))
                ORDER BY a.seq DESC LIMIT 1)
            ELSE NULL
          END
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS channel")
