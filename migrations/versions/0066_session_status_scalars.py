"""Add four monotonic scalar columns to ``sessions`` for O(1) status derivation.

Replaces the O(n) correlated-subquery status derivation (``_SESSION_STATUS_EXPR``,
``_SESSION_ERRORED_EXPR``, ``_SESSION_ACTIVE_EXPR``) with pure column arithmetic.
The four columns — ``last_reacted_seq``, ``open_tool_call_count``,
``last_error_seq``, ``last_user_seq`` — are maintained transactionally inside
``append_event`` and backfilled from the event log for existing sessions.

Status predicate (pure column arithmetic):
  errored = last_error_seq > 0 AND last_error_seq > last_user_seq
  active  = (last_event_seq > last_reacted_seq OR open_tool_call_count > 0)
            AND NOT errored
  idle    = otherwise

Revision ID: 0066
Revises: 0065
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0066"
down_revision: str = "0065"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── add columns ──────────────────────────────────────────────────────
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN IF NOT EXISTS last_reacted_seq BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN IF NOT EXISTS open_tool_call_count INT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN IF NOT EXISTS last_error_seq BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN IF NOT EXISTS last_user_seq BIGINT NOT NULL DEFAULT 0"
    )

    # ── backfill last_user_seq ───────────────────────────────────────────
    op.execute(
        """
        UPDATE sessions s
           SET last_user_seq = COALESCE((
               SELECT MAX(e.seq) FROM events e
                WHERE e.session_id = s.id
                  AND e.kind = 'message' AND e.role = 'user'
           ), 0)
        """
    )

    # ── backfill last_error_seq ──────────────────────────────────────────
    op.execute(
        """
        UPDATE sessions s
           SET last_error_seq = COALESCE((
               SELECT MAX(e.seq) FROM events e
                WHERE e.session_id = s.id
                  AND e.kind = 'lifecycle' AND e.data->>'stop_reason' = 'error'
           ), 0)
        """
    )

    # ── backfill last_reacted_seq ────────────────────────────────────────
    # GREATEST of: max assistant reacting_to, and max turn_ended seq.
    op.execute(
        """
        UPDATE sessions s
           SET last_reacted_seq = GREATEST(
               COALESCE((
                   SELECT MAX(COALESCE((e.data->>'reacting_to')::bigint, e.seq))
                     FROM events e
                    WHERE e.session_id = s.id
                      AND e.kind = 'message' AND e.role = 'assistant'
               ), 0),
               COALESCE((
                   SELECT MAX(e.seq) FROM events e
                    WHERE e.session_id = s.id
                      AND e.kind = 'lifecycle' AND e.data->>'event' = 'turn_ended'
               ), 0)
           )
        """
    )

    # ── backfill open_tool_call_count ────────────────────────────────────
    # Count tool_call_ids from assistant messages that have no paired
    # tool-role result event, using per-tool_call_id anti-join.
    op.execute(
        """
        UPDATE sessions s
           SET open_tool_call_count = COALESCE((
               SELECT COUNT(*)
                 FROM events ate
                CROSS JOIN LATERAL jsonb_array_elements(ate.data->'tool_calls') tc
                WHERE ate.session_id = s.id
                  AND ate.kind = 'message' AND ate.role = 'assistant'
                  AND ate.data ? 'tool_calls'
                  AND NOT EXISTS (
                      SELECT 1 FROM events tr
                       WHERE tr.session_id = s.id
                         AND tr.kind = 'message' AND tr.role = 'tool'
                         AND tr.data->>'tool_call_id' = tc->>'id'
                  )
           ), 0)
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_reacted_seq")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS open_tool_call_count")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_error_seq")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_user_seq")
