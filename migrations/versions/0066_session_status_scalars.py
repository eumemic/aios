"""Add five monotonic scalar columns to ``sessions`` for O(1) status derivation.

Replaces the O(n) correlated-subquery status derivation (``_SESSION_STATUS_EXPR``,
``_SESSION_ERRORED_EXPR``, ``_SESSION_ACTIVE_EXPR``) with pure column arithmetic.
The five columns — ``last_reacted_seq``, ``open_tool_call_count``,
``last_error_seq``, ``last_user_seq``, ``last_stimulus_seq`` — are maintained
transactionally inside ``append_event`` and backfilled from the event log for
existing sessions.

``last_stimulus_seq`` is the max ``seq`` of *stimulus* events the assistant
must react to: ``kind = 'message' AND role <> 'assistant'`` (user + tool
messages). It is NOT the same as ``last_user_seq`` (user-only, the error
latch): the active predicate must include unreacted tool results, the error
latch must not. Critically, the active predicate compares against
``last_stimulus_seq``, NOT ``last_event_seq`` — the latter includes the
session's own assistant replies, so a normal idle session (user → assistant
reply, no tool calls) would have ``last_event_seq > last_reacted_seq`` and read
wrongly as ``active``, driving one extra model step (#749 regression). This is
exactly the pre-#732 ``EXISTS(non-assistant message with seq >
last_reacted_seq)`` derivation, expressed as a scalar.

Status predicate (pure column arithmetic):
  errored = last_error_seq > 0 AND last_error_seq > last_user_seq
  active  = (last_stimulus_seq > last_reacted_seq OR open_tool_call_count > 0)
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
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN IF NOT EXISTS last_stimulus_seq BIGINT NOT NULL DEFAULT 0"
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

    # ── backfill last_stimulus_seq ───────────────────────────────────────
    # Max seq of the stimuli the assistant must react to: message events
    # whose role is NOT 'assistant' (user + tool). Excludes the session's
    # own assistant replies — that exclusion is the whole point of the
    # column (see module docstring).
    op.execute(
        """
        UPDATE sessions s
           SET last_stimulus_seq = COALESCE((
               SELECT MAX(e.seq) FROM events e
                WHERE e.session_id = s.id
                  AND e.kind = 'message' AND e.role <> 'assistant'
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
    # The reaction watermark: MAX(COALESCE(reacting_to, seq)) over assistant
    # messages — exactly the pre-#732 ``session_max_reacting`` CTE. NOT bumped
    # by ``turn_ended`` lifecycle events: a rescheduling ``turn_ended`` has no
    # assistant reaction, and folding it in would falsely mark an unreacted
    # user message as reacted-to (flipping a retry-pending session to idle).
    op.execute(
        """
        UPDATE sessions s
           SET last_reacted_seq = COALESCE((
               SELECT MAX(COALESCE((e.data->>'reacting_to')::bigint, e.seq))
                 FROM events e
                WHERE e.session_id = s.id
                  AND e.kind = 'message' AND e.role = 'assistant'
           ), 0)
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
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_stimulus_seq")
