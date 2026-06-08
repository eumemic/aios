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
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_reacted_seq BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS open_tool_call_count INT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_error_seq BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_user_seq BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_stimulus_seq BIGINT NOT NULL DEFAULT 0"
    )

    # ── backfill all five scalars: ONE set-based statement ───────────────
    #
    # This MUST be a single set-based pass, NOT five correlated per-session
    # subqueries. The obvious correlated form (``SELECT MAX(...) WHERE
    # e.session_id = s.id`` per column, one of them a ``COUNT(*)`` over
    # ``jsonb_array_elements`` with a ``NOT EXISTS`` anti-join) re-runs the
    # exact O(session-size) event-log scan this very feature exists to
    # eliminate — once per session, under ``ACCESS EXCLUSIVE``. On prod it
    # hung for minutes on a 320k-event session and took ``/v1/sessions`` down
    # during deploy (#748/#749 reintroduced as a one-time migration cost).
    # CI never caught it because testcontainer DBs are tiny. The set-based
    # form below scans ``events`` a handful of times TOTAL (each CTE is one
    # grouped aggregate), independent of any single session's size.
    #
    # Each CTE is the authoritative definition of one scalar, matched to how
    # ``append_event`` maintains it forward (src/aios/db/queries/__init__.py):
    #
    #   last_user_seq      MAX(seq) over user messages (the error-clear latch)
    #   last_stimulus_seq  MAX(seq) over NON-assistant messages (user + tool);
    #                      the unreacted-stimulus watermark that drives the
    #                      active predicate — see the module docstring on why
    #                      this excludes the session's own assistant replies
    #   last_error_seq     MAX(seq) over error lifecycle events
    #   last_reacted_seq   MAX(COALESCE(reacting_to, seq)) over assistant
    #                      messages — the pre-#732 ``session_max_reacting``
    #                      CTE; turn_ended lifecycle events do NOT advance it
    #   open_tool_call_count  assistant tool_call ids with no paired tool-role
    #                      result, per-id anti-join. The ``tool_calls`` JSON is
    #                      unwrapped with the codebase-canonical
    #                      ``COALESCE(NULLIF(...,'null'),'[]')`` guard (the same
    #                      form ``_unresolved_tool_calls`` uses) so a stored
    #                      ``"tool_calls": null`` yields zero rows instead of
    #                      "cannot extract elements from a scalar" — the bare
    #                      ``jsonb_array_elements(data->'tool_calls')`` form
    #                      ERRORS on that shape and would abort the migration.
    op.execute(
        """
        WITH
        user_s AS (
            SELECT session_id, MAX(seq) AS m FROM events
             WHERE kind = 'message' AND role = 'user'
             GROUP BY session_id
        ),
        stim_s AS (
            SELECT session_id, MAX(seq) AS m FROM events
             WHERE kind = 'message' AND role <> 'assistant'
             GROUP BY session_id
        ),
        err_s AS (
            SELECT session_id, MAX(seq) AS m FROM events
             WHERE kind = 'lifecycle' AND data->>'stop_reason' = 'error'
             GROUP BY session_id
        ),
        react_s AS (
            SELECT session_id,
                   MAX(COALESCE((data->>'reacting_to')::bigint, seq)) AS m
              FROM events
             WHERE kind = 'message' AND role = 'assistant'
             GROUP BY session_id
        ),
        asst AS (
            SELECT session_id,
                   jsonb_array_elements(
                       COALESCE(NULLIF(data->'tool_calls', 'null'::jsonb),
                                '[]'::jsonb)
                   )->>'id' AS tcid
              FROM events
             WHERE kind = 'message' AND role = 'assistant'
               AND data ? 'tool_calls'
        ),
        res AS (
            SELECT session_id, data->>'tool_call_id' AS tcid FROM events
             WHERE kind = 'message' AND role = 'tool'
        ),
        open_s AS (
            SELECT a.session_id,
                   COUNT(*) FILTER (WHERE r.tcid IS NULL) AS m
              FROM asst a
              LEFT JOIN res r USING (session_id, tcid)
             GROUP BY a.session_id
        )
        UPDATE sessions s
           SET last_user_seq        = COALESCE(u.m, 0),
               last_stimulus_seq     = COALESCE(st.m, 0),
               last_error_seq        = COALESCE(e.m, 0),
               last_reacted_seq      = COALESCE(rc.m, 0),
               open_tool_call_count  = COALESCE(o.m, 0)
          FROM sessions s2
          LEFT JOIN user_s  u  ON u.session_id  = s2.id
          LEFT JOIN stim_s  st ON st.session_id = s2.id
          LEFT JOIN err_s   e  ON e.session_id  = s2.id
          LEFT JOIN react_s rc ON rc.session_id = s2.id
          LEFT JOIN open_s  o  ON o.session_id  = s2.id
         WHERE s.id = s2.id
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_reacted_seq")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS open_tool_call_count")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_error_seq")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_user_seq")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS last_stimulus_seq")
