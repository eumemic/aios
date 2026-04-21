"""Promote message-structural fields to physical columns on ``events`` and
widen ``events_search`` so the agent's SQL surface matches its mental model.

Issue #117. The ``search_events`` tool gave an agent SQL access to the session
log via ``events_search``, but the view exposed only ``id, seq, role,
created_at, content_text`` — narrower than the live context (which the
context builder augments with channel headers, sender identity, notification
markers at render time). Every filter dimension the agent couldn't express
turned into an ILIKE-on-serialized-content hack.

Four promotions, all backfillable from existing JSONB and stamped by
``append_event`` going forward:

* ``role``        — ``data->>'role'`` for message events.
* ``tool_name``   — ``data->>'name'`` for tool-result events; the first
                    ``data->'tool_calls'->0->'function'->>'name'`` for
                    assistant events with tool_calls. Multi-tool turns
                    remain discoverable by that first call's name.
* ``is_error``    — ``(data->>'is_error')::boolean`` when set. The field is
                    only written when truthy (successful results omit it),
                    so the column is TRUE on failure, NULL otherwise — no
                    FALSE values in practice.
* ``sender_name`` — ``data->'metadata'->>'sender_name'`` for user events
                    carrying connector metadata.

Follows migration 0018's column-promotion pattern (derive-at-append +
single backfill query).  The ``events_search`` view is recreated to expose
all four columns plus ``channel`` (on the table since 0018 but not
previously surfaced), keeping the ``kind = 'message'`` filter from
migration 0013.  Span exposure and raw-data
JSONB access are deliberately deferred — they leak cost/token signal, which
is problematic for agents running on behalf of an agency-on-behalf-of-a-
customer, and want a real per-agent tool-access-control primitive first.

Four indexes added:

* ``(session_id, channel, seq)`` partial — recap renderer and channel filters.
* ``(session_id, created_at)`` — time-range queries for agent self-reflection.
* ``(session_id, tool_name, seq)`` partial — "my bash calls" style queries.
* ``(session_id, seq) WHERE is_error`` partial — "tool failures in the last
  hour" — high selectivity, cheap.

Revision ID: 0022
Revises: 0021
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0022"
down_revision: str = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add four nullable columns.
    op.execute("ALTER TABLE events ADD COLUMN role text")
    op.execute("ALTER TABLE events ADD COLUMN tool_name text")
    op.execute("ALTER TABLE events ADD COLUMN is_error boolean")
    op.execute("ALTER TABLE events ADD COLUMN sender_name text")

    # 2. Backfill from existing JSONB. The JSON paths mirror the ones
    #    append_event will use going forward so old and new rows stay
    #    byte-equivalent in these columns.
    op.execute(
        """
        UPDATE events SET
          role = CASE WHEN kind = 'message'
                      THEN data->>'role' END,
          tool_name = CASE
            WHEN kind = 'message' AND data->>'role' = 'tool'
              THEN data->>'name'
            WHEN kind = 'message' AND data->>'role' = 'assistant'
                 AND data ? 'tool_calls'
              THEN data->'tool_calls'->0->'function'->>'name'
            END,
          is_error = CASE WHEN kind = 'message'
                               AND (data->>'is_error') IS NOT NULL
                          THEN (data->>'is_error')::boolean END,
          sender_name = CASE
            WHEN kind = 'message' AND data->>'role' = 'user'
              THEN data->'metadata'->>'sender_name' END
        """
    )

    # 3. Indexes.
    op.execute(
        "CREATE INDEX events_session_channel_seq_idx "
        "ON events (session_id, channel, seq) "
        "WHERE channel IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX events_session_created_at_idx "
        "ON events (session_id, created_at)"
    )
    op.execute(
        "CREATE INDEX events_session_tool_name_seq_idx "
        "ON events (session_id, tool_name, seq) "
        "WHERE tool_name IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX events_session_is_error_idx "
        "ON events (session_id, seq) "
        "WHERE is_error IS TRUE"
    )

    # 4. Recreate events_search with the widened column set. Keep the
    #    kind='message' filter (span exposure deferred). channel has been
    #    on the table since 0018 but wasn't surfaced in the view.
    op.execute("DROP VIEW IF EXISTS events_search")
    op.execute(
        """
        CREATE VIEW events_search AS
        SELECT
            id,
            seq,
            role,
            channel,
            tool_name,
            is_error,
            sender_name,
            created_at,
            COALESCE(data->>'content', data::text) AS content_text
        FROM events
        WHERE session_id = current_setting('app.session_id', true)
          AND kind = 'message'
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS events_search")
    op.execute("DROP INDEX IF EXISTS events_session_is_error_idx")
    op.execute("DROP INDEX IF EXISTS events_session_tool_name_seq_idx")
    op.execute("DROP INDEX IF EXISTS events_session_created_at_idx")
    op.execute("DROP INDEX IF EXISTS events_session_channel_seq_idx")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS sender_name")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS is_error")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS tool_name")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS role")
    # Restore the pre-0022 view (post-0013 shape: messages-only, narrow).
    op.execute(
        """
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
        """
    )
