"""Maintain ``sessions.channels`` as an array column, replacing the
per-step DISTINCT scan over ``events`` (#1742, sub-issue of the
zero-inference-gap epic #1733).

``list_session_channels`` (``src/aios/db/queries/events.py``) ran
``SELECT DISTINCT channel FROM events WHERE session_id=$1 AND account_id=$2
AND kind='message' AND channel IS NOT NULL ORDER BY channel`` — an
O(session-size) scan — and is awaited on EVERY step inside the harness's
pre-inference gather (``harness/loop.py``). The channel set a session has
seen is near-static: it only grows when a message event stamps a new
``channel`` address.

This migration adds ``channels text[] NOT NULL DEFAULT '{}'`` to
``sessions``. ``append_event`` (issue #1742's companion code change)
maintains the array in the SAME seq-allocating ``UPDATE`` that already
writes the session row under its row lock on every append — zero extra
round trips, race-free by construction. The read path becomes a
``sessions`` primary-key point read instead of a table scan.

Backfill: for every existing session, compute the DISTINCT sorted channel
set from its message events and store it as the array (``array_agg(DISTINCT
… ORDER BY …)`` — sorted precisely because insertion order doesn't exist
for a backfill and the read path returns ``sorted(channels)`` regardless).
Sessions with no channelled messages get the column DEFAULT ``'{}'`` (the
``UPDATE … FROM`` join simply doesn't match those rows).

Downgrade drops the column — the array is fully derivable from ``events``
if this migration needs to be rolled back.

Revision ID: 0133
Revises: 0131
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0133"
down_revision: str = "0131"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE sessions ADD COLUMN channels text[] NOT NULL DEFAULT '{}'"
    )
    op.execute(
        """
        UPDATE sessions s
           SET channels = c.chs
          FROM (
                SELECT session_id, array_agg(DISTINCT channel ORDER BY channel) AS chs
                  FROM events
                 WHERE kind = 'message' AND channel IS NOT NULL
                 GROUP BY session_id
               ) c
         WHERE s.id = c.session_id
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN channels")
