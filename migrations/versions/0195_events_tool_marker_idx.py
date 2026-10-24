"""Partial index on the awaiting read-model's lifecycle hold markers (#2393).

``list_unresolved_tool_calls_batch`` (``src/aios/db/queries/events.py``) builds
``Session.awaiting`` for every page of ``GET /v1/sessions`` (including
``view=lite``).  After ``_unresolved_tool_calls`` returns the still-open calls,
it looks up their ``tool_confirmed``/``allow`` and ``tool_requested`` lifecycle
markers.  The lookup used to be keyed on ``session_id = ANY(page)`` alone, and
the only index it could use was ``events_session_lifecycle_seq_idx``
(``(session_id, seq) WHERE kind = 'lifecycle'``, migration 0135).  It therefore
heap-fetched and filtered EVERY lifecycle row of every page session, so the cost
grew with total lifecycle volume.  On prod that hit the statement timeout and
returned 500s on 64b76fc.  ``events_tool_confirmed_allow_idx`` (0065) covers only
the confirmed/allow arm, and nothing indexed ``tool_requested``.

The query now probes exact ``(session_id, tool_call_id)`` pairs, and this index
serves the probe: one btree descent per unresolved call, however many lifecycle
events the sessions have.  Only marker rows are indexed: at most one
``tool_requested`` plus a few ``tool_confirmed`` per gated call, so the index
is small and adds little write cost.

Built with ``CREATE INDEX CONCURRENTLY`` inside ``autocommit_block`` (no
ACCESS EXCLUSIVE lock on the live-written ``events`` table), and any invalid
remnant from an interrupted build is dropped first.  This is the same pattern
as 0181.

Revision ID: 0195
Revises: 0194
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0195"
down_revision: str | None = "0194"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # An interrupted concurrent build can leave an invalid same-named index.
        # Remove any remnant so retries always perform a usable rebuild.
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_marker_idx")
        op.execute(
            "CREATE INDEX CONCURRENTLY events_tool_marker_idx "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'lifecycle' "
            "AND data->>'event' IN ('tool_confirmed', 'tool_requested')"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_marker_idx")
