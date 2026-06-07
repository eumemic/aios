"""Partial index on operator-confirmed (allow) tool-confirmation lifecycle
events, keyed by ``(session_id, tool_call_id)``.

Two readers resolve the same "confirmed ``allow`` whose ``tool_call`` has no
``tool_result``" predicate and both benefit:

* ``sweep.CONFIRMED_ROWS_SQL`` (``find_sessions_needing_inference`` case (c))
  scans confirmed-allow lifecycle rows **cross-session** every sweep pass to
  decide which sessions need a dispatch step — previously unindexed for this
  predicate (only the wide ``events_session_seq_idx`` applied).
* ``queries.list_confirmed_unresolved_tool_calls`` resolves the dispatchable
  ``tool_call`` dicts for one session on **every** inference step; without this
  index it would scan all of a long session's lifecycle rows per step.

Confirmed-allow events are rare relative to total events (one per operator
``allow`` of an ``always_ask`` tool), so this is a small partial index. The
companion result-existence check reuses ``events_tool_result_idx`` (migration
0023) and the parent-assistant lookup reuses ``events_assistant_tool_calls_idx``.

Built with ``CREATE INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migrations 0023 / 0062.

Revision ID: 0065
Revises: 0064
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0065"
down_revision: str = "0064"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_tool_confirmed_allow_idx "
            "ON events (session_id, (data->>'tool_call_id')) "
            "WHERE kind = 'lifecycle' "
            "AND data->>'event' = 'tool_confirmed' "
            "AND data->>'result' = 'allow'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_confirmed_allow_idx")
