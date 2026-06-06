"""Partial index on terminal-error lifecycle events, for the derived
``errored`` session state.

Session ``status`` is being collapsed to a derived ``{active, idle}`` value
computed from the event log; ``errored`` becomes a special case of ``idle``
derived as "the latest ``turn_ended``/``error`` lifecycle event is more recent
than the latest user message" (see ``sweep.ERRORED_SESSIONS_SQL``). The sweep
excludes errored sessions on every pass, so the ``MAX(seq)`` over error
lifecycle events must be cheap.

Error lifecycle events are rare, so this is a tiny partial index. The
matching user-message ``MAX`` reuses ``events_session_message_seq_idx``
(``WHERE kind = 'message'``) from migration 0001.

Built with ``CREATE INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migration 0023.

Revision ID: 0062
Revises: 0061
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0062"
down_revision: str = "0061"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_turn_error_idx "
            "ON events (session_id, seq) "
            "WHERE kind = 'lifecycle' AND data->>'stop_reason' = 'error'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_turn_error_idx")
