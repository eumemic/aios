"""Drop the redundant ``events_session_seq_idx`` on ``events``.

``events_session_seq_idx`` (created in migration 0001) duplicates the implicit
btree Postgres builds to back the ``UNIQUE (session_id, seq)`` constraint on
``events``. Two identical ``(session_id, seq)`` btrees write-amplify every
insert on the hottest table in the system; dropping the redundant one removes
that overhead with no read-path regression (the unique-constraint btree serves
the same lookups). Migration 0064 already declined the same redundancy on
``wf_run_events`` — the rule was learned there but the original ``events`` copy
was never repaired.

Built with ``DROP INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migrations 0023 / 0062.

Revision ID: 0080
Revises: 0079
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0080"
down_revision: str = "0079"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_session_seq_idx")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_session_seq_idx "
            "ON events (session_id, seq)"
        )
