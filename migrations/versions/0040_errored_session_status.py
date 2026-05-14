"""Add ``errored`` to the session status check constraint.

When the wake-step retry budget is spent (4 consecutive ``rescheduling``
turn_ends) the harness now parks the session in ``errored`` instead of
``idle`` so the periodic sweep stops re-firing wakes that just time out
again.  ``flip_quiescent_to_pending`` is widened in the same change to
also clear ``errored`` on the next user message, preserving the operator
recovery contract from #353.

Downgrade rewrites ``errored`` to ``terminated`` rather than ``idle`` —
the original constraint can't represent the parked-after-budget state,
and ``terminated`` is the existing terminal vocabulary.  ``idle`` would
silently reintroduce the #353 loop on any session currently parked.

Revision ID: 0040
Revises: 0039
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0040"
down_revision: str = "0039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('pending', 'running', 'idle', 'rescheduling', "
        "'errored', 'terminated'));"
    )


def downgrade() -> None:
    op.execute("UPDATE sessions SET status = 'terminated' WHERE status = 'errored';")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_status_check;")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_status_check "
        "CHECK (status IN ('pending', 'running', 'idle', 'rescheduling', 'terminated'));"
    )
