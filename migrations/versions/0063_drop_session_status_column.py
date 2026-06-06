"""Drop the denormalized ``sessions.status`` column (and its index).

Session ``status`` is now a derived ``{active, idle}`` value computed from the
event log at read time (``queries._SESSION_STATUS_EXPR``); ``errored`` is a
special case of ``idle`` derived from the latest ``turn_ended``/``error``
lifecycle event vs. the latest user message (``sweep.ERRORED_SESSIONS_SQL``).
Nothing reads the column for control flow anymore:

* the sweep excludes errored sessions via the derived predicate (subtracted
  in-process), not ``status <> 'errored'``;
* recovery from errored is automatic (a user message overtakes the error
  event), so the ``append_event`` ``idle/errored → pending`` flip is gone;
* ``set_session_stop_reason`` writes only ``stop_reason``.

``stop_reason`` stays — it records why the last step ended and drives the
console's Errored pill (``idle`` + ``stop_reason.type == 'error'``).

Revision ID: 0063
Revises: 0062
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0063"
down_revision: str = "0062"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS sessions_status_idx")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS status")


def downgrade() -> None:
    # Best-effort restore: the historical per-status value is unrecoverable
    # (it was derived), so every session comes back as 'idle'. The partial
    # index is recreated to match migration 0001.
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'idle'")
    op.execute(
        "CREATE INDEX IF NOT EXISTS sessions_status_idx "
        "ON sessions (status) WHERE archived_at IS NULL"
    )
