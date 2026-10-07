"""Constrain ``wf_runs.visibility`` to its known values (#2513).

Revision ID: 0194
Revises: 0191

The run readers (``RunReader.can_see``, the ``list_wf_runs`` reader filter, the
``run_completion`` fire matcher) treat any value other than ``account`` as
launcher-private, so an unknown value would silently read as "private". The
insert trigger (0185, 0191) only ever stamps ``account`` or ``session``; this
CHECK makes anything else unrepresentable.

Mechanics. The constraint is added ``NOT VALID`` under a bounded lock wait (a
catalog-only change, enforced on every later write), then validated in an
autocommit block, which takes only SHARE UPDATE EXCLUSIVE and so scans the
populated table without blocking run inserts. A row with an unknown value fails
the VALIDATE and so the deploy; the ``DROP ... IF EXISTS`` makes the migration
re-runnable after such a failure leaves the unvalidated constraint behind.

The old application image never writes the column (the trigger is its only
writer), so it is unaffected by the constraint.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0194"
down_revision: str | None = "0191"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "wf_runs_visibility_known"


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(f"ALTER TABLE wf_runs DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
    op.execute(
        f"ALTER TABLE wf_runs ADD CONSTRAINT {CONSTRAINT} "
        "CHECK (visibility IN ('account', 'session')) NOT VALID"
    )
    with op.get_context().autocommit_block():
        op.execute(f"ALTER TABLE wf_runs VALIDATE CONSTRAINT {CONSTRAINT}")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(f"ALTER TABLE wf_runs DROP CONSTRAINT IF EXISTS {CONSTRAINT}")
