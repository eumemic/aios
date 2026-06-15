"""Add the DOWN-counting trusted invocation ``depth`` scalar to ``wf_runs`` (#1124).

Issue #1124 retires the run-only ancestor walk (the ``run_ancestor_depth``
``WITH RECURSIVE`` CTE over ``wf_runs.parent_run_id``) and replaces it with a
single DOWN-counting depth scalar carried on the trusted invocation edge. A run
is a servicer with exactly one launch, so until #1126 makes the run-inbound
edge first-class, the run carries its own remaining depth on the row.

``depth`` is the budget remaining for this run's OUTGOING trusted edges (run→run
sub-launches, run→session ``agent()`` children). An edgeless root (the
operator/HTTP ``POST /runs`` path, a trigger fire with no completing-run parent)
seeds at the full budget (10). A nested launch stamps ``parent.depth - 1`` and
refuses BEFORE writing the child when the parent has no budget left. Cycles are
bounded BY CONSTRUCTION — the decrement IS the cycle bound.

Purely additive — a single ``ADD COLUMN`` with a default, so existing rows
backfill to the full budget. Safe in the post-deploy new-code/old-schema window:
the new code reads/writes ``depth`` only on runs it creates under the new schema,
and the column is invisible to the running container until the migration
completes.

Revision ID: 0100
Revises: 0099
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0100"
down_revision: str = "0099"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN depth integer NOT NULL DEFAULT 10 CHECK (depth >= 0)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN depth")
