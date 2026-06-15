"""Add the trusted DOWN-counting ``wf_runs.depth`` scalar (#1124).

Issue #1124 retires the ``run_ancestor_depth`` ``WITH RECURSIVE`` up-walk over
``wf_runs.parent_run_id`` and replaces it with a single trusted DOWN-counting
depth scalar that rides the launching stimulus. A run carries its own remaining
recursion budget on the row: an **edgeless root** (operator/HTTP ``POST /runs``,
``parent_run_id IS NULL``) is seeded at the shared budget constant
(``WORKFLOW_RUN_MAX_DEPTH`` = 10); a child run is stamped ``parent.depth - 1``,
refused before write at the floor. The run's depth is the depth its ``agent()``
children's ``request_opened`` edge inherits, so the whole trusted-invocation
chain decrements by construction — the depth budget IS the cycle bound.

The wake-side ``wake_depth`` (#1083) is a separate carrier and is untouched.

Purely additive: one nullable column with a server default = the budget, so the
new-code/old-schema deploy window is safe (existing in-flight rows read as a
root-budget run; new writes stamp the computed depth). Backfill existing rows to
the budget — they predate the column and were created under the old up-walk cap,
which already bounded them at ``WORKFLOW_RUN_MAX_DEPTH``, so seeding them at the
budget is the conservative (most-permissive) grandfather value.

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

# Mirror of aios.workflows.service.WORKFLOW_RUN_MAX_DEPTH. Inlined (not imported)
# so the migration stays self-contained and replayable against any code version.
_WORKFLOW_RUN_MAX_DEPTH = 10


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE wf_runs ADD COLUMN depth integer NOT NULL "
        f"DEFAULT {_WORKFLOW_RUN_MAX_DEPTH}"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS depth")
