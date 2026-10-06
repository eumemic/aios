"""Let a run hold a reference to a request a session sent (#2474 B1).

Revision ID: 0189
Revises: 0188

A workflow-as-model run, or an eval sub-run handed one, works on a request a
session composed. Instead of each run row keeping the whole request, the row can
hold a reference to the span that captured it (#2471): ``(session_id, span event
id)``. The reference is also the run's grant: a run can resolve only the request
it was created with.

Both columns are NULL on every run that holds no reference, including every row
written before this revision or by an application image from before it. The
column adds are catalog-only. No FK: the span lives in the session's event log,
and deleting the session leaves a reference that resolves to nothing, which a
reader reports as an unavailable request.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0189"
down_revision: str | None = "0188"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN request_ref_session_id text, "
        "ADD COLUMN request_ref_id text"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs DROP COLUMN request_ref_session_id, DROP COLUMN request_ref_id"
    )
