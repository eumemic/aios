"""Record the agent version an operator re-rooted a sub-run's surface to (#2472 D2).

Revision ID: 0188
Revises: 0187

``invoke_workflow(..., as_agent={agent_id, version})`` clamps an operator run's
sub-run to that agent version's surface, so an eval arm runs with the authority the
agent would give it. The run's snapshotted surface already reflects the clamp; these
columns record why, for audit and for the parent's ``sub_runs()`` facts.

Both are NULL on every other run, including all rows written before this revision or
by an application image from before it. The column adds are catalog-only. No FK to
``agent_versions``: deleting an agent must not rewrite a finished run's record.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0188"
down_revision: str | None = "0187"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN as_agent_id text, ADD COLUMN as_agent_version integer"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN as_agent_id, DROP COLUMN as_agent_version")
