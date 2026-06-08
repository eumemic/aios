"""Workflows Block 2: bind a workflow run to an environment.

`agent : session :: workflow : run` — sessions bind to environments, so runs
bind to environments. A run's ``agent()`` children spawn into ``run.environment_id``
(inheriting it, the way a session inherits the environment chosen at creation).
The environment is chosen at run-creation time (data + API level), not baked into
the immutable workflow definition.

``wf_runs`` has no run-creation surface yet (Block 1 shipped the runtime core,
not the HTTP/CLI), so the table is empty and ``NOT NULL`` with no default is safe.

Revision ID: 0067
Revises: 0066
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0067"
down_revision: str = "0066"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN environment_id text NOT NULL REFERENCES environments(id)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS environment_id")
