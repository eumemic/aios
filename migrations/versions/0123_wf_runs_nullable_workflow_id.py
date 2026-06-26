"""Make ``wf_runs.workflow_id`` nullable for the inline-script run arm (T5, #1466).

A run has always pinned its own immutable ``script`` snapshot at launch — the
``workflows`` row was merely the *source* of that snapshot. T5 adds an
**inline-script arm** to ``create_run`` / ``call_workflow``: launch a one-shot
run directly from an inline ``{script, schemas, surface}`` body with **NO
``workflows`` row created**. Such a run has no definition to reference, so its
``workflow_id`` FK must be nullable.

This drops the ``NOT NULL`` on ``wf_runs.workflow_id``. The existing
``workflow_id text REFERENCES workflows(id) ON DELETE CASCADE`` FK is unchanged
and stays MATCH-SIMPLE, so a NULL ``workflow_id`` is exempt from the check (an
inline run references no workflow). The composite source-version FK
``(workflow_id, source_version, account_id)`` (0118) is likewise MATCH SIMPLE —
an inline run carries NULL for both ``workflow_id`` and ``source_version``, so it
too is exempt.

A *registered* run still sets ``workflow_id`` and remains fully enforced. No
data migration is needed: every existing row already has a non-NULL
``workflow_id``; this only relaxes the column so future inline rows may be NULL.

Revision ID: 0123
Revises: 0122
Create Date: 2026-06-25
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0123"
down_revision: str = "0122"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ALTER COLUMN workflow_id DROP NOT NULL")


def downgrade() -> None:
    # Re-imposing NOT NULL would fail if any inline (NULL workflow_id) run exists.
    # The downgrade is best-effort and intended only for a DB with no inline runs.
    op.execute("ALTER TABLE wf_runs ALTER COLUMN workflow_id SET NOT NULL")
