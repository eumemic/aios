"""Add ``preempt_policy`` column to agents + agent_versions (#253).

Per-agent policy for what happens to an in-flight model call when a new
wake-eligible event (e.g. a user message) arrives mid-step: ``'wait'``
(default — the step finishes, the queued wake handles the event next
step) or ``'preempt'`` (the model phase is cancelled and the step
restarts against context that includes the event).  Default ``'wait'``
so every existing agent keeps current behavior; chat-facing agents opt
in per agent.  No CHECK constraint — the ``PreemptPolicy`` Literal on
the pydantic models is the single validation point (0111 precedent).
Purely additive, safe in the new-code/old-schema deploy window.

Revision ID: 0139
Revises: 0138
Create Date: 2026-07-09
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0139"
down_revision: str = "0138"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN preempt_policy text NOT NULL DEFAULT 'wait';")
    op.execute("ALTER TABLE agent_versions ADD COLUMN preempt_policy text NOT NULL DEFAULT 'wait';")


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS preempt_policy;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS preempt_policy;")
