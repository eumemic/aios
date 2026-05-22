"""Add nullable ``stop_hook jsonb`` to ``sessions`` (#374).

A pluggable Stop hook decides when a session may transition to ``idle``
at the harness's would-stop branch.  Three v1 hook types share this
column via a discriminated union:

* ``self_check`` — agent self-evaluates a condition each end-of-turn.
* ``task_call``  — only the ``task_complete`` tool may terminate.
* ``always_continue`` — never honor end-of-turn; only external interrupts pause.

``NULL`` (default) preserves the long-standing conversational default —
no behavioral change for sessions that don't opt in.

Revision ID: 0055
Revises: 0054
Create Date: 2026-05-21
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0055"
down_revision: str = "0054"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN stop_hook jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS stop_hook")
