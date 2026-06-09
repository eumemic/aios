"""Archive-on-quiescence: a launch-time flag to reclaim a session when it idles.

``archive_when_idle`` is an immutable launch-time property of a session: when set,
the harness soft-archives the session the first time it goes idle owing nothing
(see ``reclaim_session_if_idle`` + ``loop._run_step``). Workflow ``agent()``
children launch with it ``TRUE`` (they answer one request, then they're done);
foreground/API and per-chat launches default to ``FALSE``. The same flag rides
``session_templates`` so the per-chat resolver can copy it down to spawns.

Both columns are ``NOT NULL DEFAULT false`` — existing rows keep today's
"persist forever" behavior.

Revision ID: 0070
Revises: 0069
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0070"
down_revision: str = "0069"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN archive_when_idle boolean NOT NULL DEFAULT false")
    op.execute(
        "ALTER TABLE session_templates "
        "ADD COLUMN archive_when_idle boolean NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS archive_when_idle")
    op.execute("ALTER TABLE session_templates DROP COLUMN IF EXISTS archive_when_idle")
