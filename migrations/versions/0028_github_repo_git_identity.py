"""Optional git identity (user.name / user.email) on github_repository.

Per #207, an attached repo can carry a deterministic git identity that
the sandbox stamps via ``git config`` after clone.  Both columns are
nullable text — absent means "no identity configured" (the v1 default,
preserved for callers that don't supply them).

Not encrypted: name and email aren't secrets.

Revision ID: 0028
Revises: 0027
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0028"
down_revision: str = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE session_github_repositories "
        "ADD COLUMN git_user_name text, "
        "ADD COLUMN git_user_email text"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE session_github_repositories "
        "DROP COLUMN IF EXISTS git_user_email, "
        "DROP COLUMN IF EXISTS git_user_name"
    )
