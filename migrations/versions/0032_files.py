"""Session-scoped file uploads (#324).

Revision ID: 0032
Revises: 0031
Create Date: 2026-05-11
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0032"
down_revision: str = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE files (
            id              text PRIMARY KEY,
            session_id      text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            filename        text NOT NULL,
            host_path       text NOT NULL,
            in_sandbox_path text NOT NULL,
            size            bigint NOT NULL,
            content_type    text NOT NULL,
            sha256          text NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX files_session_id_created_at_idx
            ON files (session_id, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS files_session_id_created_at_idx")
    op.execute("DROP TABLE IF EXISTS files")
