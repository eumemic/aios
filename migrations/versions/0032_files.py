"""Session-scoped file uploads (#324).

Records bytes uploaded via ``POST /v1/sessions/<id>/files``.  The file row
holds enough metadata to (a) locate the bytes on the api's filesystem
(``host_path``), (b) tell the model where to find them inside the sandbox
(``in_sandbox_path``), and (c) verify integrity (``sha256``).

``ON DELETE CASCADE`` ties row lifetime to the session — when a session is
hard-deleted, its uploads disappear too.  Cleanup of the host directory is
handled out-of-band (same policy as ``_attachments``: workspace cleanup is
a separate sweep).

The ``(session_id, created_at DESC)`` index supports a future
``GET /v1/sessions/<id>/files`` listing endpoint without a full scan; it's
cheap to maintain on the write path since uploads are not high-frequency.

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
