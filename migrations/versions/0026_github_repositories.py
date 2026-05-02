"""GitHub repository session resources.

Adds a single table ``session_github_repositories``: each row is a per-session
attachment that mounts a git repo into the sandbox at ``mount_path``. There is
no top-level ``github_repositories`` table — the row IS the resource (parity
with Anthropic Managed Agents, where ``github_repository`` resources only exist
within a session).

The clone token is stored encrypted via the same libsodium CryptoBox used by
``vault_credentials`` (``ciphertext``/``nonce`` columns). It is never echoed in
API responses; the only mutation supported is rotation via
``POST /v1/sessions/{sid}/resources/{rid}``.

Unique on ``(session_id, mount_path)`` so two repos can't fight for the same
mount inside one container.

Revision ID: 0026
Revises: 0025
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0026"
down_revision: str = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE session_github_repositories (
            id              text PRIMARY KEY,
            session_id      text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            rank            integer NOT NULL,
            repo_url        text NOT NULL,
            mount_path      text NOT NULL,
            ciphertext      bytea NOT NULL,
            nonce           bytea NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            CHECK (mount_path ~ '^(/[^/\x00]+)+$'),
            CHECK (mount_path !~ '(^|/)\.{1,2}(/|$)')
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX session_github_repos_mount_uniq
            ON session_github_repositories (session_id, mount_path)
    """)
    op.execute("""
        CREATE INDEX session_github_repos_by_session
            ON session_github_repositories (session_id, rank)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_github_repositories")
