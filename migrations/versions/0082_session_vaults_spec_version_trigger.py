"""Bump ``sessions.spec_version`` on ``session_vaults`` changes (#873).

Migration 0077 deliberately excluded ``session_vaults`` from the
spec-version bump triggers because vault bindings reached the agent only
via the MCP pool — they didn't feed ``build_spec_from_session``. Since
#873 that premise is false: ``environment_variable`` credentials resolve
through a session's bound vaults into the provisioning plan, so a vault
attach/detach changes what the spec builder produces. Per 0077's own
rule ("only the tables that actually feed build_spec_from_session get a
bump trigger" — and every table that does, must), ``session_vaults`` now
joins ``session_memory_stores`` / ``session_github_repositories``.

Without this, the Layer-1 write-path eviction is the only cover and it
is a no-op in the API process — where ``PUT /sessions/:id`` (the only
``session_vaults`` write path) actually runs. A live sandbox would keep
a stale credential set until an unrelated resource change recycled it.

Credential-LEVEL drift (rotation, create/archive within a still-bound
vault) is not a ``session_vaults`` write and stays uncovered here —
that's the recycle-on-rotation drift key (#877).

Reuses 0077's ``bump_session_spec_version()`` trigger function.

Revision ID: 0082
Revises: 0081
Create Date: 2026-06-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0082"
down_revision: str = "0081"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TRIGGER session_vaults_bump_spec_version
        AFTER INSERT OR UPDATE OR DELETE ON session_vaults
        FOR EACH ROW
        EXECUTE FUNCTION bump_session_spec_version()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS session_vaults_bump_spec_version ON session_vaults")
