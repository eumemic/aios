"""Sessions ``spec_version`` + bump triggers for sandbox staleness probe (#713).

The worker's :class:`aios.sandbox.registry.SandboxRegistry` caches a live
:class:`SandboxHandle` per session across steps. When a session-scoped
resource changes between steps, the cached sandbox no longer reflects
:func:`aios.sandbox.spec.build_spec_from_session` and must be recycled.
The write-path eviction (Layer 1) is a no-op in the API process (the
registry global is worker-only) and invisible to direct-SQL mutations,
so we add a version probe (Layer 2): a monotonically-incrementing
``sessions.spec_version`` the registry stamps onto each cached handle and
re-reads on a warm hit, recycling on mismatch.

Only the tables that actually feed ``build_spec_from_session`` get a bump
trigger:

- ``session_memory_stores`` — drives the ``/mnt/memory/*`` bind mounts.
- ``session_github_repositories`` — drives the cloned working-tree mounts.

``session_scheduled_tasks`` (read per-turn by the scheduled-task runner,
never by the spec builder) and ``session_vaults`` / connection bindings
(reach the agent via the MCP pool / per-step tool provider, not the
sandbox spec) deliberately get NO bump trigger. This Layer1/Layer2
asymmetry is intentional — Layer 1 still evicts on vault/connection
binding changes as defense-in-depth, but Layer 2 only fires on changes
that genuinely alter the sandbox spec. We do NOT touch migration 0059's
``notify_scheduled_tasks_due`` / ``session_scheduled_tasks_notify``
trigger; the two coexist on different tables.

``set_session_resources`` (DELETE-all + re-INSERT N) bumps
``spec_version`` by N+M within a single transaction — that's fine: the
registry only ever compares the version for inequality, never for an
exact delta.

Revision ID: 0077
Revises: 0076
Create Date: 2026-06-04
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0077"
down_revision: str = "0076"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN spec_version integer NOT NULL DEFAULT 0")
    op.execute(r"""
        CREATE OR REPLACE FUNCTION bump_session_spec_version() RETURNS TRIGGER AS $$
        BEGIN
            UPDATE sessions
               SET spec_version = spec_version + 1
             WHERE id = COALESCE(NEW.session_id, OLD.session_id);
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER session_memory_stores_bump_spec_version
        AFTER INSERT OR UPDATE OR DELETE ON session_memory_stores
        FOR EACH ROW
        EXECUTE FUNCTION bump_session_spec_version()
    """)
    op.execute("""
        CREATE TRIGGER session_github_repositories_bump_spec_version
        AFTER INSERT OR UPDATE OR DELETE ON session_github_repositories
        FOR EACH ROW
        EXECUTE FUNCTION bump_session_spec_version()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS session_github_repositories_bump_spec_version "
        "ON session_github_repositories"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS session_memory_stores_bump_spec_version "
        "ON session_memory_stores"
    )
    op.execute("DROP FUNCTION IF EXISTS bump_session_spec_version()")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS spec_version")
