"""ssh_servers surface arm + ssh_key vault credential kind.

Adds the fourth capability-surface dimension, ``ssh_servers``, to every
principal that carries a declared or frozen surface — agents, agent_versions,
workflows, workflow_versions, wf_runs — plus the per-session frozen overlay on
``sessions``. The five declared/pinned surfaces get ``NOT NULL DEFAULT '[]'``
(the 0052/0073/0112 http_servers precedent); ``sessions.ssh_servers`` is
NULLABLE with no default (the 0079 frozen-surface pattern — ``surface_frozen``
stays the single discriminator, so a defaulted ``'[]'`` would be ambiguous
against a legitimately-empty clamp). A frozen child row written before this
column existed reads NULL, which the loader treats as the empty grant
(fail-closed-correct: it was clamped before ssh existed, so it holds none).

Also widens ``vault_credentials_shape_check`` from the binary
``environment_variable``-vs-else split (0081) to a three-way split adding the
``ssh_key`` shape: ``target_url IS NULL AND secret_name IS NOT NULL AND
allowed_hosts IS NULL`` (worker-consumed key material — no target_url, no
egress scope; keyed by secret_name and sharing the existing partial-unique
``(vault_id, secret_name)`` index). The ``auth_type`` value CHECK is NOT
touched: 0111 dropped it (the AuthType Literal is the single source, enforced
by the typed insert writer), and a persisted-enum-drift test asserts its
absence — re-adding one here would regress that invariant.

Revision ID: 0177
Revises: 0176
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0177"
down_revision: str = "0176"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DECLARED_SURFACE_TABLES = (
    "agents",
    "agent_versions",
    "workflows",
    "workflow_versions",
    "wf_runs",
)

_SHAPE_CHECK_THREE_WAY = (
    "ALTER TABLE vault_credentials ADD CONSTRAINT vault_credentials_shape_check CHECK ("
    "(auth_type = 'environment_variable' "
    "AND target_url IS NULL AND secret_name IS NOT NULL "
    "AND allowed_hosts IS NOT NULL AND cardinality(allowed_hosts) > 0) "
    "OR (auth_type = 'ssh_key' "
    "AND target_url IS NULL AND secret_name IS NOT NULL AND allowed_hosts IS NULL) "
    "OR (auth_type NOT IN ('environment_variable', 'ssh_key') "
    "AND target_url IS NOT NULL AND secret_name IS NULL AND allowed_hosts IS NULL))"
)

# The binary form 0081 installed — restored on downgrade.
_SHAPE_CHECK_BINARY = (
    "ALTER TABLE vault_credentials ADD CONSTRAINT vault_credentials_shape_check CHECK ("
    "(auth_type = 'environment_variable' "
    "AND target_url IS NULL AND secret_name IS NOT NULL "
    "AND allowed_hosts IS NOT NULL AND cardinality(allowed_hosts) > 0) "
    "OR (auth_type <> 'environment_variable' "
    "AND target_url IS NOT NULL AND secret_name IS NULL AND allowed_hosts IS NULL))"
)


def upgrade() -> None:
    for table in _DECLARED_SURFACE_TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN ssh_servers jsonb NOT NULL DEFAULT '[]'::jsonb")
    # Nullable, no default (0079): surface_frozen is the sole discriminator.
    op.execute("ALTER TABLE sessions ADD COLUMN ssh_servers jsonb")

    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_shape_check")
    op.execute(_SHAPE_CHECK_THREE_WAY)


def downgrade() -> None:
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_shape_check")
    # Re-add the binary shape CHECK BEFORE dropping the columns, so it fails
    # loud (aborting the whole migration) if any ``ssh_key`` row exists —
    # INCLUDING archived husks (archive zeroes the secret blob but keeps
    # ``auth_type``, and archived rows are filtered out of API list output, so
    # they are invisible). Remedy:
    #   DELETE FROM vault_credentials WHERE auth_type = 'ssh_key';
    op.execute(_SHAPE_CHECK_BINARY)

    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS ssh_servers")
    for table in reversed(_DECLARED_SURFACE_TABLES):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS ssh_servers")
