"""Composite tenant FKs for secret-bearing chains.

Revision ID: 0093
Revises: 0092
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0093"
down_revision: str = "0092"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PARENT_UNIQUES: Sequence[tuple[str, str]] = (
    ("vaults", "vaults_id_account_id_key"),
    ("sessions", "sessions_id_account_id_key"),
    ("wf_runs", "wf_runs_id_account_id_key"),
)

_BARE_FKS: Sequence[tuple[str, str]] = (
    ("session_vaults", "session_vaults_session_id_fkey"),
    ("session_vaults", "session_vaults_vault_id_fkey"),
    ("wf_run_vaults", "wf_run_vaults_run_id_fkey"),
    ("wf_run_vaults", "wf_run_vaults_vault_id_fkey"),
    ("vault_credentials", "vault_credentials_vault_id_fkey"),
    ("oauth_flows", "oauth_flows_vault_id_fkey"),
)

_COMPOSITE_FKS: Sequence[tuple[str, str, str]] = (
    (
        "session_vaults",
        "session_vaults_session_account_id_fkey",
        "FOREIGN KEY (session_id, account_id) REFERENCES sessions(id, account_id) ON DELETE CASCADE",
    ),
    (
        "session_vaults",
        "session_vaults_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id)",
    ),
    (
        "wf_run_vaults",
        "wf_run_vaults_run_account_id_fkey",
        "FOREIGN KEY (run_id, account_id) REFERENCES wf_runs(id, account_id) ON DELETE CASCADE",
    ),
    (
        "wf_run_vaults",
        "wf_run_vaults_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id)",
    ),
    (
        "vault_credentials",
        "vault_credentials_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id) ON DELETE CASCADE",
    ),
    (
        "oauth_flows",
        "oauth_flows_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id) ON DELETE CASCADE",
    ),
)


def upgrade() -> None:
    for table, constraint in _PARENT_UNIQUES:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} UNIQUE (id, account_id)")

    for table, constraint in _BARE_FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")

    for table, constraint, definition in _COMPOSITE_FKS:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} {definition} NOT VALID")

    for table, constraint, _definition in _COMPOSITE_FKS:
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {constraint}")


def downgrade() -> None:
    for table, constraint, _definition in reversed(_COMPOSITE_FKS):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")

    op.execute(
        "ALTER TABLE session_vaults ADD CONSTRAINT session_vaults_session_id_fkey "
        "FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE session_vaults ADD CONSTRAINT session_vaults_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id)"
    )
    op.execute(
        "ALTER TABLE wf_run_vaults ADD CONSTRAINT wf_run_vaults_run_id_fkey "
        "FOREIGN KEY (run_id) REFERENCES wf_runs(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE wf_run_vaults ADD CONSTRAINT wf_run_vaults_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id)"
    )
    op.execute(
        "ALTER TABLE vault_credentials ADD CONSTRAINT vault_credentials_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE oauth_flows ADD CONSTRAINT oauth_flows_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE"
    )

    for table, constraint in reversed(_PARENT_UNIQUES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
