"""Generalize ``vault_credentials`` from MCP-specific to substrate-generic.

Part of #465 (`http_servers` agent-config primitive). The credential
subsystem already encrypts, refreshes, and resolves generic auth
material — only the column name and the discriminator literals were
MCP-named. This migration makes ``vault_credentials`` honestly what it
already was: a generic encrypted credential store keyed by target URL.

Specifically:

* renames ``mcp_server_url`` → ``target_url``;
* widens the ``auth_type`` CHECK from ``('mcp_oauth', 'static_bearer')``
  to ``('bearer_header', 'oauth2_refresh', 'basic')``;
* data-migrates existing rows (``static_bearer`` → ``bearer_header``,
  ``mcp_oauth`` → ``oauth2_refresh``).

Postgres references indexed columns by ``attnum`` rather than name, so
``vault_credentials_url_uniq`` carries over the column rename without a
rebuild — no index DDL needed.

The CHECK widen also adds ``basic`` (HTTP Basic auth, used by
``http_servers``).

Downgrade is symmetric, but fails loud if any ``basic`` row exists —
there is no pre-rename literal to map it back to.

Revision ID: 0051
Revises: 0050
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0051"
down_revision: str = "0050"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the old CHECK so the data-migration UPDATEs aren't rejected.
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    op.execute(
        "UPDATE vault_credentials SET auth_type = 'bearer_header' "
        "WHERE auth_type = 'static_bearer'"
    )
    op.execute(
        "UPDATE vault_credentials SET auth_type = 'oauth2_refresh' "
        "WHERE auth_type = 'mcp_oauth'"
    )
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN ('bearer_header', 'oauth2_refresh', 'basic'))"
    )

    op.execute("ALTER TABLE vault_credentials RENAME COLUMN mcp_server_url TO target_url")


def downgrade() -> None:
    op.execute("ALTER TABLE vault_credentials RENAME COLUMN target_url TO mcp_server_url")

    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    op.execute(
        "UPDATE vault_credentials SET auth_type = 'static_bearer' "
        "WHERE auth_type = 'bearer_header'"
    )
    op.execute(
        "UPDATE vault_credentials SET auth_type = 'mcp_oauth' "
        "WHERE auth_type = 'oauth2_refresh'"
    )
    # Narrowing CHECK fails loud if any ``basic`` row remains.
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN ('mcp_oauth', 'static_bearer'))"
    )
