"""Widen ``vault_credentials.auth_type`` CHECK to include ``custom_header``.

The ``custom_header`` scheme covers API keys passed in a non-Authorization
header (Anthropic ``x-api-key``, browser-use ``X-Browser-Use-API-Key``,
Posthog ``X-API-Key``, etc.) — a common pattern the v1 catalogue
(``bearer_header`` / ``oauth2_refresh`` / ``basic``) didn't cover because
RFC-defined HTTP auth schemes all use the ``Authorization`` header.

Part of #465.

Downgrade narrows the CHECK back; fails loud if any ``custom_header``
row exists.

Revision ID: 0053
Revises: 0052
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0053"
down_revision: str = "0052"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN ('bearer_header', 'oauth2_refresh', 'basic', 'custom_header'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    # Narrowing CHECK fails loud if any ``custom_header`` row remains.
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN ('bearer_header', 'oauth2_refresh', 'basic'))"
    )
