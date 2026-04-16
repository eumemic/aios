"""ON DELETE CASCADE on vault_credentials.vault_id.

Previously, ``delete_vault`` had to run two SQL statements: a manual
``DELETE FROM vault_credentials WHERE vault_id = $1`` followed by the
``DELETE FROM vaults``. Adding ``ON DELETE CASCADE`` to the FK lets
Postgres handle the child row cleanup, so ``delete_vault`` collapses
to a single statement.

Note: this only affects ``DELETE`` paths. ``archive_vault`` is an
``UPDATE`` and does not trigger the cascade — the service-layer code
explicitly zeros child credentials' encrypted blobs in the same
transaction.

Revision ID: 0015
Revises: 0014
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015"
down_revision: str = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE vault_credentials DROP CONSTRAINT IF EXISTS vault_credentials_vault_id_fkey"
    )
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id) ON DELETE CASCADE"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE vault_credentials DROP CONSTRAINT IF EXISTS vault_credentials_vault_id_fkey"
    )
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_vault_id_fkey "
        "FOREIGN KEY (vault_id) REFERENCES vaults(id)"
    )
