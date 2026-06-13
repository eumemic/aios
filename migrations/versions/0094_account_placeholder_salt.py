"""Account placeholder salt for vault-key-stable env placeholders.

Revision ID: 0094
Revises: 0093
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0094"
down_revision: str = "0093"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE accounts ADD COLUMN placeholder_salt_ciphertext bytea")
    op.execute("ALTER TABLE accounts ADD COLUMN placeholder_salt_nonce bytea")
    op.execute(
        "ALTER TABLE accounts ADD CONSTRAINT accounts_placeholder_salt_pair_ck "
        "CHECK ((placeholder_salt_ciphertext IS NULL) = (placeholder_salt_nonce IS NULL))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE accounts DROP CONSTRAINT accounts_placeholder_salt_pair_ck")
    op.execute("ALTER TABLE accounts DROP COLUMN placeholder_salt_nonce")
    op.execute("ALTER TABLE accounts DROP COLUMN placeholder_salt_ciphertext")
