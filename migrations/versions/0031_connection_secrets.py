"""Encrypted-at-rest secrets on the connection record.

Platform credentials (Telegram ``bot_token``, Signal ``phone``, etc.)
move off connector container env into the connection row.  The shape
mirrors ``vault_credentials`` (libsodium SecretBox, paired
ciphertext + nonce columns) so the existing ``CryptoBox`` machinery is
the single source of encryption-at-rest patterns in the schema.

Plural ``secrets_*`` because the encrypted blob is a JSON-serialised
``dict[str, str]`` — typically a single ``bot_token`` key today, but
the shape accommodates any number of platform-specific creds.

The check constraint enforces pair-or-neither: either both columns are
NULL (no secrets configured) or both are NOT NULL (the encrypted blob
is present).  The pair is meaningless without both halves so we reject
half-populated states at the schema level.

Revision ID: 0031
Revises: 0030
Create Date: 2026-05-08
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0031"
down_revision: str = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE connections
          ADD COLUMN secrets_ciphertext bytea,
          ADD COLUMN secrets_nonce bytea,
          ADD CONSTRAINT connections_secrets_pair_ck
            CHECK ((secrets_ciphertext IS NULL) = (secrets_nonce IS NULL))
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE connections
          DROP CONSTRAINT IF EXISTS connections_secrets_pair_ck,
          DROP COLUMN IF EXISTS secrets_ciphertext,
          DROP COLUMN IF EXISTS secrets_nonce
        """
    )
