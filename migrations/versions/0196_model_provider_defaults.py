"""Add generalized model-provider defaults, optimistic versioning, and
credential-presence flags.

``api_key_set`` / ``credentials_set`` are computed from the plaintext at write
time (the blob is encrypted, and an encrypted ``{}`` is non-empty ciphertext,
so blob length cannot say whether a row holds a secret). Nullable: a NULL
means "written before the flags existed" and the read path falls back to the
pre-0196 ``ciphertext`` length rule, which is exact for those rows (every
pre-0196 blob is a bare API key, and archive zeroes the blob).

Revision ID: 0196
Revises: 0195
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0196"
down_revision: str = "0195"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_providers "
        "ADD COLUMN litellm_defaults jsonb NOT NULL DEFAULT '{}'::jsonb, "
        "ADD COLUMN version integer NOT NULL DEFAULT 1, "
        "ADD COLUMN api_key_set boolean, "
        "ADD COLUMN credentials_set boolean"
    )
    op.execute(
        "UPDATE model_providers SET api_key_set = length(ciphertext) > 0, "
        "credentials_set = length(ciphertext) > 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_providers DROP COLUMN credentials_set, DROP COLUMN api_key_set, "
        "DROP COLUMN version, DROP COLUMN litellm_defaults")
