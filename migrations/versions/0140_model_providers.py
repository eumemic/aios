"""Per-account model-provider config: encrypted API key + proxy base URL.

Model-provider API keys have been worker-global process env vars (LiteLLM
resolves them by model-string prefix). This table lets an account configure
its own ``(provider, api_key, api_base)`` — resolved by a nearest-ancestor
walk up the account tree at model-call time (see
``db.queries.model_providers.resolve_model_provider``), so a child account
without its own row inherits its nearest configured ancestor's, falling back
to the worker's env vars when no account in the chain has one.

``ciphertext``/``nonce`` hold the encrypted ``api_key`` (per-account HKDF
subkey, same scheme as ``vault_credentials``); ``api_base`` is plaintext
proxy/self-hosted routing, stored alongside the key so resolution is
row-atomic — a child's api_base can never combine with an ancestor's api_key
(that combination is a key-exfiltration primitive; see the harness-side
conflict guard). The partial unique index enforces one active row per
``(account, provider)``.

Revision ID: 0140
Revises: 0139
Create Date: 2026-07-09
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0140"
down_revision: str = "0139"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE model_providers (
            id           text PRIMARY KEY,
            account_id   text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            provider     text NOT NULL,
            api_base     text,
            ciphertext   bytea NOT NULL,
            nonce        bytea NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now(),
            archived_at  timestamptz
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX model_providers_account_provider_uniq
            ON model_providers (account_id, provider)
            WHERE archived_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS model_providers")
