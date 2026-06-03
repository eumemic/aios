"""Interactive OAuth flow state for vault-credential "Connect".

The console's "Connect" UX lets a user authorize an MCP server interactively:
the server initiates an OAuth 2.0 authorization-code flow (optionally with
Dynamic Client Registration + PKCE), redirects the user to sign in at the
provider, then exchanges the returned code for tokens and stores them as an
``oauth2_refresh`` vault credential.

That flow spans two HTTP requests (start → user consents at the provider →
complete), so the transient state must live server-side between them. Each
row holds the random ``state`` (CSRF), the ``redirect_uri`` (which MUST be
byte-identical across registration, /authorize and the token exchange — so it
is persisted and reused on complete, never re-accepted from the client), and
an encrypted blob carrying the PKCE ``code_verifier``, the (possibly
dynamically-registered) client_id/secret, and the resolved endpoints. The blob
is encrypted with the same per-account subkey as vault credentials because it
contains the code_verifier and an optional client_secret.

Rows are single-use (deleted on complete) and short-lived (``expires_at`` ~10
min); ``start`` opportunistically prunes expired rows so the table stays small
without a separate janitor.

Revision ID: 0061
Revises: 0060
Create Date: 2026-06-02
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0061"
down_revision: str = "0060"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE oauth_flows (
            id            text PRIMARY KEY,
            account_id    text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            vault_id      text NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
            target_url    text NOT NULL,
            state         text NOT NULL,
            redirect_uri  text NOT NULL,
            ciphertext    bytea NOT NULL,
            nonce         bytea NOT NULL,
            created_at    timestamptz NOT NULL DEFAULT now(),
            expires_at    timestamptz NOT NULL,
            UNIQUE (account_id, state)
        )
    """)
    # The complete-flow lookup is keyed by (account_id, vault_id, state); the
    # UNIQUE(account_id, state) above already covers it. This index serves the
    # expiry prune (DELETE ... WHERE expires_at < now()).
    op.execute("""
        CREATE INDEX oauth_flows_expires_at ON oauth_flows (expires_at)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS oauth_flows")
