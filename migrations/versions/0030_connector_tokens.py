"""Per-connection scoped bearer tokens for connector containers.

Per #301: each connector container authenticates as the connection it
serves, not with the global ``AIOS_API_KEY``.  A token resolves to one
``connection_id``; routes that take a ``ConnectorAuthDep`` use that id
to scope the request (e.g. inbound goes to that connection only; the
calls stream emits only that connection's pending tool calls).

Plaintext tokens are returned ONCE on issue and never stored — the row
holds a SHA-256 hash that the auth dep compares against.  The
plaintext format (``aios_conn_<base64url>``) is opaque to the DB.

Revocation is soft (``revoked_at``) so audit trails survive.

Revision ID: 0030
Revises: 0029
Create Date: 2026-05-08
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0030"
down_revision: str = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE connector_tokens (
            id            text PRIMARY KEY,
            connection_id text NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            token_hash    text NOT NULL UNIQUE,
            label         text,
            created_at    timestamptz NOT NULL DEFAULT now(),
            last_used_at  timestamptz,
            revoked_at    timestamptz
        )
        """
    )
    op.execute(
        "CREATE INDEX connector_tokens_connection_id_idx "
        "ON connector_tokens (connection_id) WHERE revoked_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS connector_tokens")
