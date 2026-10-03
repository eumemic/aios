"""Content-addressed request parts, per account (#2471).

Revision ID: 0186
Revises: 0185

A composed request's system prompt, tool list and params aren't in the event
log: they come from agent versions that can be pruned, live MCP discovery,
connector tables and worker memory. Request capture stores each one here once
per account, keyed by the sha256 of its bytes (``aios.harness.request_capture``),
and the request span references them by hash. An unchanged system prompt is
therefore stored once, not once per turn.

``body`` is ``bytea``: the stored bytes are exactly the hashed bytes, including
NUL characters and lone surrogates that jsonb rejects. The key is per account,
never global, so tenants can't learn about each other's content through
deduplication. Rows go when their account is deleted. Nothing else removes them:
a blob may be referenced by any number of requests.

Additive: an application image from before this revision never reads or writes
the table.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0186"
down_revision: str | None = "0185"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE request_blobs (
            account_id text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            sha256 text NOT NULL,
            body bytea NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (account_id, sha256)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE request_blobs")
