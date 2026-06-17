"""Per-session outbound-suppression mode (#710).

Adds the ``outbound_suppression`` column to ``sessions``: the last remaining
substrate dependency for jarbot's v1→v2 shard migration. When a session runs
with this set ``'on'``, the tool brokers (HTTP via ``http_request``, MCP via
``mcp call``) intercept side-effecting outbound calls — http_server writes
(POST/PUT/PATCH/DELETE by default, per-route overridable) and all mcp_server
calls (default-deny; per-tool ``read_allow`` opts in known reads) — returning a
synthesized success and appending a ``tool_call_suppressed`` audit span instead
of letting the request leave the broker. Reads pass through against real
credentials. This enables tier-3 safe testing (real vaults + memory, no real
sends) and live-cutover parallel runs (v2 suppressed alongside v1, then flip).

* ``outbound_suppression`` (text, NOT NULL, default ``'off'``) — the per-session
  mode. A plain text column with a CHECK to the two-value domain {off, on} so a
  later "fully sandboxed mode" can extend the domain without a type migration.
  Default ``'off'`` makes every existing and new session normal; suppression is
  strictly opt-in. Backfill is the column default — no in-flight session changes
  behavior.

``sessions`` is not the hot ``events`` table, so plain in-transaction DDL is fine.

Revision ID: 0106
Revises: 0104
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0106"
down_revision: str = "0104"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE sessions "
        "ADD COLUMN outbound_suppression text NOT NULL DEFAULT 'off' "
        "CONSTRAINT sessions_outbound_suppression_chk "
        "CHECK (outbound_suppression IN ('off', 'on'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS outbound_suppression")
