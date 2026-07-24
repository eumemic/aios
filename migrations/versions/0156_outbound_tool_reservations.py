"""Durable outbound tool quota reservations.

The redesign of the per-``(session_id, verb)`` outbound dispatch quota
(issue #1903): admission is an atomic purge+count+insert against this table
inside one short, DB-only transaction — not a rolling count over ``events``.
A row is durable evidence that one outbound dispatch was admitted (capacity
consumed at admission); quota refusals insert nothing. ``verb`` is the
canonical connector verb (MCP server namespace stripped), so sibling
effectors share one quota. Rows age out of the rolling window and are purged
opportunistically per key at admission; session deletion cascades.

``state`` records the reservation lifecycle: ``admitted`` (capacity owned;
the external call may have started — a crash leaves this state and the row
conservatively counts until the window rolls past it) → ``completed``
(connector invocation returned; best-effort observability mark).

Revision ID: 0156
Revises: 0155
"""

from alembic import op

revision = "0156"
down_revision = "0155"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE outbound_tool_reservations (
            id          text PRIMARY KEY DEFAULT gen_random_uuid()::text,
            session_id  text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            verb        text NOT NULL,
            state       text NOT NULL DEFAULT 'admitted'
                CHECK (state IN ('admitted', 'completed')),
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX outbound_tool_reservations_key_idx "
        "ON outbound_tool_reservations (session_id, verb, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE outbound_tool_reservations")
