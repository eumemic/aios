"""A workflow child session's frozen, run-attenuated capability surface (#794).

A run's ``agent()`` child must wield ``agent ∩ run`` authority, not the agent's full
surface — and that clamped surface must be **frozen at spawn** so replay reads the same
value every wake (never re-running the meet against a since-changed agent or operator
default). The child's surface lives on the ``sessions`` row:

* ``tools`` / ``mcp_servers`` / ``http_servers`` (jsonb, nullable) — the frozen meet result.
* ``surface_frozen`` (boolean) — the discriminator. A foreground/operator session leaves
  it ``false`` and its surface columns ``NULL`` (``load_for_session`` reads the live agent).
  A workflow child sets it ``true`` (even when the frozen surface is empty ``[]`` — an
  empty clamp ≠ an absent one), and ``load_for_session`` reads the frozen columns,
  **failing closed** if a ``parent_run_id`` child is somehow not frozen.

Columns are nullable rather than ``NOT NULL DEFAULT '[]'`` precisely so ``surface_frozen``
is the single source of truth for "is this a frozen child" — a defaulted ``'[]'`` would be
ambiguous against a legitimately-empty clamp.

**Grandfather backfill:** in-flight ``parent_run_id`` children that predate this column
read the pinned ``AgentVersion``'s surface verbatim today. Freeze exactly that — the meet
result for them is unknowable retroactively, but their *observed* surface is the pinned
version's, so copying it preserves behavior with zero bricked runs (the fail-closed branch
would otherwise error every in-flight child's next step). A child whose ``agent_versions``
row is missing stays unfrozen → fails closed, which is correct for that corrupt state.

``sessions`` is not the hot ``events`` table, so plain in-transaction DDL is fine.

Revision ID: 0079
Revises: 0078
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0079"
down_revision: str = "0078"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN tools jsonb")
    op.execute("ALTER TABLE sessions ADD COLUMN mcp_servers jsonb")
    op.execute("ALTER TABLE sessions ADD COLUMN http_servers jsonb")
    op.execute("ALTER TABLE sessions ADD COLUMN surface_frozen boolean NOT NULL DEFAULT false")
    # Grandfather in-flight children: freeze the pinned AgentVersion's surface verbatim —
    # exactly what load_for_session returns for them today, so behavior is unchanged.
    op.execute(
        """
        UPDATE sessions s
           SET tools = av.tools,
               mcp_servers = av.mcp_servers,
               http_servers = av.http_servers,
               surface_frozen = TRUE
          FROM agent_versions av
         WHERE s.parent_run_id IS NOT NULL
           AND av.agent_id = s.agent_id
           AND av.version = s.agent_version
           AND av.account_id = s.account_id
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS surface_frozen")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS http_servers")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS mcp_servers")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS tools")
