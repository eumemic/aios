"""A workflow child session's frozen, clamped model identity (#823).

#794 froze a child's capability *surface* (tools/mcp/http) onto its ``sessions`` row so
replay reads the same authority every wake. The model-identity axis (#823) is the
orthogonal freeze: ``litellm_extra`` (``api_base`` foremost) decides *where the child's
mind runs* — which inference endpoint its entire prompt context is sent to. Today a
named-agent child re-resolves ``litellm_extra`` from the live ``agent_versions`` row at
load time, which is **not replay-sound** (a later ``update_agent`` shifts it) and leaves
no spawn-edge clamp point. Freeze the clamped value here, mirroring the surface snapshot:

* ``litellm_extra`` (jsonb, nullable) — the model identity frozen at spawn, read back by
  ``load_for_session`` instead of the live agent version. ``surface_frozen`` (added by
  0079) stays the single "is this a frozen child" discriminator; this column rides it
  (a frozen child reads the frozen ``litellm_extra``; a non-frozen child reads the live
  agent). Nullable so an empty model identity (``{}``) is representable distinctly from
  an absent freeze, exactly like the surface columns.

**Grandfather backfill:** in-flight ``parent_run_id`` children that predate this column
read the pinned ``AgentVersion``'s ``litellm_extra`` verbatim today. Freeze exactly that
— their *observed* model identity — so behavior is unchanged and no in-flight child's
next step changes its endpoint. A child whose ``agent_versions`` row is missing (or an
agentless generic child, ``agent_id IS NULL``) is left ``NULL`` → ``load_for_session``
reads ``{}`` for it, exactly as today.

``sessions`` is not the hot ``events`` table, so plain in-transaction DDL is fine.

Revision ID: 0104
Revises: 0103
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0104"
down_revision: str = "0103"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN litellm_extra jsonb")
    # Grandfather in-flight named-agent children: freeze the pinned AgentVersion's
    # litellm_extra verbatim — exactly what load_for_session returns for them today.
    op.execute(
        """
        UPDATE sessions s
           SET litellm_extra = av.litellm_extra
          FROM agent_versions av
         WHERE s.parent_run_id IS NOT NULL
           AND s.surface_frozen = TRUE
           AND s.agent_id IS NOT NULL
           AND av.agent_id = s.agent_id
           AND av.version = s.agent_version
           AND av.account_id = s.account_id
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS litellm_extra")
