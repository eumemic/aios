"""Strip the dropped ``allow_mcp_servers`` key from persisted ``environments.config`` JSONB.

#1477 collapses the dead ``NetworkPolicy`` value-object family and deletes the
API-honored-but-runtime-ignored ``allow_mcp_servers`` boolean from
``LimitedNetworking`` (``models/environments.py``). ``LimitedNetworking`` (and
``EnvironmentConfig``) carry ``model_config = extra="forbid"``, so a persisted
``environments`` row whose ``config.networking`` object still carries the dropped
``allow_mcp_servers`` key would fail ``EnvironmentConfig.model_validate`` on read
after the code change ships.

Per the chairman clean-break directive (issue #1477, 2026-06-23): do NOT add a
read-tolerance validator. Instead this data-only migration rewrites any persisted
``environments`` row whose ``config.networking`` still carries
``allow_mcp_servers`` (strips the key via JSONB ``#-`` path removal), filtered to
only the affected rows, so zero rows remain on the old shape after deploy. The
``environments`` table is small and the UPDATE is set-based with no per-row
correlated subqueries.

No schema change. ``downgrade()`` is a no-op: the key is retired, nothing to
restore (it drove no behavior).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0129"
down_revision: str = "0128"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Strip config.networking.allow_mcp_servers from any row still carrying it.
    # ``#- '{networking,allow_mcp_servers}'`` removes the key at that JSONB path;
    # the WHERE filters to only rows that actually carry it (set-based, no
    # per-row correlated subquery).
    op.execute(
        r"""
        UPDATE environments
        SET config = config #- '{networking,allow_mcp_servers}'
        WHERE config #> '{networking,allow_mcp_servers}' IS NOT NULL
        """
    )


def downgrade() -> None:
    # Forward-only: allow_mcp_servers is retired (drove no behavior), nothing to restore.
    pass
