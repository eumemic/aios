"""Focal-channel attention model.

Adds state + per-event stamps for the focal-channel redesign of
connector-aware sessions (issue #29, focal-channel plan):

* ``sessions.focal_channel`` — the bound channel the agent's attention
  is currently directed at. NULL is a valid "phone down" state.
* ``events.orig_channel`` — the channel a user event originated from
  (derived from ``metadata["channel"]`` at append time).
* ``events.focal_channel_at_arrival`` — the session's focal channel at
  the moment the event was appended. Rendering (full vs notification)
  is a deterministic function of ``(orig_channel, focal_channel_at_arrival)``.
* ``channel_bindings.notification_mode`` — per-binding noisy/silent flag.
  Silent bindings don't produce inline notification markers and are
  rendered quieter in the ephemeral tail block.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0017"
down_revision: str = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ADD COLUMN focal_channel text")
    op.execute("ALTER TABLE events ADD COLUMN orig_channel text")
    op.execute("ALTER TABLE events ADD COLUMN focal_channel_at_arrival text")
    op.execute(
        "ALTER TABLE channel_bindings "
        "ADD COLUMN notification_mode text NOT NULL DEFAULT 'focal_candidate'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE channel_bindings DROP COLUMN IF EXISTS notification_mode")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS focal_channel_at_arrival")
    op.execute("ALTER TABLE events DROP COLUMN IF EXISTS orig_channel")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS focal_channel")
