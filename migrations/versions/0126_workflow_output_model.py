"""Declared effective model on a workflow (the ``workflow:`` binding descriptor).

Part of the **Workflows-as-Models** epic (#1637). A ``workflow:<id>[@version]``
model binding must not silently degrade the capability gates that key on the
literal model string — vision (``supports_vision``), extended-thinking
continuity (``model_descriptor(...).supports_thinking``), and token-window
calibration (``read_windowed_events(model=...)``). The opaque ``workflow:``
string matches none of them, so images get dropped, thinking-blocks stripped,
and token counting reverts to the model-neutral under-counting path.

This adds the **declared effective model** the binding carries: a nullable
``text`` column ``output_model`` on both ``workflows`` (the head) and
``workflow_versions`` (the immutable snapshot, mirroring the existing
``output_schema`` parallel). When set it is the raw provider model the workflow
ultimately emits (e.g. ``anthropic/claude-opus-4-6``); the gates resolve to it
before context-build. ``NULL`` keeps the pre-#1637 degraded posture (the raw
``workflow:`` string drives the gates) — so every existing workflow reads back
unchanged.

``downgrade()`` drops both columns.

Revision ID: 0126
Revises: 0125
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0126"
down_revision: str = "0125"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflows ADD COLUMN output_model text")
    op.execute("ALTER TABLE workflow_versions ADD COLUMN output_model text")


def downgrade() -> None:
    op.execute("ALTER TABLE workflow_versions DROP COLUMN output_model")
    op.execute("ALTER TABLE workflows DROP COLUMN output_model")
