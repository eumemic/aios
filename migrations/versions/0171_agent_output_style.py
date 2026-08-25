"""Add ``output_style`` column to agents + agent_versions.

Purely additive, default 'default' -- every existing agent keeps current
behavior; versioned like every other config field. No CHECK constraint --
the ``OutputStyle`` Literal on the pydantic models is the single
validation point (0139/0111 precedent). Mechanism prose lives in
``aios.harness.concise``.

Revision ID: 0171
Revises: 0169
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0171"
down_revision = "0169"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ADD COLUMN output_style text NOT NULL DEFAULT 'default';")
    op.execute(
        "ALTER TABLE agent_versions ADD COLUMN output_style text NOT NULL DEFAULT 'default';"
    )


def downgrade() -> None:
    # A non-default value carries behavior that the prior schema cannot
    # represent.  Check both tables before either DROP so refusal leaves the
    # schema wholly intact, including when only the second target is unsafe.
    unsafe: list[str] = []
    bind = op.get_bind()
    for table in ("agents", "agent_versions"):
        count = bind.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE output_style <> 'default'")
        ).scalar_one()
        if count:
            unsafe.append(f"{table}.output_style ({count} non-default rows)")
    if unsafe:
        raise RuntimeError(
            "cannot downgrade 0171: output_style is not representable in the prior schema: "
            + ", ".join(unsafe)
        )

    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS output_style;")
    op.execute("ALTER TABLE agent_versions DROP COLUMN IF EXISTS output_style;")
