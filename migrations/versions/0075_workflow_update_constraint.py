"""Workflows become updatable in place: name-unique, not (name, version)-unique.

``update_workflow`` bumps ``workflows.version`` in place (agent-style optimistic
concurrency; no snapshot table — runs already snapshot script + surface at launch).
Under in-place bumps the version-qualified unique is wrong: renaming workflow A (at v3)
to workflow B's name (at v1) would NOT collide on ``(account_id, name, version)``,
leaving two live workflows with one name. Swap to the agent-style name constraint.

The ``workflows`` table is small (definitions, not events); a brief ACCESS EXCLUSIVE
for the constraint swap is fine.

Revision ID: 0075
Revises: 0074
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0075"
down_revision: str = "0074"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflows DROP CONSTRAINT workflows_account_id_name_version_key")
    op.execute(
        "ALTER TABLE workflows ADD CONSTRAINT workflows_account_id_name_key "
        "UNIQUE (account_id, name)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE workflows DROP CONSTRAINT workflows_account_id_name_key")
    op.execute(
        "ALTER TABLE workflows ADD CONSTRAINT workflows_account_id_name_version_key "
        "UNIQUE (account_id, name, version)"
    )
