"""Restore ON DELETE CASCADE on ``bindings.session_id`` (#1095).

Of the ~12 child tables referencing ``sessions(id)``, all but one declare
``ON DELETE CASCADE``. The lone outlier is ``bindings.session_id``: the
0033 connector redesign recreated ``bindings`` with a bare
``session_id text REFERENCES sessions(id)`` (no ``ON DELETE``), regressing
the cascade the *original* ``bindings`` table carried
(``0015``: ``session_id ... REFERENCES sessions(id) ON DELETE CASCADE``).
``delete_session`` has papered over the regression with a hand-``DELETE
FROM bindings`` ever since — an invariant held by one remembered statement
+ its comment, which any new session-deletion path can silently violate.

This migration makes the cascade uniform across all session children by
swapping the bare ``bindings_session_id_fkey`` for the **composite tenant
FK** form ``FOREIGN KEY (session_id, account_id)
REFERENCES sessions(id, account_id) ON DELETE CASCADE`` — the direction
established by ``0093`` for the other secret/session-bearing chains
(``session_vaults`` etc). The composite form addresses the steelman's
tenant-tightness concern structurally (a binding can only be cascaded by
its own tenant's session) while restoring the cascade.

``bindings.account_id`` is ``NOT NULL`` (0044) and ``sessions`` already
carries the ``sessions_id_account_id_key`` UNIQUE (id, account_id) that the
composite FK targets (added by 0093). ``session_id`` is nullable (per_chat
bindings have ``session_id IS NULL``); MATCH SIMPLE (the default) leaves
those rows unconstrained, exactly as the prior single-column FK did.

The constraint is added ``NOT VALID`` then ``VALIDATE``d in a second step
(the ``0093`` precedent) so the validating table scan takes only a
``SHARE UPDATE EXCLUSIVE`` lock rather than holding ``ACCESS EXCLUSIVE``
for the whole scan.

Revision ID: 0105
Revises: 0104
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0105"
down_revision: str = "0104"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE bindings DROP CONSTRAINT bindings_session_id_fkey")
    op.execute(
        "ALTER TABLE bindings ADD CONSTRAINT bindings_session_account_id_fkey "
        "FOREIGN KEY (session_id, account_id) "
        "REFERENCES sessions(id, account_id) ON DELETE CASCADE NOT VALID"
    )
    op.execute("ALTER TABLE bindings VALIDATE CONSTRAINT bindings_session_account_id_fkey")


def downgrade() -> None:
    op.execute("ALTER TABLE bindings DROP CONSTRAINT bindings_session_account_id_fkey")
    op.execute(
        "ALTER TABLE bindings ADD CONSTRAINT bindings_session_id_fkey "
        "FOREIGN KEY (session_id) REFERENCES sessions(id)"
    )
