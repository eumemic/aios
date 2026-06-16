"""Restore ``ON DELETE CASCADE`` on ``bindings.session_id``.

The ``bindings`` table originally (migration 0015) declared
``session_id text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE``.
The 0033 connector redesign *recreated* ``bindings`` and dropped the
cascade, leaving a bare ``session_id text REFERENCES sessions(id)``.
``delete_session`` has compensated ever since with an explicit
hand-``DELETE FROM bindings`` — the lone session-child held by
application vigilance rather than the schema. Every other session-child
FK (events, session_vaults, session_memory_stores, files, chat_sessions,
triggers, routing_rules-via-binding_id, …) already cascades.

This migration makes the cascade uniform again. Following the 0093
composite-FK precedent for session-children, we re-add the FK in its
tenant-scoped composite form ``(session_id, account_id) REFERENCES
sessions(id, account_id) ON DELETE CASCADE`` — structurally matching the
tenant-tightness of the old hand-``DELETE`` (which carried
``AND account_id = $2``) and reusing the ``sessions_id_account_id_key``
unique added by 0093.

``bindings.session_id`` is nullable (``per_chat`` bindings target a
template and leave ``session_id`` NULL). With the default ``MATCH
SIMPLE`` semantics the composite FK is simply not checked when any
referencing column is NULL, so ``per_chat`` rows are unaffected.

The swap uses the ``NOT VALID`` → ``VALIDATE`` two-step (per the 0093
precedent) so the constraint is added without a long ``ACCESS EXCLUSIVE``
lock while existing rows are validated.

Revision ID: 0109
Revises: 0108
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0109"
down_revision: str = "0108"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The bare single-column FK created inline by 0033's ``CREATE TABLE
# bindings`` — Postgres auto-names it ``<table>_<column>_fkey``.
_OLD_FK = "bindings_session_id_fkey"
# The new tenant-scoped composite FK that carries the cascade.
_NEW_FK = "bindings_session_account_id_fkey"


def upgrade() -> None:
    op.execute(f"ALTER TABLE bindings DROP CONSTRAINT {_OLD_FK}")
    op.execute(
        f"ALTER TABLE bindings ADD CONSTRAINT {_NEW_FK} "
        "FOREIGN KEY (session_id, account_id) "
        "REFERENCES sessions(id, account_id) ON DELETE CASCADE NOT VALID"
    )
    op.execute(f"ALTER TABLE bindings VALIDATE CONSTRAINT {_NEW_FK}")


def downgrade() -> None:
    op.execute(f"ALTER TABLE bindings DROP CONSTRAINT {_NEW_FK}")
    # Restore the bare, cascade-less single-column FK as recreated by 0033.
    op.execute(
        f"ALTER TABLE bindings ADD CONSTRAINT {_OLD_FK} "
        "FOREIGN KEY (session_id) REFERENCES sessions(id)"
    )
