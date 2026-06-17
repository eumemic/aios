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

This migration makes the cascade uniform again by re-adding the FK in
exactly the *single-column* form the original 0015 table declared:
``session_id REFERENCES sessions(id) ON DELETE CASCADE``.

A tenant-scoped composite ``(session_id, account_id) REFERENCES
sessions(id, account_id)`` (the 0093 precedent for some other
session-children) is deliberately **not** used here: ``bindings`` is the
one session-child whose ``account_id`` is rewritten *independently* of
its ``session_id`` by ``reparent_connection`` (a cross-account connection
move carries the active binding to the destination account while its
``session_id`` keeps pointing at the source-account session). A composite
FK would make that established reparent contract impossible — the
``(session_id, account_id)`` pair would momentarily reference a
``(session, destination_account)`` tuple that does not exist in
``sessions``. The single-column FK gives the cascade-on-session-delete
this migration is about without entangling ``bindings.account_id``.

``bindings.session_id`` is nullable (``per_chat`` bindings target a
template and leave ``session_id`` NULL). A NULL referencing column simply
isn't checked by the FK, so ``per_chat`` rows are unaffected.

The swap uses the ``NOT VALID`` → ``VALIDATE`` two-step (per the 0093
precedent) so the constraint is added without a long ``ACCESS EXCLUSIVE``
lock while existing rows are validated.

Revision ID: 0110
Revises: 0109
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0110"
down_revision: str = "0109"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The bare single-column FK created inline by 0033's ``CREATE TABLE
# bindings`` — Postgres auto-names it ``<table>_<column>_fkey``. We reuse
# the same name for the cascade-carrying single-column FK so the schema
# stays byte-identical to 0015's shape (only the ON DELETE action differs).
_FK = "bindings_session_id_fkey"


def upgrade() -> None:
    op.execute(f"ALTER TABLE bindings DROP CONSTRAINT {_FK}")
    op.execute(
        f"ALTER TABLE bindings ADD CONSTRAINT {_FK} "
        "FOREIGN KEY (session_id) "
        "REFERENCES sessions(id) ON DELETE CASCADE NOT VALID"
    )
    op.execute(f"ALTER TABLE bindings VALIDATE CONSTRAINT {_FK}")


def downgrade() -> None:
    op.execute(f"ALTER TABLE bindings DROP CONSTRAINT {_FK}")
    # Restore the bare, cascade-less single-column FK as recreated by 0033.
    op.execute(
        f"ALTER TABLE bindings ADD CONSTRAINT {_FK} "
        "FOREIGN KEY (session_id) REFERENCES sessions(id)"
    )
