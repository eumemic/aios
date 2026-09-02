"""browser_calls: admit the ``peek`` control method.

0175 pinned ``browser_calls.method`` to the five control-plane methods of
the day with a CHECK. #2336 added ``peek`` (the product's read-only look at
a page) to the dispatcher and the API but not to the constraint, so every
``GET /v1/browser/peek`` failed at the INSERT with a CheckViolationError —
a 500 the mocked route/dispatch tests could not see. Found the first time
the route hit a real database (the jarbot local devstack, 2026-09-02).

Revision ID: 0178
Revises: 0177
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0178"
down_revision: str = "0177"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_METHODS_WITH_PEEK = "('open', 'close', 'status', 'revoke_site', 'clear_state', 'peek')"
_METHODS_WITHOUT_PEEK = "('open', 'close', 'status', 'revoke_site', 'clear_state')"


def _swap_check(methods: str) -> None:
    # A tiny table with short-lived rows: the ACCESS EXCLUSIVE lock is
    # momentary, and there is no backfill.
    op.execute("ALTER TABLE browser_calls DROP CONSTRAINT browser_calls_method_check")
    op.execute(
        "ALTER TABLE browser_calls ADD CONSTRAINT browser_calls_method_check "
        f"CHECK (method IN {methods})"
    )


def upgrade() -> None:
    _swap_check(_METHODS_WITH_PEEK)


def downgrade() -> None:
    # Any peek rows still pending would violate the narrower check; they are
    # ephemeral (the submitter's wait is seconds), so clear them first.
    op.execute("DELETE FROM browser_calls WHERE method = 'peek'")
    _swap_check(_METHODS_WITHOUT_PEEK)
