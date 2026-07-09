"""Add ``sessions.open_request_scan_floor`` — the monotone scan-floor column
behind the bounded open-obligations anti-join (#1747, sub-issue of the
zero-inference-gap budget epic #1733).

``get_open_obligations`` / ``get_open_request_ids`` (``db/queries/sessions.py``)
anti-join EVERY ``request_opened`` edge a session ever accrued, every step —
unbounded lifetime growth on the hottest path for a re-invoked servicer
session (#1127/#1128). The fix is a monotone per-session floor: readers bound
their scan to ``req.seq >= open_request_scan_floor`` instead of scanning from
seq 1 forever. The floor only ever advances past *answered* edges (never past
a genuinely-open one — see the SEQ-PREFIX INVARIANT and
:func:`aios.db.queries.sessions.advance_open_request_scan_floor`), so its
staleness is perf-only: a stale (low) floor reproduces today's exact O(R)
behavior, never a correctness gap.

Purely additive, metadata-only in PG11+ (``DEFAULT 0`` needs no table
rewrite): ``0`` means "scan from the beginning" — i.e. today's behavior — so
existing rows and any in-flight deploy-window reader see no semantic change
before the column is ratcheted. No backfill: the floor is lazily ratcheted on
each hot session's first post-deploy step (see the companion service-code
change), deliberately sidestepping a backfill-under-lock over large sessions.

Split into its own revision — separate from 0135's ``CREATE INDEX
CONCURRENTLY`` — because a single revision that mixes transactional DDL with
an ``autocommit_block()`` is a partial-apply wedge: the ADD COLUMN commits
when the autocommit block opens, but ``alembic_version`` is stamped only
after ``upgrade()`` returns, so a killed/failed concurrent build leaves the
column committed but the revision unstamped; a re-run then replays the ADD
COLUMN into ``column already exists`` (the #152 dup-0122 hazard class).
Precedent: 0099 / 0128 / 0131 do *only* ``CONCURRENTLY`` per revision — this
revision follows the transactional-DDL half of that split, mirrored by 0135.

The ``ALTER TABLE`` needs ACCESS EXCLUSIVE on ``sessions`` — the hottest
table in the system (``UPDATE sessions ...`` on every event append). Even a
metadata-only ALTER queues behind any in-flight ``sessions`` transaction and,
via lock-queue ordering, blocks *all* subsequent session writes until it
acquires. ``SET LOCAL lock_timeout`` + a short retry-with-backoff loop keeps a
momentary long-running transaction from stalling the whole append path during
deploy — the ALTER simply retries instead of queuing indefinitely.

Rollback is asymmetric: dropping this column while *new* code (which reads
it unconditionally via ``COALESCE((SELECT open_request_scan_floor ...), 0)``)
is still live is a **full outage** (``42703`` on every step / append) — see
the runbook in #1747. Downgrade is marked production-unsafe-while-new-code-live;
do not wire it into an automatic rollback path.

Revision ID: 0134
Revises: 0133
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0134"
down_revision: str = "0133"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# How many attempts (each bounded by the per-attempt lock_timeout below) to
# make before giving up and letting the ALTER's exception propagate normally.
_MAX_ATTEMPTS = 5
# Per-attempt lock wait ceiling — short enough that one blocked ALTER attempt
# can't itself become the long-lived blocker queuing up behind it.
_LOCK_TIMEOUT = "2s"
# Backoff between attempts (seconds), linear — deploy-time migration, not a
# hot path; no need for jitter/exponential sophistication here.
_RETRY_SLEEP_SECONDS = 1.0


def _add_column_with_retry() -> None:
    bind = op.get_bind()
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with bind.begin_nested() if bind.in_transaction() else bind.begin():
                bind.execute(sa.text(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'"))
                bind.execute(
                    sa.text(
                        "ALTER TABLE sessions "
                        "ADD COLUMN IF NOT EXISTS open_request_scan_floor "
                        "bigint NOT NULL DEFAULT 0"
                    )
                )
            return
        except Exception as exc:
            # failure (lock_timeout expiry surfaces as a DB error) should retry;
            # a non-lock error will keep failing identically on every retry and
            # surface once the budget is exhausted below.
            last_exc = exc
            if attempt == _MAX_ATTEMPTS:
                break
            time.sleep(_RETRY_SLEEP_SECONDS)
    assert last_exc is not None
    raise last_exc


def upgrade() -> None:
    _add_column_with_retry()


def downgrade() -> None:
    # PRODUCTION-UNSAFE while new code (which reads this column unconditionally
    # via a COALESCE-guarded scalar subquery) is still live — see the runbook
    # in #1747. New code hits ``42703 undefined_column`` on every step/append
    # the moment this column disappears. Do not auto-run in a live rollback.
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS open_request_scan_floor")
