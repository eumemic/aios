"""Add ``sessions.open_tool_call_floor_seq`` — the sweep-maintained ghost-scan floor (#1746).

Sub-issue of the zero-inference-gap budget epic (#1733). The ghost-repair trio
(``GHOST_ASST_SQL`` / ``ALL_RESULT_ROWS_SQL`` / ``GHOST_LIFECYCLE_SQL`` in
``harness/sweep.py``) currently scans a session's ENTIRE tool-call history on
every tail-sweep tool completion and every 30s cross-session sweep, for every
session with ``open_tool_call_count > 0``. This column adds a proven lower
bound on the oldest OPEN tool call's ``seq`` so the trio can be bounded to
``e.seq >= s.open_tool_call_floor_seq`` instead of scanning from the start of
the log.

Column semantics
-----------------
``open_tool_call_floor_seq`` is written in EXACTLY ONE place: the ghost sweep
(:func:`aios.harness.sweep.find_and_repair_ghosts`), which advances it
``GREATEST``-only from a reconciliation it already computes (the oldest
still-open assistant tool_call batch it found in the current scan). It is
**never** stamped by the write path (``append_event``) off the
``open_tool_call_count == 0`` transition — that edge is not a trustworthy
signal of "nothing open" (see the issue body's invariant 0: the counter can
undercount via id-blind dedup-skip decrements, so ``count == 0`` does NOT
imply no open call). A write-path stamp keyed off that edge was evaluated and
REJECTED during red-team as unsound (it can permanently exclude a
genuinely-orphaned older batch — the exact permanent-wedge class ghost repair
exists to prevent).

``0`` means "unbounded scan" (today's behavior) — the safe default for every
existing session until its first post-deploy ghost sweep with an open batch
advances it. No backfill is needed or possible: the floor value is *derived*
by the sweep from the event log, not from any existing column.

Migration mechanics
--------------------
Pattern: ``0066_session_status_scalars.py`` (additive scalar column with a
default, no backfill needed here since ``0`` is the correct starting value for
every row — unlike 0066's five columns, which required a set-based backfill
from the event log to reflect ALREADY-observed history).

Metadata-only ``ADD COLUMN ... DEFAULT 0`` — no table rewrite (SAFE per the
migration-skill classification: fixed-width column with a non-volatile
default), but still takes a brief ACCESS EXCLUSIVE lock on the hot ``sessions``
table to update the catalog. ``sessions`` receives a write on every
``append_event`` call (the seq-allocating UPDATE), so an ACCESS EXCLUSIVE
grab that has to queue behind a long-running ``sessions`` transaction would
itself queue every subsequent ``append_event`` behind it — the wedged-``aios
migrate`` failure class. Mitigated with ``SET LOCAL lock_timeout`` + a bounded
retry loop: the ALTER either acquires the lock quickly (the common case, since
the table is not usually mid-long-transaction) or backs off and retries rather
than blocking indefinitely and stalling the fleet.

Rollout ordering (BLOCKING — see the issue body's Rollout section in full):
this column must land in prod BEFORE the code that reads it
(``find_and_repair_ghosts``) is deployed — ``aios-worker`` boots straight into
new code with no migration step of its own, so a naive single-image promote
would serve code that references a column the post-deploy ``aios migrate``
hasn't created yet. The sweep code changes in this same PR catch a missing
column (``UndefinedColumnError``) and fail closed to the pre-fix unbounded
scan rather than aborting the whole cross-session sweep.

``downgrade()`` drops the column. Per the asymmetric-rollback guidance: a code
rollback is always safe first (old code ignores the column; a stale-low floor
during any window is just a safe over-scan, never an under-scan). Only run
``downgrade()`` once no new-code (column-reading) process remains — dropping
the column while new code is still live reintroduces the exact
``UndefinedColumnError`` this migration's ordering is designed to avoid.

Revision ID: 0134
Revises: 0133

Renumbered 0132 -> 0134 at build time: master's 0133
(``sessions_channels_array.py``) also revises 0131, creating a migration-chain
fork with this issue's original 0132 head. Chained after 0133 per the issue's
own L2-7 disposition ("confirm alembic head is still 0130 at merge and
renumber if another 013x lands first").
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from alembic import op

revision: str = "0134"
down_revision: str = "0133"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Keep the lock-acquisition wait short and retry rather than blocking
# indefinitely behind a long-running ``sessions`` transaction (the
# wedged-``aios migrate`` guidance referenced in the module docstring).
_LOCK_TIMEOUT_MS = 3_000
_MAX_ATTEMPTS = 20
_RETRY_SLEEP_SECONDS = 2.0


def _add_column_with_retry() -> None:
    bind = op.get_bind()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with bind.begin_nested():
                bind.exec_driver_sql(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT_MS}ms'")
                bind.exec_driver_sql(
                    "ALTER TABLE sessions "
                    "ADD COLUMN IF NOT EXISTS open_tool_call_floor_seq "
                    "BIGINT NOT NULL DEFAULT 0"
                )
            return
        except Exception:
            if attempt == _MAX_ATTEMPTS:
                raise
            # asyncpg/psycopg surfaces lock_timeout as a QueryCanceled-class
            # error; retry unconditionally (bounded by _MAX_ATTEMPTS) rather
            # than pattern-matching the driver-specific exception text.
            time.sleep(_RETRY_SLEEP_SECONDS)


def upgrade() -> None:
    _add_column_with_retry()


def downgrade() -> None:
    # Asymmetric rollback (see module docstring): only safe to run once no
    # new-code (column-reading) process remains. Code rollback alone is
    # always safe and should happen first.
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS open_tool_call_floor_seq")
