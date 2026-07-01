"""Per-content-class cumulative token mass + cumulative message count on ``events``.

Removes the two O(session-size) terms from the per-turn context read path
(issue #1657), both self-flagged in ``queries/events.py``.

**PRIMARY — ``cumulative_*_mass`` (4 columns).** ``_retained_class_mass``
re-derived each message's neutral token mass from the ``cumulative_tokens``
delta, classified it by role+data, and summed per class with a full-session
``LAG() OVER (ORDER BY seq)`` WindowAgg over the *whole* message slate —
genuinely O(session-size) (measured 3.8s / 90k rows / 46 MB spill on Ultron).
The single ``cumulative_tokens`` running TOTAL cannot be decomposed by class
after the fact, so we extend the same append-time running-sum mechanism to be
per-content-class: four ``bigint`` running sums, one per class
(``text``/``tool_result``/``thinking``/``tool_use``), classified at append
time with the exact same CASE the query used. Because the mass is pooled over
the WHOLE slate, the answer is just the latest message row's four cumulative
totals — one index seek, O(1). Storage is four columns (not a JSONB blob) so
each stays a plain ``bigint`` the append UPDATE increments without JSON
round-trips.

**SECONDARY — ``cumulative_messages``.** The omitted-message ``count(*)
FILTER (role IN ('user','assistant'))`` under ``cumulative_tokens <= drop``
(the ``events.py`` in-code comment's "cumulative_messages counter") becomes a
running count of user/assistant messages, read at the boundary row = O(1).

All five columns are ``NULL`` on non-message events and on pre-backfill
message rows (mirroring ``cumulative_tokens`` — see migration 0012). The read
path already falls back to the full-scan estimator whenever cumulative data is
absent, so a rolling deploy / un-backfilled tail is safe. We do NOT backfill
historical rows here: the running sums are seeded from the latest row's value
at append time, so once the first new message lands after this migration the
counters are live for that session's tail; older sessions transparently keep
using the fallback until they churn. This matches how 0012 shipped
``cumulative_tokens`` (forward-only).

``downgrade()`` drops the five columns.

Revision ID: 0127
Revises: 0126
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0127"
down_revision: str = "0126"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE events "
        "ADD COLUMN cumulative_messages bigint, "
        "ADD COLUMN cumulative_text_mass bigint, "
        "ADD COLUMN cumulative_tool_result_mass bigint, "
        "ADD COLUMN cumulative_thinking_mass bigint, "
        "ADD COLUMN cumulative_tool_use_mass bigint"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE events "
        "DROP COLUMN IF EXISTS cumulative_messages, "
        "DROP COLUMN IF EXISTS cumulative_text_mass, "
        "DROP COLUMN IF EXISTS cumulative_tool_result_mass, "
        "DROP COLUMN IF EXISTS cumulative_thinking_mass, "
        "DROP COLUMN IF EXISTS cumulative_tool_use_mass"
    )
