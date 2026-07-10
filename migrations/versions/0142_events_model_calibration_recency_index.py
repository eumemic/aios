"""Add the ``events_model_calibration_recency_idx`` partial index behind the
recency-ordered per-model token-ratio calibration fetch (#1798 / #1711).

``model_token_class_ratios`` (``src/aios/db/queries/events.py``) fetches the
globally most-recent N successful ``model_request_end`` spans for one model to
fit the per-class token-count coefficients.  It is read on the step-hot window
path (``read_windowed_events`` ← ``harness/loop.py``), so its access cost is
paid synchronously before a model request whenever the 60 s ratio cache misses.

Issue #1798 changed the recency key from ``seq DESC`` to
``ORDER BY created_at DESC, session_id DESC, seq DESC`` because ``seq`` is
session-local and cannot rank spans pooled across sessions.  Migration 0024's
``events_model_request_end_calibration_idx`` is keyed ``((data->>'model'),
seq DESC)``, so it can seek to the model but no longer satisfies the new sort:
the planner has to read **every** qualifying lifetime row for the model and
Sort it before ``LIMIT 1000`` applies.  The limit therefore no longer bounds
the scan, regressing the #1711 timeout protection — and the cost grows without
bound as calibration history accumulates.

This migration adds the composite partial index whose column order matches the
query's access path exactly:

* Leading ``(data->>'model')`` — the sole equality predicate — gives the
  planner an equality seek into the matching rows.
* Trailing ``created_at DESC, session_id DESC, seq DESC`` reproduces the
  ``ORDER BY`` verbatim (matching DESC directions so a forward index scan
  yields rows already in order), so ``LIMIT 1000`` bounds the scan to an index
  seek instead of a full-model scan + Sort.  ``seq`` is the deterministic
  tie-breaker.
* The partial predicate mirrors migration 0024's predicate verbatim — it is a
  strict subset of the query's ``WHERE`` (which adds
  ``data ? 'local_tokens_by_class'`` and the input/local token gates), so the
  planner can prove the partial index applicable while keeping the index to
  exactly the calibration-bearing spans (a small fraction of ``events``).

Purely additive — a single ``CREATE INDEX CONCURRENTLY`` built inside
``op.get_context().autocommit_block()`` so it never takes an ACCESS EXCLUSIVE
lock on the live-written ``events`` table (same pattern as 0024 / 0128).  Safe
in the post-deploy new-code/old-schema window: the index only speeds the read;
the running container works against the old schema until it completes, and the
index is invisible while building.  The now-redundant ``seq``-ordered 0024
index is left in place (never-delete; additive change only).

Revision ID: 0142
Revises: 0141
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0142"
down_revision: str = "0141"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_model_calibration_recency_idx "
            "ON events ((data->>'model'), created_at DESC, session_id DESC, seq DESC) "
            "WHERE kind = 'span' "
            "AND data->>'event' = 'model_request_end' "
            "AND (data->>'is_error')::boolean = false "
            "AND data ? 'local_tokens' "
            "AND data ? 'model'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_model_calibration_recency_idx")
