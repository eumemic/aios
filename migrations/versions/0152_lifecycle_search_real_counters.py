"""Align lifecycle_search redaction with the real cumulative counters.

Revision ID: 0152
Revises: 0151

Fence invariant: ``data - ARRAY[...]`` removes top-level JSON keys only. Writers
must keep sensitive cost and cumulative counters at the top level; nesting one
under another payload key bypasses this view's redaction and requires a new
fence design before deployment.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0152"
down_revision: str = "0151"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BASE_REDACTED_KEYS = (
    "event",
    "request_id",
    "summary",
    "status",
    "awaited",
    "caller",
    "is_error",
    "frozen_surface",
    "output_schema",
    "vault_ids",
    "environment_id",
    "model",
    "model_usage",
    "local_tokens",
    "local_tokens_by_class",
    "cost_usd",
)
_USAGE_COUNTER_KEYS = (
    "cumulative_tokens",
    "cumulative_class_mass",
    "cumulative_messages",
    "cumulative_text_mass",
    "cumulative_tool_result_mass",
    "cumulative_thinking_mass",
    "cumulative_tool_use_mass",
)
REDACTED_KEYS = _BASE_REDACTED_KEYS + _USAGE_COUNTER_KEYS


def _replace_view(redacted_keys: tuple[str, ...]) -> None:
    keys = ",".join(f"'{key}'" for key in redacted_keys)
    detail = f"(data - ARRAY[{keys}])::text"
    marker = "…[truncated]"
    payload = 8192 - len(marker.encode())
    capped = (
        f"CASE WHEN octet_length({detail}) <= 8192 THEN {detail} "
        f"ELSE left({detail}, {payload} - greatest(0, "
        f"octet_length(left({detail}, {payload})) - {payload})) "
        f"|| '{marker}' END"
    )
    op.execute(f"""
    CREATE OR REPLACE VIEW lifecycle_search AS
    SELECT id AS event_id, seq, created_at, data->>'event' AS lifecycle_kind,
      data->>'request_id' AS request_id, data->>'summary' AS summary,
      CASE WHEN data->>'event'='request_response' THEN data->>'is_error' ELSE data->>'status' END AS status,
      CASE WHEN data ? 'awaited' THEN (data->>'awaited')::boolean END AS awaited,
      data->'caller'->>'kind' AS caller_kind, {capped} AS detail_text,
      octet_length({detail}) AS detail_len
    FROM events WHERE session_id=current_setting('app.session_id', true) AND kind='lifecycle'
      AND data->>'event' IN ('request_opened','request_response','turn_ended','tool_confirmed','trigger_fired','trigger_disabled','trigger_enabled')
    """)


def upgrade() -> None:
    _replace_view(REDACTED_KEYS)


def downgrade() -> None:
    _replace_view((*_BASE_REDACTED_KEYS, "cumulative_tokens", "cumulative_class_mass"))
