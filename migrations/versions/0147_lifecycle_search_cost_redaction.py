"""Close lifecycle_search cumulative usage redaction gap.

Revision ID: 0147
Revises: 0146
"""

from alembic import op

revision = "0147"
down_revision = "0146"
branch_labels = None
depends_on = None

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
_USAGE_COUNTER_KEYS = ("cumulative_tokens", "cumulative_class_mass")


def _replace_view(redacted_keys: tuple[str, ...]) -> None:
    keys = ",".join(f"'{key}'" for key in redacted_keys)
    detail = f"(data - ARRAY[{keys}])::text"
    # Match migration 0144's UTF-8-safe cap.  Cutting the encoded bytea at an
    # arbitrary byte boundary can split a multibyte character, and the marker
    # must be included in the 8 KiB budget.
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
    _replace_view(_BASE_REDACTED_KEYS + _USAGE_COUNTER_KEYS)


def downgrade() -> None:
    _replace_view(_BASE_REDACTED_KEYS)
