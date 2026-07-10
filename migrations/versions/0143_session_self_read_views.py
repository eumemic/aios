"""Session-scoped tool-call, span, lifecycle, and schema-help search views.

Revision ID: 0143
Revises: 0142
"""
from __future__ import annotations
from collections.abc import Sequence
from alembic import op

revision: str = "0143"
down_revision: str = "0142"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Remove at least one byte per excess character; this is UTF-8 safe and ensures
# the resulting cell (including marker) never exceeds its byte budget.
def _capped(expr: str, budget: int) -> str:
    marker = "…[truncated]"
    payload = budget - len(marker.encode())
    return f"CASE WHEN octet_length({expr}) <= {budget} THEN {expr} ELSE left({expr}, {payload} - greatest(0, octet_length(left({expr}, {payload})) - {payload})) || '{marker}' END"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS events_session_span_seq_idx ON events (session_id, seq) WHERE kind = 'span'")

    args = "tc.call->'function'->>'arguments'"
    result = "r.data->>'content'"
    op.execute(f"""
    CREATE VIEW tool_calls_search AS
    SELECT a.id AS event_id, a.seq, a.created_at, tc.ordinality AS call_ordinal,
      a.channel, tc.call->>'id' AS tool_call_id,
      tc.call->'function'->>'name' AS tool_name,
      {_capped(args, 16384)} AS arguments_text,
      octet_length({args}) AS args_len,
      encode(sha256(convert_to({args}, 'UTF8')), 'hex') AS args_sha256,
      r.seq AS result_seq, r.created_at AS result_created_at,
      r.is_error AS result_is_error, {_capped(result, 16384)} AS result_text
    FROM events a
    CROSS JOIN LATERAL jsonb_array_elements(a.data->'tool_calls') WITH ORDINALITY AS tc(call, ordinality)
    LEFT JOIN events r ON r.session_id=a.session_id AND r.kind='message' AND r.role='tool'
      AND r.data->>'tool_call_id'=tc.call->>'id'
    WHERE a.session_id=current_setting('app.session_id', true)
      AND a.kind='message' AND a.role='assistant' AND a.data ? 'tool_calls'
    """)
    op.execute("""
    CREATE VIEW spans_search AS
    SELECT id AS event_id, seq, created_at, data->>'event' AS span_kind,
      focal_channel_at_arrival AS focal_channel, data->>'tool_call_id' AS tool_call_id,
      data->>'tool_name' AS tool_name, data->>'start_event_id' AS start_event_id,
      CASE WHEN data ? 'is_error' THEN (data->>'is_error')::boolean END AS is_error
    FROM events WHERE session_id=current_setting('app.session_id', true) AND kind='span'
    """)
    detail = "(data - ARRAY['event','request_id','summary','status','awaited','caller','is_error','frozen_surface','output_schema','vault_ids','environment_id','model','model_usage','local_tokens','local_tokens_by_class','cost_usd'])::text"
    op.execute(f"""
    CREATE VIEW lifecycle_search AS
    SELECT id AS event_id, seq, created_at, data->>'event' AS lifecycle_kind,
      data->>'request_id' AS request_id, data->>'summary' AS summary,
      CASE WHEN data->>'event'='request_response' THEN data->>'is_error' ELSE data->>'status' END AS status,
      CASE WHEN data ? 'awaited' THEN (data->>'awaited')::boolean END AS awaited,
      data->'caller'->>'kind' AS caller_kind, {_capped(detail, 8192)} AS detail_text,
      octet_length({detail}) AS detail_len
    FROM events WHERE session_id=current_setting('app.session_id', true) AND kind='lifecycle'
      AND data->>'event' IN ('request_opened','request_response','turn_ended','tool_confirmed','trigger_fired','trigger_completed')
    """)

    columns = {
      'events_search': [('id','text','message event id'),('seq','bigint','session order'),('role','text','message role'),('channel','text','channel stamp'),('tool_name','text','promoted tool name'),('is_error','boolean','tool failure'),('sender_name','text','sender display name'),('created_at','timestamptz','append time'),('content_text','text','raw message content')],
      'tool_calls_search': [('event_id','text','assistant event id'),('seq','bigint','assistant sequence'),('created_at','timestamptz','append time'),('call_ordinal','bigint','array position; cursor is (seq,call_ordinal)'),('channel','text','assistant channel'),('tool_call_id','text','call/span join key'),('tool_name','text','function name'),('arguments_text','text','arguments, capped at 16 KiB'),('args_len','integer','full UTF-8 byte length'),('args_sha256','text','full arguments SHA-256'),('result_seq','bigint','paired result sequence'),('result_created_at','timestamptz','paired result time'),('result_is_error','boolean','paired result failure'),('result_text','text','paired result, capped at 16 KiB')],
      'spans_search': [('event_id','text','span event id'),('seq','bigint','session order'),('created_at','timestamptz','append time'),('span_kind','text','span event kind'),('focal_channel','text','channel at arrival'),('tool_call_id','text','call join key'),('tool_name','text','payload tool name'),('start_event_id','text','end-to-start link'),('is_error','boolean','payload failure flag')],
      'lifecycle_search': [('event_id','text','lifecycle event id'),('seq','bigint','session order'),('created_at','timestamptz','append time'),('lifecycle_kind','text','allowlisted lifecycle kind'),('request_id','text','request edge id'),('summary','text','safe summary'),('status','text','turn status or response is_error'),('awaited','boolean','whether response is awaited'),('caller_kind','text','caller category'),('detail_text','text','redacted payload, capped at 8 KiB'),('detail_len','integer','full redacted byte length')],
    }
    rows=[]
    for relation, cols in columns.items():
        for name, typ, semantics in cols:
            rows.append("(" + ",".join("'"+v.replace("'","''")+"'" for v in (relation,name,typ,semantics,'')) + ")")
    rows += [
      "('tool_calls_search','__example__','example','keyset pagination','SELECT * FROM tool_calls_search WHERE (seq,call_ordinal) > (100,2) ORDER BY seq,call_ordinal LIMIT 50')",
      "('spans_search','__example__','example','windowed non-CTE duration join','SELECT e.seq, e.created_at-s.created_at AS duration FROM spans_search e JOIN spans_search s ON s.event_id=e.start_event_id WHERE e.seq BETWEEN 100 AND 200')",
      "('events_search','__example__','example','channel address forms drift; suffix match','SELECT * FROM events_search WHERE channel LIKE ''%/12345'' ORDER BY seq DESC LIMIT 20')",
      "('tool_calls_search','__cast_guard__','example','only cast complete JSON arguments','SELECT CASE WHEN args_len <= 16384 THEN arguments_text::jsonb END FROM tool_calls_search')",
      "('lifecycle_search','__scope__','note','request edges are logged by the servicer session','SELECT * FROM lifecycle_search ORDER BY seq DESC LIMIT 50')",
    ]
    op.execute("CREATE VIEW search_views_help AS SELECT * FROM (VALUES " + ','.join(rows) + ") AS h(relation_name,column_name,data_type,semantics,example_sql) WHERE current_setting('app.session_id', true) IS NOT NULL")


def downgrade() -> None:
    for view in ('search_views_help','lifecycle_search','spans_search','tool_calls_search'):
        op.execute(f"DROP VIEW IF EXISTS {view}")
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_session_span_seq_idx")
