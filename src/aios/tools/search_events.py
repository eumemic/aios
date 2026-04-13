"""The search_events tool — read-only SQL access to the session event log.

Gives agents the ability to search their own conversation history via SQL
queries against the ``events_search`` view. The view is scoped per-session
using ``SET LOCAL app.session_id``, so the agent can only see its own events.

Safety:
- Only SELECT queries are accepted (validated before execution).
- Queries run inside a READ ONLY transaction.
- A 10-second statement_timeout prevents runaway queries.
- Results are capped at 200 rows.

Return shape: {"result": "<formatted text table>"}
On error: {"error": "..."}
"""

from __future__ import annotations

import re
from typing import Any

import asyncpg

from aios.harness import runtime
from aios.logging import get_logger
from aios.tools.registry import registry

log = get_logger("aios.tools.search_events")

MAX_ROWS = 200
QUERY_TIMEOUT_MS = 10_000

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXECUTE|COPY|GRANT|REVOKE|SET_CONFIG|CALL)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> str | None:
    """Validate SQL is a safe SELECT query. Returns an error string or None."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed"
    if ";" in stripped:
        return "Multiple statements (semicolons) are not allowed"
    m = _FORBIDDEN_KEYWORDS.search(stripped)
    if m:
        return f"Forbidden keyword '{m.group()}' is not allowed in queries"
    return None


def _format_results(rows: list[asyncpg.Record], columns: list[str], truncated: bool) -> str:
    """Format query result rows as a readable text table."""
    if not rows:
        return "No results."
    header = " | ".join(columns)
    divider = "-" * len(header)
    data_lines = [
        " | ".join(str(v) if v is not None else "NULL" for v in row.values()) for row in rows
    ]
    lines = [header, divider, *data_lines]
    result = "\n".join(lines)
    if truncated:
        result += f"\n\n(Results truncated to {MAX_ROWS} rows)"
    return result


async def _execute_query(
    pool: asyncpg.Pool[Any],
    session_id: str,
    sql: str,
) -> tuple[list[asyncpg.Record], list[str]]:
    """Execute a read-only SQL query scoped to session_id.

    Acquires a connection from the pool, starts a READ ONLY transaction,
    sets the session scope and statement timeout, wraps the user query
    in a LIMIT to enforce the row cap, and returns the results.
    """
    async with pool.acquire() as conn:
        await conn.execute("BEGIN READ ONLY")
        try:
            await conn.execute(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'")
            # Use set_config() instead of SET LOCAL so the session_id
            # goes through as a parameter, not string interpolation.
            await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)

            # Wrap user query to enforce row limit (fetch MAX_ROWS+1 to detect
            # truncation). Using text interpolation is safe here because `sql`
            # has already been validated as a SELECT-only query without
            # semicolons, and the LIMIT value is a constant integer.
            wrapped = f"SELECT * FROM ({sql}) _q LIMIT {MAX_ROWS + 1}"
            rows = await conn.fetch(wrapped)
            columns = list(rows[0].keys()) if rows else []
        finally:
            await conn.execute("ROLLBACK")

    return rows, columns


SEARCH_EVENTS_DESCRIPTION = (
    "Query this session's event log using SQL. Use this tool to search past "
    "conversation history — especially useful when earlier messages have fallen "
    "out of the context window.\n\n"
    "Schema — events_search view columns:\n"
    "- id (text): unique event ID\n"
    "- seq (integer): sequence number (chronological order within the session)\n"
    "- kind (text): 'message', 'lifecycle', 'span', or 'interrupt'\n"
    "- role (text): 'user', 'assistant', or 'tool' (NULL for non-message events)\n"
    "- created_at (timestamptz): when the event was created\n"
    "- content_text (text): the message content (or full JSON for non-message events)\n\n"
    "Limits: results capped at 200 rows. Only SELECT queries allowed.\n\n"
    "Common patterns:\n\n"
    "Keyword search (case-insensitive):\n"
    "  SELECT seq, role, created_at, substr(content_text, 1, 200) AS preview\n"
    "  FROM events_search\n"
    "  WHERE content_text ILIKE '%docker%'\n"
    "  ORDER BY seq DESC LIMIT 20\n\n"
    "Find user messages about a topic:\n"
    "  SELECT seq, created_at, substr(content_text, 1, 300) AS preview\n"
    "  FROM events_search\n"
    "  WHERE role = 'user' AND content_text ILIKE '%deploy%'\n"
    "  ORDER BY seq DESC LIMIT 10\n\n"
    "Time-range filter (last 7 days):\n"
    "  SELECT * FROM events_search\n"
    "  WHERE created_at > now() - interval '7 days'\n"
    "  ORDER BY seq\n\n"
    "Count messages by role:\n"
    "  SELECT role, count(*) AS n\n"
    "  FROM events_search\n"
    "  WHERE kind = 'message'\n"
    "  GROUP BY role\n\n"
    "Most recent messages:\n"
    "  SELECT seq, role, created_at, substr(content_text, 1, 100) AS preview\n"
    "  FROM events_search\n"
    "  WHERE kind = 'message'\n"
    "  ORDER BY seq DESC LIMIT 10\n\n"
    "Exact date range:\n"
    "  SELECT * FROM events_search\n"
    "  WHERE created_at >= '2026-03-01'\n"
    "    AND created_at < '2026-03-08'\n"
    "  ORDER BY seq\n\n"
    "Tips:\n"
    "- ORDER BY seq for chronological output (seq is the canonical ordering)\n"
    "- Use LIMIT to control result size (hard cap is 200 rows)\n"
    "- Use substr(content_text, 1, N) for content previews to keep output compact\n"
    "- ILIKE '%term%' is case-insensitive; use multiple ILIKE clauses for OR searches\n"
    "- Filter by kind = 'message' to exclude lifecycle/span events\n"
    "- Filter by role = 'user'/'assistant'/'tool' to narrow to specific speakers"
)

SEARCH_EVENTS_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "SQL SELECT query against the events_search view.\n"
                "Available columns: id (text), seq (integer), kind (text), "
                "role (text), created_at (timestamptz), content_text (text).\n\n"
                "Examples:\n"
                "  SELECT * FROM events_search WHERE content_text ILIKE '%keyword%' LIMIT 20\n"
                "  SELECT role, count(*) FROM events_search WHERE kind = 'message' GROUP BY role\n"
                "  SELECT * FROM events_search "
                "WHERE created_at > now() - interval '7 days' ORDER BY seq"
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def search_events_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the search_events tool."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "search_events requires a non-empty 'query' string"}

    err = _validate_sql(query)
    if err:
        return {"error": f"SQL validation: {err}"}

    pool = runtime.require_pool()

    try:
        rows, columns = await _execute_query(pool, session_id, query)
    except asyncpg.exceptions.QueryCanceledError:
        return {"error": f"Query timed out after {QUERY_TIMEOUT_MS}ms"}
    except Exception as exc:
        log.warning("search_events.query_failed", error=str(exc))
        return {"error": f"Query failed: {exc}"}

    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[:MAX_ROWS]

    text = _format_results(rows, columns, truncated)
    return {"result": text}


def _register() -> None:
    registry.register(
        name="search_events",
        description=SEARCH_EVENTS_DESCRIPTION,
        parameters_schema=SEARCH_EVENTS_PARAMETERS_SCHEMA,
        handler=search_events_handler,
    )


_register()
