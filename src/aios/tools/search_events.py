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
On an expected failure (empty query, SQL validation, timeout, query error) raises
:class:`~aios.tools.invoke.ToolBail` (one typed failure channel — #1680); the single
event writer stamps ``is_error``, which is what the ``WHERE is_error`` example queries
below actually filter on.
"""

from __future__ import annotations

import re
from typing import Any

import asyncpg

from aios.harness import runtime
from aios.logging import get_logger
from aios.tools.invoke import ToolBail
from aios.tools.registry import registry

log = get_logger(__name__)

MAX_ROWS = 200
QUERY_TIMEOUT_MS = 10_000

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXECUTE|COPY|GRANT|REVOKE|SET_CONFIG|CALL)\b",
    re.IGNORECASE,
)

# Identifiers that, if referenced, mean the query is reaching outside the
# per-session ``events_search`` view.  Stamping the events table directly
# bypasses the view's ``session_id = current_setting('app.session_id', true)``
# scoping; querying any other application table leaks rows that belong to
# other sessions and other accounts (multi-tenancy).  ``pg_*`` /
# ``information_schema`` are the schema-introspection surfaces that would
# enumerate the rest of the catalogue.  The check runs on the *literal-stripped*
# SQL so a substring inside an ILIKE pattern (e.g. ``'%events%'``) doesn't
# false-positive.
_FORBIDDEN_REFERENCES = re.compile(
    r"\b("
    r"events|sessions|accounts|account_keys|agents|agent_versions|environments|"
    r"connections|connectors|vaults|vault_credentials|memory_stores|memories|"
    r"memory_versions|session_vaults|session_resources|session_memory_stores|"
    r"session_github_repositories|session_templates|github_repositories|"
    r"pending_management_calls|attachments|files|skills|skill_versions|"
    r"procrastinate_\w+|pg_\w+|information_schema|pg_catalog"
    r")\b",
    re.IGNORECASE,
)

# Strip SQL string literals (single-quote and dollar-quoted) and comments so
# the identifier check above runs against the structural SQL only.  Double-
# quoted identifiers are NOT stripped — ``"events"`` is the events table,
# not a string — and the structural regex's ``\b`` boundaries still match
# inside them.
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SINGLE_QUOTE_RE = re.compile(r"'(?:''|[^'])*'")
_DOLLAR_QUOTE_RE = re.compile(r"\$(\w*)\$.*?\$\1\$", re.DOTALL)


def _strip_string_literals(sql: str) -> str:
    """Replace SQL string literals and comments with empty placeholders.

    Double-quoted identifiers are intentionally preserved — they are how
    Postgres references tables/columns with reserved-word or mixed-case
    names, and ``\\b`` word boundaries still match identifiers inside
    quotes.
    """
    sql = _LINE_COMMENT_RE.sub("", sql)
    sql = _BLOCK_COMMENT_RE.sub("", sql)
    sql = _SINGLE_QUOTE_RE.sub("''", sql)
    sql = _DOLLAR_QUOTE_RE.sub("$$$$", sql)
    return sql


def _validate_sql(sql: str) -> str | None:
    """Validate SQL is a safe SELECT query. Returns an error string or None.

    The agent's SQL surface is exclusively the per-session ``events_search``
    view.  Direct access to any other table — including the underlying
    ``events`` table — bypasses the view's per-session scoping and leaks
    rows belonging to other sessions and (post multi-tenancy v1) other
    accounts.  The structural check runs on the literal-stripped SQL so
    substring matches inside ``ILIKE`` patterns don't false-positive.
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed"
    if ";" in stripped:
        return "Multiple statements (semicolons) are not allowed"
    kw_match = _FORBIDDEN_KEYWORDS.search(stripped)
    if kw_match:
        return f"Forbidden keyword '{kw_match.group()}' is not allowed in queries"
    structural = _strip_string_literals(stripped)
    ref_match = _FORBIDDEN_REFERENCES.search(structural)
    if ref_match:
        return (
            f"Identifier {ref_match.group()!r} is not allowed; query the events_search view instead"
        )
    return None


def _format_results(rows: list[asyncpg.Record], truncated: bool) -> str:
    """Format query result rows as a readable text table."""
    if not rows:
        return "No results."
    columns = list(rows[0].keys())
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
) -> tuple[list[asyncpg.Record], bool]:
    """Execute a read-only SQL query scoped to session_id.

    Returns (rows, truncated). Rows are capped at MAX_ROWS; truncated
    is True when the query produced more rows than the cap.
    """
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await conn.execute(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'")
        # set_config() accepts session_id as a parameter, avoiding interpolation.
        await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
        wrapped = f"SELECT * FROM ({sql}) _q LIMIT {MAX_ROWS + 1}"
        rows = await conn.fetch(wrapped)

    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[:MAX_ROWS]
    return rows, truncated


SEARCH_EVENTS_DESCRIPTION = (
    "Query this session's message log using PostgreSQL SQL. events_search "
    "sees every message event for the session — a superset of what's in your "
    "live context window — so use it to recover memory that has scrolled out "
    "or to filter across the session on dimensions your context doesn't "
    "surface directly.\n\n"
    "IMPORTANT — content_text is the RAW stored content, not the rendered "
    "form you see in live context. Channel headers, sender/timestamp lines, "
    "notification markers for non-focal channels, and recap fences are all "
    "applied by the render pipeline at context-build time — they are not in "
    "content_text. To filter by channel or sender, use the promoted columns "
    "below instead of ILIKE on content_text.\n\n"
    "Schema — events_search columns:\n"
    "- id (text)            unique event ID\n"
    "- seq (integer)        gapless sequence number (chronological order)\n"
    "- role (text)          'user', 'assistant', or 'tool'\n"
    "- channel (text)       which channel the event belongs to. For user\n"
    "                       events, the origin channel; for assistant, the\n"
    "                       focal channel when the turn was produced; for\n"
    "                       tool events, the parent assistant's focal\n"
    "                       channel. NULL for events outside any channel.\n"
    "- tool_name (text)     the tool involved. For role='tool' rows it's the\n"
    "                       name of the tool whose RESULT this is. For\n"
    "                       role='assistant' rows with tool_calls, it's the\n"
    "                       first tool_call's function name (multi-tool\n"
    "                       turns only expose the first — the full list\n"
    "                       isn't queryable from this view). NULL otherwise.\n"
    "- is_error (boolean)   TRUE when a tool-result event represents a\n"
    "                       failure; NULL otherwise (successful results omit\n"
    "                       the flag entirely — never FALSE).\n"
    "- sender_name (text)   display name of the human who sent a user\n"
    "                       message, when the connector stamped metadata.\n"
    "                       NULL for assistant/tool events and for legacy\n"
    "                       user events without metadata.\n"
    "- created_at (timestamptz)  when the event was appended\n"
    "- content_text (text)  the raw stored content (see IMPORTANT above)\n\n"
    "NOT in the view (known limitations): the full metadata object on user\n"
    "events (reactions, reply_to, sender_uuid, chat_type, ...), the full\n"
    "tool_calls array on assistant events, span events (cost/timing/tokens),\n"
    "and the rendered channel header. If you need any of those, ask — there\n"
    "isn't currently a way to get them from SQL.\n\n"
    "Role semantics: role='tool' rows are tool RESULTS. Tool CALLS\n"
    "live on assistant rows and are discoverable via tool_name.\n\n"
    "Dialect: PostgreSQL 16. ILIKE, CTEs, window functions, JSON operators\n"
    "(->, ->>, @>) all work. Results capped at 200 rows. SELECT only.\n\n"
    "Examples:\n\n"
    "Messages on a specific channel, most recent first:\n"
    "  SELECT seq, role, sender_name, created_at,\n"
    "         substr(content_text, 1, 200) AS preview\n"
    "  FROM events_search\n"
    "  WHERE channel = 'slack:C0123ABCD'\n"
    "  ORDER BY seq DESC LIMIT 50\n\n"
    "My tool calls of a given kind:\n"
    "  SELECT seq, created_at, substr(content_text, 1, 120) AS preview\n"
    "  FROM events_search\n"
    "  WHERE role = 'assistant' AND tool_name = 'bash'\n"
    "  ORDER BY seq DESC LIMIT 20\n\n"
    "Recent tool failures:\n"
    "  SELECT seq, tool_name, created_at,\n"
    "         substr(content_text, 1, 300) AS error_preview\n"
    "  FROM events_search\n"
    "  WHERE is_error AND created_at > now() - interval '1 hour'\n"
    "  ORDER BY seq DESC\n\n"
    "What did a particular person say this week:\n"
    "  SELECT seq, channel, created_at,\n"
    "         substr(content_text, 1, 300) AS preview\n"
    "  FROM events_search\n"
    "  WHERE sender_name = 'Matt'\n"
    "    AND created_at > now() - interval '7 days'\n"
    "  ORDER BY seq DESC\n\n"
    "Errors by tool (cross-tab):\n"
    "  SELECT tool_name, count(*) AS errors\n"
    "  FROM events_search\n"
    "  WHERE is_error\n"
    "  GROUP BY tool_name\n"
    "  ORDER BY errors DESC\n\n"
    "Keyword search on body text (slow on large sessions — prefer the\n"
    "promoted columns above when they apply):\n"
    "  SELECT seq, role, created_at, substr(content_text, 1, 200) AS preview\n"
    "  FROM events_search\n"
    "  WHERE content_text ILIKE '%docker%'\n"
    "  ORDER BY seq DESC LIMIT 20\n\n"
    "Tips:\n"
    "- ORDER BY seq (or seq DESC) for chronological output — seq is the\n"
    "  canonical ordering, tighter than created_at.\n"
    "- Use substr(content_text, 1, N) in SELECT to keep output compact.\n"
    "- Filter by is_error (not `is_error = TRUE`) — it's a boolean; the NULL\n"
    "  case is implicitly excluded. is_error is stamped for every expected\n"
    "  tool failure going forward; some tool-result events recorded before\n"
    "  that stamp was made uniform may carry NULL and won't match this filter.\n"
    "- Prefer promoted columns (channel, tool_name, is_error, sender_name)\n"
    "  over ILIKE on content_text when they apply — they're indexed and\n"
    "  faster, and they match what you'd mentally ask."
)

SEARCH_EVENTS_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "SQL SELECT query against the events_search view.\n"
                "Columns: id (text), seq (integer), role (text), "
                "channel (text), tool_name (text), is_error (boolean), "
                "sender_name (text), created_at (timestamptz), "
                "content_text (text).\n\n"
                "Examples:\n"
                "  SELECT * FROM events_search "
                "WHERE channel = 'slack:C0123ABCD' ORDER BY seq DESC LIMIT 50\n"
                "  SELECT * FROM events_search "
                "WHERE role = 'assistant' AND tool_name = 'bash' "
                "ORDER BY seq DESC\n"
                "  SELECT tool_name, count(*) FROM events_search "
                "WHERE is_error GROUP BY tool_name"
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
        raise ToolBail("search_events requires a non-empty 'query' string")

    err = _validate_sql(query)
    if err:
        raise ToolBail(f"SQL validation: {err}")

    pool = runtime.require_pool()

    try:
        rows, truncated = await _execute_query(pool, session_id, query)
    except asyncpg.exceptions.QueryCanceledError as exc:
        # Expected refusals (#1680): raise ToolBail so the single writer stamps
        # ``is_error`` — never let the raw asyncpg error escape and evict the sandbox.
        raise ToolBail(f"Query timed out after {QUERY_TIMEOUT_MS}ms") from exc
    except Exception as exc:
        log.warning("search_events.query_failed", error=str(exc))
        raise ToolBail(f"Query failed: {exc}") from exc

    text = _format_results(rows, truncated)
    return {"result": text}


def _register() -> None:
    registry.register(
        name="search_events",
        description=SEARCH_EVENTS_DESCRIPTION,
        parameters_schema=SEARCH_EVENTS_PARAMETERS_SCHEMA,
        handler=search_events_handler,
        transport="both",
    )


_register()
