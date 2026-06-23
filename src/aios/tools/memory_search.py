"""The memory_search tool — full-text recall over attached memory stores.

Gives agents ranked, stemmed keyword search across the content of every
memory store attached to their session, backed by a Postgres FTS (GIN)
index on ``memories.content`` (migration 0119). This replaces the linear
``rg -i '<keyword>' /mnt/memory/`` grep over network-backed mounts with a
proper full-text query: ``websearch_to_tsquery`` + ``ts_rank``.

Surface (v1, LOCKED): a NARROWED ``{query: str}`` wrapper — NOT a
raw-SELECT surface. The handler runs a fixed, parameterised query against
the per-session ``memories_search`` view:

    SELECT ... , ts_rank(content_tsv, q) AS rank
    FROM memories_search, websearch_to_tsquery('english', $1) q
    WHERE content_tsv @@ q
    ORDER BY rank DESC
    LIMIT MAX_ROWS

The model's input is bound as ``$1`` (never interpolated), so there is no
SQL surface to validate — but the query still runs inside a READ ONLY
transaction so no mutation is possible even in principle. (Raw-SELECT
parity with ``search_events`` is the documented deferred follow-up.)

Safety:
- The user query is a bind parameter — no SQL injection surface.
- The query runs inside a READ ONLY transaction (no mutation possible).
- A 10-second statement_timeout prevents runaway queries.
- The ``memories_search`` view is scoped per-session via
  ``SET LOCAL app.session_id`` joined through ``session_memory_stores``,
  so a session only sees memories of its own attached stores.
- Results are capped at MAX_ROWS rows, ordered by ts_rank DESC.

Return shape: {"result": "<formatted text table>"}
On error: {"error": "..."}
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.harness import runtime
from aios.logging import get_logger
from aios.tools.registry import registry

log = get_logger(__name__)

MAX_ROWS = 200
QUERY_TIMEOUT_MS = 10_000

# Fixed, parameterised search query against the per-session ``memories_search``
# view. The model's keyword string is bound as ``$1`` — never interpolated —
# so there is no SQL surface to validate. ``websearch_to_tsquery`` parses the
# Google-style query syntax (quoted phrases, ``or``, ``-`` negation) the model
# is likely to reach for; matches are ranked by ``ts_rank`` (DESC) and capped.
_SEARCH_SQL = f"""
    SELECT
        m.store,
        m.path,
        m.content_size_bytes,
        m.updated_at,
        ts_rank(m.content_tsv, q) AS rank,
        m.content
    FROM memories_search m,
         websearch_to_tsquery('english', $1) AS q
    WHERE m.content_tsv @@ q
    ORDER BY rank DESC, m.updated_at DESC
    LIMIT {MAX_ROWS + 1}
"""

# How much of each memory's body to show in the result preview. The full
# content can be up to 100 KiB; a compact preview keeps the tool output
# readable and lets the model follow up with ``read`` on the path.
_PREVIEW_CHARS = 500


def _format_results(rows: list[asyncpg.Record], truncated: bool) -> str:
    """Format query result rows as a readable text table.

    The raw ``content`` column is truncated to a compact preview so a few
    large memories don't blow out the tool output; the model can ``read``
    the path for the full body.
    """
    if not rows:
        return "No results."
    lines: list[str] = []
    for row in rows:
        content = row["content"] or ""
        preview = content[:_PREVIEW_CHARS]
        if len(content) > _PREVIEW_CHARS:
            preview += "…"
        preview = preview.replace("\n", " ")
        lines.append(
            f"store={row['store']} path={row['path']} "
            f"size={row['content_size_bytes']} "
            f"updated_at={row['updated_at']} rank={row['rank']:.4f}\n"
            f"  {preview}"
        )
    result = "\n".join(lines)
    if truncated:
        result += f"\n\n(Results truncated to {MAX_ROWS} rows)"
    return result


async def _execute_query(
    pool: asyncpg.Pool[Any],
    session_id: str,
    query: str,
) -> tuple[list[asyncpg.Record], bool]:
    """Run the fixed full-text search, scoped to session_id.

    Returns (rows, truncated). Rows are capped at MAX_ROWS; truncated is
    True when the query produced more rows than the cap. The query runs in
    a READ ONLY transaction so no mutation is possible, and the user's
    keyword string is bound as ``$1`` (no interpolation).
    """
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await conn.execute(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'")
        # set_config() accepts session_id as a parameter, avoiding interpolation.
        await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
        rows = await conn.fetch(_SEARCH_SQL, query)

    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[:MAX_ROWS]
    return rows, truncated


MEMORY_SEARCH_DESCRIPTION = (
    "Full-text search across the content of every memory store attached to "
    "this session. Backed by a Postgres FTS (GIN) index — ranked and "
    "stemmed, far better than grepping the memory mounts. Use it to recall "
    "stored notes/cards by keyword when you don't know the exact path.\n\n"
    "Pass a keyword query in the Google-style websearch syntax: bare words "
    "are AND-ed, \"quoted phrases\" match adjacent words, 'or' is "
    "disjunction, and a leading '-' negates a term. Matching is stemmed and "
    "case-insensitive (English).\n\n"
    "Results are ranked by relevance (ts_rank DESC), capped at 200 rows, and "
    "scoped to THIS session's attached stores only — you never see another "
    "session's or tenant's memories. Each row shows the store, path, size, "
    "last-updated time, rank, and a content preview; use the read tool on "
    "the path for the full body.\n\n"
    "Examples:\n"
    "  deploy rollback procedure\n"
    '  "incident postmortem" database\n'
    "  oncall -slack"
)

MEMORY_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Full-text keyword query (websearch syntax: bare words are "
                "AND-ed, \"quoted phrases\" match adjacency, 'or' is "
                "disjunction, leading '-' negates). Matched against memory "
                "content; results are ranked by relevance."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def memory_search_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the memory_search tool."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "memory_search requires a non-empty 'query' string"}

    pool = runtime.require_pool()

    try:
        rows, truncated = await _execute_query(pool, session_id, query)
    except asyncpg.exceptions.QueryCanceledError:
        return {"error": f"Query timed out after {QUERY_TIMEOUT_MS}ms"}
    except Exception as exc:
        log.warning("memory_search.query_failed", error=str(exc))
        return {"error": f"Query failed: {exc}"}

    text = _format_results(rows, truncated)
    return {"result": text}


def _register() -> None:
    registry.register(
        name="memory_search",
        description=MEMORY_SEARCH_DESCRIPTION,
        parameters_schema=MEMORY_SEARCH_PARAMETERS_SCHEMA,
        handler=memory_search_handler,
        transport="both",
    )


_register()
