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
- Results are capped at MAX_ROWS rows, ordered by the fused RRF score DESC.

Hybrid recall (#1629): the fixed query fuses the ``ts_rank`` arm with a
pgvector cosine arm over ``memories.embedding`` (migration 0196) via
Reciprocal-Rank Fusion. See ``_SEARCH_SQL``.

Return shape: :class:`~aios.tools.registry.ToolResult` whose ``content`` is the
formatted text table as plain multi-line text (#2291) — not a JSON envelope, so a
spilled result stays line-oriented for ``grep``/``sed``/``wc -l``/``read``.
On an expected failure (empty query, timeout, query error) raises
:class:`~aios.tools.invoke.ToolBail` (one typed failure channel — #1680); the single
event writer stamps ``is_error``.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.harness import runtime
from aios.logging import get_logger
from aios.services.memory_embedding import RRF_K, Embedded, embed_memory_content, vector_literal
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry

log = get_logger(__name__)

MAX_ROWS = 200
QUERY_TIMEOUT_MS = 10_000

# Fixed, parameterised HYBRID search query (#1629) against the per-session
# ``memories_search`` view. The model's keyword string is bound as ``$1`` and
# the query embedding (pgvector text form, or NULL) as ``$2`` — never
# interpolated. Two ranking arms, both read through the SAME view so
# per-session/per-tenant isolation is inherited unchanged:
#   * ``fts`` — ``websearch_to_tsquery`` + ``ts_rank`` (stemmed keyword recall);
#   * ``vec`` — pgvector cosine distance ``<=>`` over ``embedding``.
# They are fused by Reciprocal-Rank Fusion with K = RRF_K (scale-free: by rank,
# never by raw score). ``$2 IS NOT NULL`` makes the vec arm contribute nothing
# when the query embed failed — a bare ``ORDER BY embedding <=> NULL`` would
# still return embedded rows with NULL distances.
_SEARCH_SQL = f"""
    WITH fts AS (
        SELECT f.id, row_number() OVER (ORDER BY f.score DESC, f.id) AS r
        FROM (
            SELECT s.id, ts_rank(s.content_tsv, q) AS score
            FROM memories_search s, websearch_to_tsquery('english', $1) AS q
            WHERE s.content_tsv @@ q
            ORDER BY score DESC, s.id
            LIMIT {MAX_ROWS}
        ) f
    ),
    vec AS (
        SELECT v.id, row_number() OVER (ORDER BY v.dist, v.id) AS r
        FROM (
            SELECT s.id, s.embedding <=> $2::vector AS dist
            FROM memories_search s
            WHERE $2::vector IS NOT NULL AND s.embedding IS NOT NULL
            ORDER BY dist, s.id
            LIMIT {MAX_ROWS}
        ) v
    )
    SELECT
        m.store,
        m.path,
        m.content_size_bytes,
        m.updated_at,
        (COALESCE(1.0 / ({RRF_K} + fts.r), 0)
         + COALESCE(1.0 / ({RRF_K} + vec.r), 0))::float8 AS rank,
        m.content
    FROM memories_search m
    LEFT JOIN fts ON fts.id = m.id
    LEFT JOIN vec ON vec.id = m.id
    WHERE fts.id IS NOT NULL OR vec.id IS NOT NULL
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

    The query embedding is computed OUTSIDE the transaction (model I/O never
    holds a connection). Read-path failure policy is DECOUPLED: if the embed
    is ``Skipped`` the vector is bound as NULL and the search degrades to the
    FTS arm alone — search never fails because the embedder is down.
    """
    outcome = await embed_memory_content(query)
    qvec = vector_literal(outcome.vector) if isinstance(outcome, Embedded) else None
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await conn.execute(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'")
        # set_config() accepts session_id as a parameter, avoiding interpolation.
        await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
        rows = await conn.fetch(_SEARCH_SQL, query, qvec)

    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[:MAX_ROWS]
    return rows, truncated


MEMORY_SEARCH_DESCRIPTION = (
    "Hybrid keyword + semantic search across the content of every memory "
    "store attached to this session. Fuses a Postgres full-text (stemmed "
    "keyword) ranking with a vector-similarity (meaning) ranking, so a memory "
    "can match even when it shares no words with your query. Far better than "
    "grepping the memory mounts. Use it to recall stored notes/cards when you "
    "don't know the exact path.\n\n"
    "Pass a keyword query in the Google-style websearch syntax: bare words "
    "are AND-ed, \"quoted phrases\" match adjacent words, 'or' is "
    "disjunction, and a leading '-' negates a term. Matching is stemmed and "
    "case-insensitive (English).\n\n"
    "Results are ranked by fused relevance (reciprocal-rank fusion), capped at 200 rows, and "
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


async def memory_search_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    """Handler for the memory_search tool."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolBail("memory_search requires a non-empty 'query' string")

    pool = runtime.require_pool()

    try:
        rows, truncated = await _execute_query(pool, session_id, query)
    except asyncpg.exceptions.QueryCanceledError as exc:
        # Expected refusals (#1680): raise ToolBail so the single writer stamps
        # ``is_error`` — never let the raw asyncpg error escape and evict the sandbox.
        raise ToolBail(f"Query timed out after {QUERY_TIMEOUT_MS}ms") from exc
    except Exception as exc:
        log.warning("memory_search.query_failed", error=str(exc))
        raise ToolBail(f"Query failed: {exc}") from exc

    # Plain-string ``ToolResult`` content, never ``{"result": text}`` (#2291) — see the
    # note on ``search_events_handler``: only a ``ToolResult`` reaches the consumer
    # verbatim; any other return type is JSON-encoded and collapses the table onto one
    # escaped line.
    return ToolResult(content=_format_results(rows, truncated))


def _register() -> None:
    registry.register(
        name="memory_search",
        description=MEMORY_SEARCH_DESCRIPTION,
        parameters_schema=MEMORY_SEARCH_PARAMETERS_SCHEMA,
        handler=memory_search_handler,
        transport="both",
    )


_register()
