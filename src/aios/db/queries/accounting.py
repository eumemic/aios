"""Creation-edge inference accounting queries (#2151).

Sessions and workflow runs are two node types in one tree.  Every traversal is
account-scoped and uses recursive ``UNION`` (not ``UNION ALL``), so a malformed
legacy cycle terminates and a node is counted at most once per root.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from aios.errors import AiosError
from aios.models.accounting import (
    AttributedUsage,
    UsageConsumer,
    UsageCounters,
    UsageMetric,
    UsageNodeRef,
    UsageRate,
)

_COUNTER_NAMES = (
    "cost_microusd",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _counters(row: Any, prefix: str, *, tokens_complete: bool = True) -> UsageCounters:
    return UsageCounters(
        **{name: int(row[f"{prefix}_{name}"] or 0) for name in _COUNTER_NAMES},
        tokens_complete=tokens_complete,
    )


def _rate(counters: UsageCounters, *, window_seconds: int, observed_seconds: int) -> UsageRate:
    scale = 3600 / max(1, observed_seconds)
    return UsageRate(
        window_seconds=window_seconds,
        observed_seconds=observed_seconds,
        complete=observed_seconds >= window_seconds,
        cost_microusd_per_hour=counters.cost_microusd * scale,
        input_tokens_per_hour=counters.input_tokens * scale,
        output_tokens_per_hour=counters.output_tokens * scale,
        cache_read_input_tokens_per_hour=counters.cache_read_input_tokens * scale,
        cache_creation_input_tokens_per_hour=counters.cache_creation_input_tokens * scale,
    )


def _attributed(row: Any, *, window_seconds: int) -> AttributedUsage:
    observed_seconds = int(row["observed_seconds"])
    own_window = _counters(row, "own_window")
    subtree_window = _counters(row, "subtree_window")
    return AttributedUsage(
        own=_counters(row, "own", tokens_complete=bool(row["own_tokens_complete"])),
        subtree=_counters(row, "subtree", tokens_complete=bool(row["subtree_tokens_complete"])),
        own_rate=_rate(
            own_window, window_seconds=window_seconds, observed_seconds=observed_seconds
        ),
        subtree_rate=_rate(
            subtree_window, window_seconds=window_seconds, observed_seconds=observed_seconds
        ),
    )


_BATCH_USAGE_SQL = r"""
WITH RECURSIVE
roots(kind, id) AS (
    SELECT * FROM unnest($1::text[], $2::text[])
),
tree(root_kind, root_id, kind, id) AS (
    SELECT r.kind, r.id, r.kind, r.id
      FROM roots r
     WHERE (r.kind = 'session' AND EXISTS (
                SELECT 1 FROM sessions s WHERE s.id = r.id AND s.account_id = $3
           ))
        OR (r.kind = 'run' AND EXISTS (
                SELECT 1 FROM wf_runs w WHERE w.id = r.id AND w.account_id = $3
           ))
    UNION
    SELECT t.root_kind, t.root_id, child.kind, child.id
      FROM tree t
      JOIN LATERAL (
           SELECT 'session'::text AS kind, s.id
             FROM sessions s
            WHERE s.account_id = $3
              AND ((t.kind = 'session' AND s.creator_session_id = t.id)
                OR (t.kind = 'run' AND s.creator_run_id = t.id))
           UNION ALL
           SELECT 'run'::text AS kind, w.id
             FROM wf_runs w
            WHERE w.account_id = $3
              AND ((t.kind = 'session' AND w.creator_session_id = t.id)
                OR (t.kind = 'run' AND w.creator_run_id = t.id))
      ) child ON TRUE
),
node_usage(kind, id, cost_microusd, input_tokens, output_tokens,
           cache_read_input_tokens, cache_creation_input_tokens, tokens_complete) AS (
    SELECT 'session', s.id, s.cost_microusd, s.input_tokens, s.output_tokens,
           s.cache_read_input_tokens, s.cache_creation_input_tokens, TRUE
      FROM sessions s WHERE s.account_id = $3
    UNION ALL
    SELECT 'run', w.id, w.call_llm_cost_microusd, w.call_llm_input_tokens,
           w.call_llm_output_tokens, w.call_llm_cache_read_input_tokens,
           w.call_llm_cache_creation_input_tokens, w.call_llm_tokens_complete
      FROM wf_runs w WHERE w.account_id = $3
),
window_node(kind, id, cost_microusd, input_tokens, output_tokens,
            cache_read_input_tokens, cache_creation_input_tokens) AS (
    SELECT CASE WHEN l.session_id IS NOT NULL THEN 'session' ELSE 'run' END,
           COALESCE(l.session_id, l.run_id), SUM(l.cost_microusd),
           SUM(l.input_tokens), SUM(l.output_tokens),
           SUM(l.cache_read_input_tokens), SUM(l.cache_creation_input_tokens)
      FROM inference_usage_ledger l
     WHERE l.account_id = $3
       AND l.occurred_at >= now() - ($4 * interval '1 second')
     GROUP BY 1, 2
),
totals AS (
    SELECT t.root_kind, t.root_id,
           SUM(n.cost_microusd)::bigint AS subtree_cost_microusd,
           SUM(n.input_tokens)::bigint AS subtree_input_tokens,
           SUM(n.output_tokens)::bigint AS subtree_output_tokens,
           SUM(n.cache_read_input_tokens)::bigint AS subtree_cache_read_input_tokens,
           SUM(n.cache_creation_input_tokens)::bigint AS subtree_cache_creation_input_tokens,
           BOOL_AND(n.tokens_complete) AS subtree_tokens_complete,
           SUM(COALESCE(wn.cost_microusd, 0))::bigint AS subtree_window_cost_microusd,
           SUM(COALESCE(wn.input_tokens, 0))::bigint AS subtree_window_input_tokens,
           SUM(COALESCE(wn.output_tokens, 0))::bigint AS subtree_window_output_tokens,
           SUM(COALESCE(wn.cache_read_input_tokens, 0))::bigint
               AS subtree_window_cache_read_input_tokens,
           SUM(COALESCE(wn.cache_creation_input_tokens, 0))::bigint
               AS subtree_window_cache_creation_input_tokens
      FROM tree t
      JOIN node_usage n ON n.kind = t.kind AND n.id = t.id
      LEFT JOIN window_node wn ON wn.kind = t.kind AND wn.id = t.id
     GROUP BY t.root_kind, t.root_id
),
coverage AS (
    SELECT usage_ledger_started_at AS coverage_started_at,
           LEAST(
               $4,
               GREATEST(
                   1,
                   FLOOR(EXTRACT(EPOCH FROM (now() - usage_ledger_started_at)))::integer
               )
           ) AS observed_seconds
      FROM accounts WHERE id = $3
)
SELECT r.kind AS root_kind, r.id AS root_id,
       own.cost_microusd AS own_cost_microusd,
       own.input_tokens AS own_input_tokens,
       own.output_tokens AS own_output_tokens,
       own.cache_read_input_tokens AS own_cache_read_input_tokens,
       own.cache_creation_input_tokens AS own_cache_creation_input_tokens,
       own.tokens_complete AS own_tokens_complete,
       total.subtree_cost_microusd,
       total.subtree_input_tokens,
       total.subtree_output_tokens,
       total.subtree_cache_read_input_tokens,
       total.subtree_cache_creation_input_tokens,
       total.subtree_tokens_complete,
       COALESCE(own_window.cost_microusd, 0)::bigint AS own_window_cost_microusd,
       COALESCE(own_window.input_tokens, 0)::bigint AS own_window_input_tokens,
       COALESCE(own_window.output_tokens, 0)::bigint AS own_window_output_tokens,
       COALESCE(own_window.cache_read_input_tokens, 0)::bigint
           AS own_window_cache_read_input_tokens,
       COALESCE(own_window.cache_creation_input_tokens, 0)::bigint
           AS own_window_cache_creation_input_tokens,
       total.subtree_window_cost_microusd,
       total.subtree_window_input_tokens,
       total.subtree_window_output_tokens,
       total.subtree_window_cache_read_input_tokens,
       total.subtree_window_cache_creation_input_tokens,
       coverage.coverage_started_at, coverage.observed_seconds
  FROM roots r
  JOIN node_usage own ON own.kind = r.kind AND own.id = r.id
  JOIN totals total ON total.root_kind = r.kind AND total.root_id = r.id
  LEFT JOIN window_node own_window ON own_window.kind = r.kind AND own_window.id = r.id
  CROSS JOIN coverage
"""


async def usage_for_nodes(
    conn: asyncpg.Connection[Any],
    roots: list[UsageNodeRef],
    *,
    account_id: str,
    window_seconds: int,
) -> dict[tuple[str, str], AttributedUsage]:
    """Return coherent own/subtree usage for many roots in one SQL statement."""
    if not roots:
        return {}
    rows = await conn.fetch(
        _BATCH_USAGE_SQL,
        [root.kind for root in roots],
        [root.id for root in roots],
        account_id,
        window_seconds,
    )
    return {
        (str(row["root_kind"]), str(row["root_id"])): _attributed(
            row, window_seconds=window_seconds
        )
        for row in rows
    }


async def usage_for_node(
    conn: asyncpg.Connection[Any],
    root: UsageNodeRef,
    *,
    account_id: str,
    window_seconds: int,
) -> AttributedUsage | None:
    """Point form of :func:`usage_for_nodes`."""
    return (
        await usage_for_nodes(conn, [root], account_id=account_id, window_seconds=window_seconds)
    ).get((root.kind, root.id))


_SESSION_STATUS_SQL = """
CASE WHEN s.archived_at IS NOT NULL THEN 'archived'
     WHEN s.last_error_seq > 0 AND s.last_error_seq > s.last_user_seq THEN 'errored'
     WHEN (s.last_stimulus_seq > s.last_reacted_seq OR s.open_tool_call_count > 0)
          THEN 'active'
     ELSE 'idle' END
"""


def _ranked_consumers_sql(metric: UsageMetric) -> str:
    metric_column = {
        "cost_microusd": "subtree_window_cost_microusd",
        "total_tokens": "subtree_window_total_tokens",
    }[metric]
    return rf"""
WITH RECURSIVE
tree(root_kind, root_id, kind, id) AS (
    SELECT 'session', s.id, 'session', s.id
      FROM sessions s
     WHERE s.account_id = $1
       AND s.creator_session_id IS NULL AND s.creator_run_id IS NULL
    UNION
    SELECT 'run', w.id, 'run', w.id
      FROM wf_runs w
     WHERE w.account_id = $1
       AND w.creator_session_id IS NULL AND w.creator_run_id IS NULL
    UNION
    SELECT t.root_kind, t.root_id, child.kind, child.id
      FROM tree t
      JOIN LATERAL (
           SELECT 'session'::text AS kind, s.id
             FROM sessions s
            WHERE s.account_id = $1
              AND ((t.kind = 'session' AND s.creator_session_id = t.id)
                OR (t.kind = 'run' AND s.creator_run_id = t.id))
           UNION ALL
           SELECT 'run'::text AS kind, w.id
             FROM wf_runs w
            WHERE w.account_id = $1
              AND ((t.kind = 'session' AND w.creator_session_id = t.id)
                OR (t.kind = 'run' AND w.creator_run_id = t.id))
      ) child ON TRUE
),
nodes(kind, id, parent_kind, parent_id, label, status, created_at, archived_at,
      cost_microusd, input_tokens, output_tokens, cache_read_input_tokens,
      cache_creation_input_tokens, tokens_complete) AS (
    SELECT 'session', s.id,
           CASE WHEN s.creator_session_id IS NOT NULL THEN 'session'
                WHEN s.creator_run_id IS NOT NULL THEN 'run' END,
           COALESCE(s.creator_session_id, s.creator_run_id),
           COALESCE(s.title, s.agent_id, s.id), {_SESSION_STATUS_SQL},
           s.created_at, s.archived_at, s.cost_microusd, s.input_tokens,
           s.output_tokens, s.cache_read_input_tokens, s.cache_creation_input_tokens, TRUE
      FROM sessions s WHERE s.account_id = $1
    UNION ALL
    SELECT 'run', w.id,
           CASE WHEN w.creator_session_id IS NOT NULL THEN 'session'
                WHEN w.creator_run_id IS NOT NULL THEN 'run' END,
           COALESCE(w.creator_session_id, w.creator_run_id),
           COALESCE(w.workflow_id, 'inline workflow ' || w.id), w.status,
           w.created_at, w.archived_at, w.call_llm_cost_microusd,
           w.call_llm_input_tokens, w.call_llm_output_tokens,
           w.call_llm_cache_read_input_tokens, w.call_llm_cache_creation_input_tokens,
           w.call_llm_tokens_complete
      FROM wf_runs w WHERE w.account_id = $1
),
window_node(kind, id, cost_microusd, input_tokens, output_tokens,
            cache_read_input_tokens, cache_creation_input_tokens) AS (
    SELECT CASE WHEN l.session_id IS NOT NULL THEN 'session' ELSE 'run' END,
           COALESCE(l.session_id, l.run_id), SUM(l.cost_microusd),
           SUM(l.input_tokens), SUM(l.output_tokens),
           SUM(l.cache_read_input_tokens), SUM(l.cache_creation_input_tokens)
      FROM inference_usage_ledger l
     WHERE l.account_id = $1
       AND l.occurred_at >= now() - ($2 * interval '1 second')
     GROUP BY 1, 2
),
rollup AS (
    SELECT t.root_kind, t.root_id,
           SUM(n.cost_microusd)::bigint AS subtree_cost_microusd,
           SUM(n.input_tokens)::bigint AS subtree_input_tokens,
           SUM(n.output_tokens)::bigint AS subtree_output_tokens,
           SUM(n.cache_read_input_tokens)::bigint AS subtree_cache_read_input_tokens,
           SUM(n.cache_creation_input_tokens)::bigint
               AS subtree_cache_creation_input_tokens,
           BOOL_AND(n.tokens_complete) AS subtree_tokens_complete,
           SUM(COALESCE(wn.cost_microusd, 0))::bigint
               AS subtree_window_cost_microusd,
           SUM(COALESCE(wn.input_tokens, 0))::bigint
               AS subtree_window_input_tokens,
           SUM(COALESCE(wn.output_tokens, 0))::bigint
               AS subtree_window_output_tokens,
           SUM(COALESCE(wn.input_tokens, 0) + COALESCE(wn.output_tokens, 0))::bigint
               AS subtree_window_total_tokens,
           SUM(COALESCE(wn.cache_read_input_tokens, 0))::bigint
               AS subtree_window_cache_read_input_tokens,
           SUM(COALESCE(wn.cache_creation_input_tokens, 0))::bigint
               AS subtree_window_cache_creation_input_tokens
      FROM tree t
      JOIN nodes n ON n.kind = t.kind AND n.id = t.id
      LEFT JOIN window_node wn
        ON wn.kind = t.kind AND wn.id = t.id
     GROUP BY t.root_kind, t.root_id
),
coverage AS (
    SELECT usage_ledger_started_at AS coverage_started_at,
           LEAST(
               $2,
               GREATEST(
                   1,
                   FLOOR(EXTRACT(EPOCH FROM (now() - usage_ledger_started_at)))::integer
               )
           ) AS observed_seconds
      FROM accounts WHERE id = $1
),
rankable AS (
    SELECT n.*, r.*,
           n.cost_microusd AS own_cost_microusd,
           n.input_tokens AS own_input_tokens,
           n.output_tokens AS own_output_tokens,
           n.cache_read_input_tokens AS own_cache_read_input_tokens,
           n.cache_creation_input_tokens AS own_cache_creation_input_tokens,
           n.tokens_complete AS own_tokens_complete,
           COALESCE(wn.cost_microusd, 0)::bigint AS own_window_cost_microusd,
           COALESCE(wn.input_tokens, 0)::bigint AS own_window_input_tokens,
           COALESCE(wn.output_tokens, 0)::bigint AS own_window_output_tokens,
           COALESCE(wn.cache_read_input_tokens, 0)::bigint
               AS own_window_cache_read_input_tokens,
           COALESCE(wn.cache_creation_input_tokens, 0)::bigint
               AS own_window_cache_creation_input_tokens,
           SUM(r.{metric_column}) OVER ()::bigint AS pool_window_metric
      FROM nodes n
      JOIN rollup r ON r.root_kind = n.kind AND r.root_id = n.id
      LEFT JOIN window_node wn ON wn.kind = n.kind AND wn.id = n.id
     WHERE n.parent_id IS NULL
),
graph_state AS (
    SELECT (SELECT COUNT(*) FROM nodes) = (SELECT COUNT(*) FROM tree) AS graph_complete,
           (SELECT COUNT(*) FROM nodes)::bigint AS graph_node_count,
           (SELECT COUNT(*) FROM tree)::bigint AS graph_traversal_rows
)
SELECT ranked.*, coverage.coverage_started_at, coverage.observed_seconds,
       graph_state.graph_complete, graph_state.graph_node_count,
       graph_state.graph_traversal_rows
  FROM graph_state
  CROSS JOIN coverage
  LEFT JOIN LATERAL (
       SELECT * FROM rankable
        ORDER BY {metric_column} DESC, created_at DESC, id DESC
        LIMIT $3
  ) ranked ON TRUE
 ORDER BY ranked.{metric_column} DESC NULLS LAST,
          ranked.created_at DESC NULLS LAST, ranked.id DESC NULLS LAST
"""


async def ranked_consumers(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    window_seconds: int,
    metric: UsageMetric,
    limit: int,
) -> tuple[datetime, float, list[UsageConsumer]]:
    """Rank additive root consumers by rolling subtree rate in one view."""
    rows = await conn.fetch(_ranked_consumers_sql(metric), account_id, window_seconds, limit)
    if rows and not bool(rows[0]["graph_complete"]):
        raise AiosError(
            "usage creation graph contains a rootless component",
            detail={
                "account_id": account_id,
                "node_count": int(rows[0]["graph_node_count"]),
                "reachable_count": int(rows[0]["graph_traversal_rows"]),
            },
        )
    if rows:
        coverage_started_at = rows[0]["coverage_started_at"]
        observed_seconds = int(rows[0]["observed_seconds"])
    else:
        coverage_started_at = await conn.fetchval(
            "SELECT usage_ledger_started_at FROM accounts WHERE id = $1", account_id
        )
        observed_seconds = max(
            1,
            min(
                window_seconds,
                int(
                    (datetime.now(coverage_started_at.tzinfo) - coverage_started_at).total_seconds()
                ),
            ),
        )

    ranked_rows = [row for row in rows if row["id"] is not None]
    pool_window = int(ranked_rows[0]["pool_window_metric"] or 0) if ranked_rows else 0
    scale = 3600 / max(1, observed_seconds)
    items: list[UsageConsumer] = []
    for rank, row in enumerate(ranked_rows, start=1):
        usage = _attributed(row, window_seconds=window_seconds)
        window_value = (
            int(row["subtree_window_cost_microusd"])
            if metric == "cost_microusd"
            else int(row["subtree_window_total_tokens"])
        )
        items.append(
            UsageConsumer(
                rank=rank,
                kind=row["kind"],
                id=row["id"],
                label=row["label"],
                status=row["status"],
                created_at=row["created_at"],
                archived_at=row["archived_at"],
                share=window_value / pool_window if pool_window else 0.0,
                usage=usage,
            )
        )
    return coverage_started_at, pool_window * scale, items
