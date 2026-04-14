"""Unified session wake/recovery sweep.

Replaces the per-session ``defer_wake`` + ``should_call_model`` +
``recover_orphans`` triad with a single function that:

1. **Repairs ghosts** — tool calls that were dispatched but never
   completed (SIGKILL, crash before launch, etc.).
2. **Finds sessions needing inference** — unreacted user messages,
   completed tool batches, or ghost repairs.
3. **Defers procrastinate wakes** for those sessions.

Called from three sites:

- **Recurring sweep** — starts at worker boot (immediate), then periodic.
- **Tool result appended** — scoped to the completing session.
- **API endpoints** — user messages and tool confirmations continue to
  use the existing ``defer_wake`` hot path.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.harness.task_registry import TaskRegistry
from aios.harness.wake import defer_wake
from aios.logging import get_logger
from aios.services import sessions as sessions_service

log = get_logger("aios.harness.sweep")


# ─── ghost repair ────────────────────────────────────────────────────────────


async def find_and_repair_ghosts(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    *,
    session_id: str | None = None,
) -> list[tuple[str, str]]:
    """Find ghost tool calls and append synthetic error results.

    A ghost is a tool_call_id from an assistant message where:

    - No tool-role result event exists in the log.
    - No asyncio task is in-flight (TaskRegistry).
    - The harness would have dispatched the tool (i.e. it's not a
      custom tool or an unconfirmed ``always_ask`` tool still waiting
      for client action).

    Returns a list of ``(session_id, tool_call_id)`` pairs that were
    repaired.
    """
    in_flight = task_registry.all_in_flight_tool_call_ids()

    scope_clause = "AND e.session_id = $1" if session_id else ""
    scope_params: list[Any] = [session_id] if session_id else []

    async with pool.acquire() as conn:
        asst_rows = await conn.fetch(
            f"""
            SELECT e.session_id, e.data
              FROM events e
              JOIN sessions s ON s.id = e.session_id
             WHERE s.archived_at IS NULL
               AND e.kind = 'message'
               AND e.data->>'role' = 'assistant'
               AND jsonb_array_length(COALESCE(e.data->'tool_calls', '[]'::jsonb)) > 0
               {scope_clause}
            """,
            *scope_params,
        )

        if not asst_rows:
            return []

        session_ids = list({r["session_id"] for r in asst_rows})

        result_rows = await conn.fetch(
            """
            SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
              FROM events e
             WHERE e.session_id = ANY($1::text[])
               AND e.kind = 'message'
               AND e.data->>'role' = 'tool'
            """,
            session_ids,
        )
        results_by_session: dict[str, set[str]] = {}
        for r in result_rows:
            results_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

        lifecycle_rows = await conn.fetch(
            """
            SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
              FROM events e
             WHERE e.session_id = ANY($1::text[])
               AND e.kind = 'lifecycle'
               AND e.data->>'event' = 'tool_confirmed'
               AND e.data->>'result' = 'allow'
            """,
            session_ids,
        )
        confirmed_by_session: dict[str, set[str]] = {}
        for r in lifecycle_rows:
            confirmed_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

    # First pass: find candidate ghosts (no result, no in-flight task).
    # We don't yet know their dispatch status — that requires agent config.
    candidates: list[tuple[str, str, str]] = []  # (session_id, tool_call_id, tool_name)

    for row in asst_rows:
        sid = row["session_id"]
        data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
        existing_results = results_by_session.get(sid, set())
        session_in_flight = in_flight.get(sid, set())

        for tc in data.get("tool_calls") or []:
            tcid = tc.get("id")
            if not tcid or tcid in existing_results or tcid in session_in_flight:
                continue
            name = (tc.get("function") or {}).get("name", "")
            candidates.append((sid, tcid, name))

    if not candidates:
        return []

    # Second pass: load agent config only for sessions with candidates,
    # then filter to actually-dispatched tools.
    candidate_sids = list({sid for sid, _, _ in candidates})
    async with pool.acquire() as conn:
        # LEFT JOIN agent_versions to respect version pinning.
        agent_rows = await conn.fetch(
            """
            SELECT s.id AS session_id,
                   COALESCE(av.tools, a.tools) AS tools
              FROM sessions s
              JOIN agents a ON a.id = s.agent_id
              LEFT JOIN agent_versions av
                ON av.agent_id = s.agent_id AND av.version = s.agent_version
             WHERE s.id = ANY($1::text[])
            """,
            candidate_sids,
        )
    agent_tools_by_session: dict[str, Any] = {}
    for r in agent_rows:
        raw = r["tools"]
        agent_tools_by_session[r["session_id"]] = json.loads(raw) if isinstance(raw, str) else raw

    ghosts: list[tuple[str, str, str]] = []
    for sid, tcid, name in candidates:
        confirmed = confirmed_by_session.get(sid, set())
        agent_tools = agent_tools_by_session.get(sid, [])
        if _was_dispatched(name, tcid, confirmed, agent_tools):
            ghosts.append((sid, tcid, name))

    for sid, tcid, name in ghosts:
        content = json.dumps(
            {"error": "No result was received for this tool call."},
            ensure_ascii=False,
        )
        await sessions_service.append_event(
            pool,
            sid,
            "message",
            {
                "role": "tool",
                "tool_call_id": tcid,
                "name": name,
                "content": content,
                "is_error": True,
            },
        )
        log.info("sweep.ghost_repaired", session_id=sid, tool_call_id=tcid, tool_name=name)

    return [(sid, tcid) for sid, tcid, _ in ghosts]


def _was_dispatched(
    name: str,
    tool_call_id: str,
    confirmed_ids: set[str],
    agent_tools: list[dict[str, Any]],
) -> bool:
    """Determine whether a tool call was dispatched by the harness.

    A dispatched tool that has no result and no in-flight task is a
    ghost. A tool that was never dispatched (custom, or unconfirmed
    ``always_ask``) is legitimately waiting for the client.
    """
    from aios.tools.registry import registry

    if name.startswith("mcp__"):
        server_name = name.split("__", 2)[1]
        for spec in agent_tools:
            if spec.get("type") == "mcp_toolset" and spec.get("mcp_server_name") == server_name:
                dc = spec.get("default_config") or {}
                pp = dc.get("permission_policy") or {}
                perm = pp.get("type") or spec.get("permission")
                if perm == "always_allow":
                    return True
                return tool_call_id in confirmed_ids
        return tool_call_id in confirmed_ids

    if not registry.has(name):
        return False

    for spec in agent_tools:
        tool_name = spec.get("name") if spec.get("type") == "custom" else spec.get("type")
        if tool_name == name:
            if spec.get("permission") == "always_ask":
                return tool_call_id in confirmed_ids
            return True

    return True


# ─── sessions needing inference ──────────────────────────────────────────────


async def find_sessions_needing_inference(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    *,
    session_id: str | None = None,
) -> set[str]:
    """Return session IDs that need an inference step.

    A session needs inference when:

    (a) It has message events but no assistant message (first turn).
    (b) It has non-assistant message events with ``seq`` greater than
        the last assistant message's ``reacting_to`` — these are events
        the model hasn't reacted to yet.
    (c) It has a ``tool_confirmed allow`` lifecycle event for a
        ``tool_call_id`` that has no result and no in-flight task
        (needs dispatch via ``_dispatch_confirmed_tools``).

    Sessions from (a)/(b) are filtered: if the only unreacted events are
    tool results from a batch with in-flight tasks, the session is not
    yet ready. Case (c) sessions bypass this filter.
    """
    scope_clause = "AND s.id = $1" if session_id else ""
    scope_params: list[Any] = [session_id] if session_id else []

    async with pool.acquire() as conn:
        candidate_rows = await conn.fetch(
            f"""
            SELECT DISTINCT e.session_id
              FROM events e
              JOIN sessions s ON s.id = e.session_id
             WHERE s.archived_at IS NULL
               AND e.kind = 'message'
               AND e.data->>'role' != 'assistant'
               AND (
                   NOT EXISTS (
                       SELECT 1 FROM events a
                        WHERE a.session_id = e.session_id
                          AND a.kind = 'message'
                          AND a.data->>'role' = 'assistant'
                   )
                   OR
                   e.seq > COALESCE(
                       (SELECT MAX(COALESCE((a.data->>'reacting_to')::bigint, a.seq))
                          FROM events a
                         WHERE a.session_id = e.session_id
                           AND a.kind = 'message'
                           AND a.data->>'role' = 'assistant'),
                       0
                   )
               )
               {scope_clause}
            """,
            *scope_params,
        )

        candidates = {r["session_id"] for r in candidate_rows}

        # Case (c) bypasses the batch filter — confirmed tools need dispatch.
        confirmed_rows = await conn.fetch(
            f"""
            SELECT DISTINCT lc.session_id
              FROM events lc
              JOIN sessions s ON s.id = lc.session_id
             WHERE s.archived_at IS NULL
               AND lc.kind = 'lifecycle'
               AND lc.data->>'event' = 'tool_confirmed'
               AND lc.data->>'result' = 'allow'
               AND NOT EXISTS (
                   SELECT 1 FROM events tr
                    WHERE tr.session_id = lc.session_id
                      AND tr.kind = 'message'
                      AND tr.data->>'role' = 'tool'
                      AND tr.data->>'tool_call_id' = lc.data->>'tool_call_id'
               )
               {scope_clause}
            """,
            *scope_params,
        )
        confirmed_sessions = {r["session_id"] for r in confirmed_rows}

    to_filter = candidates - confirmed_sessions
    filtered = (
        await _filter_incomplete_batches(pool, task_registry, to_filter) if to_filter else set()
    )
    return filtered | confirmed_sessions


async def _filter_incomplete_batches(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    candidates: set[str],
) -> set[str]:
    """Remove sessions whose only unreacted events are tool results from
    in-progress batches (where sibling tools are still in-flight).

    Uses three batched queries across all candidates (no N+1).
    """
    session_list = list(candidates)

    async with pool.acquire() as conn:
        unreacted_rows = await conn.fetch(
            """
            SELECT e.session_id, e.data
              FROM events e
             WHERE e.session_id = ANY($1::text[])
               AND e.kind = 'message'
               AND e.data->>'role' != 'assistant'
               AND e.seq > COALESCE(
                   (SELECT MAX(COALESCE((a.data->>'reacting_to')::bigint, a.seq))
                      FROM events a
                     WHERE a.session_id = e.session_id
                       AND a.kind = 'message'
                       AND a.data->>'role' = 'assistant'),
                   0)
            """,
            session_list,
        )

        all_result_rows = await conn.fetch(
            """
            SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
              FROM events e
             WHERE e.session_id = ANY($1::text[])
               AND e.kind = 'message'
               AND e.data->>'role' = 'tool'
            """,
            session_list,
        )

        all_asst_rows = await conn.fetch(
            """
            SELECT e.session_id, e.data
              FROM events e
             WHERE e.session_id = ANY($1::text[])
               AND e.kind = 'message'
               AND e.data->>'role' = 'assistant'
               AND jsonb_array_length(COALESCE(e.data->'tool_calls', '[]'::jsonb)) > 0
            """,
            session_list,
        )

    unreacted_by_sid = _group_event_data(unreacted_rows)
    results_by_sid = _group_tool_call_ids(all_result_rows)
    asst_by_sid = _group_event_data(all_asst_rows)

    result: set[str] = set()
    for sid in candidates:
        in_flight = task_registry.in_flight_tool_call_ids(sid)
        unreacted = unreacted_by_sid.get(sid, [])

        if not unreacted:
            if not in_flight:
                result.add(sid)
            continue

        if any(evt.get("role") == "user" for evt in unreacted):
            result.add(sid)
            continue

        unreacted_tcids = {evt.get("tool_call_id") for evt in unreacted if evt.get("tool_call_id")}
        all_result_ids = results_by_sid.get(sid, set())

        for asst_data in asst_by_sid.get(sid, []):
            batch_ids = {tc["id"] for tc in (asst_data.get("tool_calls") or []) if tc.get("id")}
            if not (batch_ids & unreacted_tcids):
                continue
            if batch_ids <= all_result_ids:
                result.add(sid)
                break

    return result


def _group_event_data(rows: list[Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
        grouped.setdefault(r["session_id"], []).append(data)
    return grouped


def _group_tool_call_ids(rows: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for r in rows:
        grouped.setdefault(r["session_id"], set()).add(r["tool_call_id"])
    return grouped


# ─── main entry point ────────────────────────────────────────────────────────


async def wake_sessions_needing_inference(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    *,
    session_id: str | None = None,
) -> set[str]:
    """The main sweep function.

    1. Repairs ghosts (appends synthetic error results).
    2. Finds sessions needing inference.
    3. Defers procrastinate wakes for those sessions.

    Returns the set of session IDs that were woken.
    """
    await find_and_repair_ghosts(pool, task_registry, session_id=session_id)
    session_ids = await find_sessions_needing_inference(pool, task_registry, session_id=session_id)
    for sid in session_ids:
        await defer_wake(sid, cause="sweep")
    return session_ids
