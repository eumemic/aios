"""Unified session wake/recovery sweep.

A single function that:

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

import datetime as dt
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from aios.models.agents import HttpServerSpec, ToolSpec

from aios.config import get_settings
from aios.db.queries import (
    confirmed_unresolved_predicate,
    parse_jsonb,
    session_active_predicate,
    session_errored_predicate,
)
from aios.harness.task_registry import TaskRegistry
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.services.wake import defer_wake

log = get_logger("aios.harness.sweep")

# The three shared wake-decision predicate generators are imported (not
# redefined) from ``aios.db.queries`` and composed into the sweep's detector
# SQL below — re-exported here by identity so the structural sync guard
# (``tests/unit/test_wake_predicate_single_source.py``) can assert that the
# sweep consumes the SAME source objects as the read/dispatch path. Listing
# them in ``__all__`` marks them as explicit re-exports under ``strict`` mypy
# (``no_implicit_reexport``), which would otherwise flag attribute access on
# the module as ``attr-defined``.
__all__ = [
    "confirmed_unresolved_predicate",
    "session_active_predicate",
    "session_errored_predicate",
    "wake_sessions_needing_inference",
]


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Return value of :func:`wake_sessions_needing_inference`.

    Exposes both counts so the tail-site sweep span can stamp them on
    ``sweep_end`` without unrolling the composition. ``woken_sessions``
    is the number of procrastinate wakes deferred; ``repaired_ghosts``
    is the number of synthetic tool-error events appended during ghost
    repair.
    """

    repaired_ghosts: int
    woken_sessions: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A tool_call with no result and no in-flight task — a ghost-repair
    candidate before its dispatch status is known.

    ``created_at`` is the assistant turn's emit time; the abandoned-client-call
    branch (#752) age-bounds off it.
    """

    session_id: str
    tool_call_id: str
    tool_name: str
    created_at: dt.datetime
    # Raw ``function.arguments`` (str or dict, provider-dependent), carried so
    # the sweep can apply the SAME arg-aware route refinement the dispatch and
    # read paths do (#1076). Already read at candidate construction; previously
    # discarded. ``None`` for the rare malformed assistant turn with no
    # ``function`` blob — the classifier falls through to the base permission.
    arguments: Any | None = None


@dataclass(frozen=True, slots=True)
class _SweepAgentSurface:
    """The minimal agent surface the disposition classifier (#1076) consumes.

    The classifier reads only ``.tools`` (permission resolution) and
    ``.http_servers`` (arg-aware route refinement for ``http_request``), so the
    sweep builds this thin stand-in per candidate session rather than hydrating
    a full :class:`~aios.models.agents.Agent`. Structurally a duck-typed
    ``Agent`` for the two attributes the classifier touches.
    """

    tools: list[ToolSpec]
    http_servers: list[HttpServerSpec]


# ─── query constants ─────────────────────────────────────────────────────────
#
# Sweep SQL lives here as module constants so tests/e2e/test_sweep_perf.py
# can EXPLAIN the exact production query text. ``CANDIDATE_ROWS_SQL``
# and ``ERRORED_SESSIONS_SQL`` now use the four maintained scalar columns
# on ``sessions`` (migration 0066) — pure column arithmetic, no event-log
# scans. Ghost-repair queries still scan ``events`` because they need
# per-tool_call_id resolution the scalars don't carry.


# ``GHOST_ASST_SQL`` is the cross-session entry point of ghost repair: it
# returns every assistant-with-tool_calls event whose session might contain a
# ghost.  Bounded by ``s.open_tool_call_count > 0`` (the same maintained scalar
# ``CANDIDATE_ROWS_SQL`` uses, migration 0066): that count is ``> 0`` exactly
# iff the session has an assistant tool_call with no paired tool-result — i.e.
# exactly the sessions that can hold a ghost.  Without the bound, a fully
# resolved session's entire tool-call history is rescanned on every sweep pass
# (#840).  Sessions with open calls still return ALL their
# assistant-with-tool_calls events, so the per-tcid candidate loop downstream is
# behaviorally unchanged.
#
# Also bounded by the errored-session predicate (#897): an errored session is
# parked until a user message recovers it, so its open tool_calls are part of
# the terminal landing pad and must NOT be reaped.  ``find_and_repair_ghosts``
# already drops these via the Python ``_errored_session_ids`` post-filter;
# pushing the same predicate here stops the wasted event-log fetch on every 30s
# sweep for the small set of errored sessions that still carry open calls.  This
# ``NOT`` is composed from the SAME single source ``session_errored_predicate``
# that backs ``ERRORED_SESSIONS_SQL`` (and the read path) — both consume the
# maintained scalar columns (migration 0066), so the SQL pre-filter's errored
# set EQUALS the Python post-filter's; the post-filter is retained as defense in
# depth.
GHOST_ASST_SQL = f"""
    SELECT e.session_id, e.data, e.created_at
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE s.archived_at IS NULL
       AND s.open_tool_call_count > 0
       AND NOT {session_errored_predicate("s")}
       AND e.kind = 'message'
       AND e.role = 'assistant'
       AND jsonb_array_length(COALESCE(NULLIF(e.data->'tool_calls', 'null'::jsonb), '[]'::jsonb)) > 0
       {{scope_clause}}
"""

GHOST_LIFECYCLE_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'lifecycle'
       AND e.data->>'event' = 'tool_confirmed'
       AND e.data->>'result' = 'allow'
"""

# Dispatch-marker spans: pre-invoke ``tool_execute_start`` events keyed by
# tool_call_id.  Scope by both session set and tcid set so the seq-scan
# stays bounded without a new index — the candidate counts are typically
# single-digit per sweep.  Drives the two-branch recovery synthesis in
# :func:`find_and_repair_ghosts` (#685).  If profiling under load shows
# this as a hot spot, add a partial expression index following migration
# 0024's pattern:
#   CREATE INDEX events_tool_execute_start_idx ON events ((data->>'tool_call_id'))
#       WHERE kind = 'span' AND data->>'event' = 'tool_execute_start';
GHOST_SPAN_START_SQL = """
    SELECT DISTINCT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'span'
       AND e.data->>'event' = 'tool_execute_start'
       AND e.data->>'tool_call_id' = ANY($2::text[])
"""

# Candidate filter — the wake predicate composed from the SINGLE source
# ``queries.session_active_predicate`` (bound at the ``s`` alias here, at the
# ``sessions`` alias for the read-path status derivation). The read-path status
# predicate and this wake predicate therefore agree BY CONSTRUCTION — they
# cannot drift, so the worker can no longer wake a session with no progress to
# make (#155 symptom) or skip one that needs inference. ``last_stimulus_seq``
# (non-assistant messages — user + tool), NOT ``last_event_seq`` (which includes
# the session's own assistant replies): the latter classifies an idle turn
# (user → assistant reply) as a candidate and drives one extra model step (#749).
# Kept as full standalone SQL (composed from, not replaced by, the fragment) so
# the perf guard can EXPLAIN the exact production text.
CANDIDATE_ROWS_SQL = f"""
    SELECT s.id AS session_id
      FROM sessions s
     WHERE s.archived_at IS NULL
       AND {session_active_predicate("s")}
       {{scope_clause}}
"""

# Cross-session detection of confirmed-but-unresolved tools, for the wake
# decision (case (c)).  The dispatch-side counterpart that resolves these same
# confirmed-allow, result-less tool_calls into the actual tool_call dicts to
# launch is ``queries.list_confirmed_unresolved_tool_calls`` (per-session).
# Both compose the ``lc`` WHERE sub-predicate from the SINGLE source
# ``queries.confirmed_unresolved_predicate`` (``tool_confirmed``/``allow`` ∧
# no ``role='tool'`` result ∧ confirm event within
# ``confirmed_dispatch_max_age_seconds``), so detection and dispatch resolve the
# IDENTICAL condition by construction — they cannot drift.  The age bound is on
# ``lc.created_at`` (the CONFIRM event), NOT the assistant turn: a fresh confirm
# of an old proposal is a fresh intent to dispatch (#746).  Both are served by
# ``events_tool_confirmed_allow_idx`` (0065).  Kept as full standalone SQL
# (composed from, not replaced by, the fragment) so the perf guard can EXPLAIN
# the exact production text.  ``{age_param}`` survives the fragment to remain a
# ``str.format`` placeholder bound to the positional ``$N`` at call time.
CONFIRMED_ROWS_SQL = f"""
    SELECT DISTINCT lc.session_id
      FROM events lc
      JOIN sessions s ON s.id = lc.session_id
     WHERE s.archived_at IS NULL
       AND {confirmed_unresolved_predicate("lc", "{age_param}")}
       {{scope_clause}}
"""

# The reaction watermark — ``MAX(COALESCE(reacting_to, seq))`` over assistant
# messages — lives in exactly ONE writable place: the ``last_reacted_seq``
# UPDATE in ``append_event`` (db/queries/events.py), seeded once by migration
# 0066's backfill. This gate consumes that maintained scalar directly via a
# JOIN; it does NOT recompute the watermark. Re-deriving the formula here (the
# pre-#1080 ``session_max_reacting`` CTE) re-introduces the #155-class drift
# the deletion in #1080 foreclosed — keep this an equality JOIN, not a CTE.
# The ``no_reaction`` clause excludes fire-and-forget delivery confirmations (a
# connector ``signal_send``/``telegram_react``/… success, stamped
# ``data['no_reaction']=true`` by ``append_tool_result``) from the unreacted-
# stimulus set the batch filter inspects — matching the scalar gate, where such
# a result does not bump ``last_stimulus_seq`` (``append_event``'s
# ``is_stimulus``). ``IS DISTINCT FROM 'true'`` is NULL-safe: a missing key →
# NULL → TRUE → the row still counts (every historical/unmarked result wakes
# exactly as before); only the literal JSON ``true`` is excluded.
UNREACTED_ROWS_SQL = """
    SELECT e.session_id, e.data
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role <> 'assistant'
       AND e.data->>'no_reaction' IS DISTINCT FROM 'true'
       AND e.seq > s.last_reacted_seq
"""

ALL_RESULT_ROWS_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role = 'tool'
"""

ALL_ASST_ROWS_SQL = """
    SELECT e.session_id, e.data
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role = 'assistant'
       AND jsonb_array_length(COALESCE(NULLIF(e.data->'tool_calls', 'null'::jsonb), '[]'::jsonb)) > 0
"""

# Sessions currently in the terminal "errored" state, derived from the
# maintained scalar columns on ``sessions`` (migration 0066). The errored
# boolean is composed from the SINGLE source ``queries.session_errored_predicate``
# (bound at the ``s`` alias here, at the ``sessions`` alias for the read path),
# so the sweep's errored filter and the read-path ``errored`` derivation agree
# by construction. A session is errored when ``last_error_seq > 0 AND
# last_error_seq > last_user_seq``; a later user message bumps ``last_user_seq``,
# flipping the inequality — exactly the recovery semantics the pre-derivation
# status flip provided. Kept as full standalone SQL (composed from, not replaced
# by, the fragment) so the perf guard can EXPLAIN the exact production text.
ERRORED_SESSIONS_SQL = f"""
    SELECT s.id AS session_id
      FROM sessions s
     WHERE s.archived_at IS NULL
       AND {session_errored_predicate("s")}
       {{scope_clause}}
"""

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

    Plus a second, age-bounded category — **abandoned client-side tool
    calls** (#752). A *client-result-pending* call (the non-MCP,
    not-in-registry branch of :func:`_was_dispatched` — a tool the
    *client* runs and returns a result for) is normally left alone: it's
    legitimately waiting for the client. But if its assistant turn is
    older than ``client_tool_call_max_age_seconds`` (default 24h) with
    still no result and no in-flight task, the client has disconnected and
    will never return — so we synthesise an abandoned/timeout error
    result. Without this the call's ``open_tool_call_count`` contribution
    keeps the session a permanent wake candidate
    (``CANDIDATE_ROWS_SQL``) with no progress to make — the #155
    wake-no-progress loop, a regression from the open-tool-call-count
    predicate added in #750. *Confirmation-pending* ``always_ask`` calls
    (awaiting a ``tool_confirmed`` event) are EXCLUDED: those wait on the
    USER, not a client, and erroring them would kill a slow
    human-in-the-loop confirmation.

    Returns a list of ``(session_id, tool_call_id)`` pairs that were
    repaired (both categories count as repairs).
    """
    in_flight = task_registry.all_in_flight_tool_call_ids()

    scope_clause = "AND e.session_id = $1" if session_id else ""
    scope_params: list[Any] = [session_id] if session_id else []

    async with pool.acquire() as conn:
        asst_rows = await conn.fetch(
            GHOST_ASST_SQL.format(scope_clause=scope_clause),
            *scope_params,
        )

        if not asst_rows:
            return []

        session_ids = list({r["session_id"] for r in asst_rows})

        # Skip errored sessions: their dispatched-but-unresolved tool calls are
        # part of the terminal landing pad and stay unreaped until a user
        # message recovers the session (mirrors the pre-derivation status skip).
        # ``GHOST_ASST_SQL`` already pushes the identical errored predicate, so
        # ``asst_rows`` carries no errored session here; this post-filter is
        # retained as defense in depth (#897).
        errored = await _errored_session_ids(conn, session_id=session_id)
        if errored:
            asst_rows = [r for r in asst_rows if r["session_id"] not in errored]
            if not asst_rows:
                return []
            session_ids = [s for s in session_ids if s not in errored]

        result_rows = await conn.fetch(ALL_RESULT_ROWS_SQL, session_ids)
        results_by_session: dict[str, set[str]] = {}
        for r in result_rows:
            results_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

        lifecycle_rows = await conn.fetch(GHOST_LIFECYCLE_SQL, session_ids)
        confirmed_by_session: dict[str, set[str]] = {}
        for r in lifecycle_rows:
            confirmed_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

    # First pass: find candidate ghosts (no result, no in-flight task).
    # We don't yet know their dispatch status — that requires agent config.
    # ``created_at`` is the assistant turn's emit time, carried so the
    # abandoned-client-call branch can age-bound off it (#752).
    candidates: list[_Candidate] = []

    for row in asst_rows:
        sid = row["session_id"]
        data = parse_jsonb(row["data"])
        created_at = row["created_at"]
        existing_results = results_by_session.get(sid, set())
        session_in_flight = in_flight.get(sid, set())

        for tc in data.get("tool_calls") or []:
            tcid = tc.get("id")
            if not tcid or tcid in existing_results or tcid in session_in_flight:
                continue
            function = tc.get("function") or {}
            name = function.get("name", "")
            candidates.append(
                _Candidate(sid, tcid, name, created_at, arguments=function.get("arguments"))
            )

    if not candidates:
        return []

    # Second pass: load agent config only for sessions with candidates,
    # then classify each candidate into one of two repairable buckets
    # (dispatched ghost, or abandoned client-result-pending call past the age
    # bound). All other candidates are left alone (legitimately waiting).
    candidate_sids = list({c.session_id for c in candidates})
    async with pool.acquire() as conn:
        # LEFT JOIN agent_versions to respect version pinning. ``http_servers``
        # is fetched alongside ``tools`` so the classifier can apply the
        # arg-aware route refinement for ``http_request`` (#1076) — the same
        # refinement the dispatch and read paths apply.
        agent_rows = await conn.fetch(
            """
            SELECT s.id AS session_id,
                   COALESCE(av.tools, a.tools) AS tools,
                   COALESCE(av.http_servers, a.http_servers) AS http_servers
              FROM sessions s
              JOIN agents a ON a.id = s.agent_id
              LEFT JOIN agent_versions av
                ON av.agent_id = s.agent_id AND av.version = s.agent_version
             WHERE s.id = ANY($1::text[])
            """,
            candidate_sids,
        )
    from aios.models.agents import HttpServerSpec, ToolSpec

    agent_surface_by_session: dict[str, _SweepAgentSurface] = {}
    for r in agent_rows:
        tools_list = parse_jsonb(r["tools"])
        http_list = parse_jsonb(r["http_servers"])
        agent_surface_by_session[r["session_id"]] = _SweepAgentSurface(
            tools=[ToolSpec.model_validate(t) for t in (tools_list or [])],
            http_servers=[HttpServerSpec.model_validate(h) for h in (http_list or [])],
        )

    # Abandoned-client-call bound (#752): a client-result-pending call whose
    # assistant turn is older than this is treated as abandoned. The cutoff is
    # computed once per sweep against ``created_at`` (the assistant turn's emit
    # time), consistent with the dispatched-ghost age semantics.
    client_max_age = get_settings().client_tool_call_max_age_seconds
    abandoned_cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=client_max_age)

    empty_surface = _SweepAgentSurface(tools=[], http_servers=[])
    ghosts: list[_Candidate] = []
    abandoned: list[_Candidate] = []
    for c in candidates:
        confirmed = confirmed_by_session.get(c.session_id, set())
        surface = agent_surface_by_session.get(c.session_id, empty_surface)
        if _was_dispatched(c, confirmed, surface):
            ghosts.append(c)
        elif (
            _is_client_result_pending(c.tool_name, surface.tools)
            and c.created_at < abandoned_cutoff
        ):
            # Not dispatched AND client-result-pending AND older than the bound:
            # the client disconnected and will never return a result. Without
            # resolving it, the call's open_tool_call_count contribution keeps
            # the session a permanent wake candidate with no progress (#155 loop,
            # regression from #750). Confirmation-pending always_ask calls are
            # NOT client-result-pending, so they fall through here and are left
            # to wait on the user.
            abandoned.append(c)

    # ``tool_execute_start`` span presence per (session, tcid) — drives the
    # two-branch recovery message below (#685).  Tcids missing from this set
    # never reached the lifecycle body, so the tool definitely did not run;
    # tcids present may have executed and committed side effects.  Scope to
    # the post-``_was_dispatched`` ghost set (not the wider candidate set) so
    # the seq-scan touches only the tcids the per-ghost loop will actually
    # consult.  Abandoned client calls are never dispatched by the harness, so
    # they have no ``tool_execute_start`` span and don't need this lookup.
    started: set[tuple[str, str]] = set()
    if ghosts:
        ghost_sids = list({c.session_id for c in ghosts})
        ghost_tcids = list({c.tool_call_id for c in ghosts})
        async with pool.acquire() as conn:
            span_rows = await conn.fetch(GHOST_SPAN_START_SQL, ghost_sids, ghost_tcids)
        started = {(r["session_id"], r["tool_call_id"]) for r in span_rows}

    # Build the unified repair worklist: each item carries its candidate, an
    # operational ``branch`` tag (for the log), and the synthetic error text.
    repair_items: list[tuple[_Candidate, str, str]] = []
    for c in ghosts:
        # Don't lie: distinguish "never dispatched" (safe to retry) from
        # "may have executed" (verify before retrying).  The previous
        # single fabricated "No result was received" message double-fired
        # non-idempotent tools (bash mutations, http_request POST,
        # connector send) on the model's retry — see #685.
        #
        # The "may have completed" branch is conservatively over-pessimistic:
        # it also fires for crashes/cancels in the window between the span
        # commit and the actual side-effectful invoke (MCP auth resolve,
        # parameter validation, ``asyncio.CancelledError`` arriving inside
        # the span ``await``).  Those produce false "verify the outcome"
        # advice but never the dangerous false "safe to retry" that this
        # design eliminates.  Tighter classification would require a second
        # marker written immediately before each tool's side-effectful call
        # — deferred until the over-pessimism is shown to matter.
        if (c.session_id, c.tool_call_id) in started:
            repair_items.append(
                (
                    c,
                    "may_have_completed",
                    "Tool dispatch was interrupted after execution began. "
                    "The tool may have completed and side effects may have "
                    "committed. Verify the outcome before retrying.",
                )
            )
        else:
            repair_items.append(
                (
                    c,
                    "did_not_run",
                    "Tool dispatch was lost before execution began; "
                    "the tool did not run. You may retry.",
                )
            )
    for c in abandoned:
        # Distinct wording from the dispatched-ghost branches: this tool is run
        # by the CLIENT, which never returned a result within the bound.
        repair_items.append(
            (
                c,
                "client_abandoned",
                f"Tool call abandoned: the client returned no result within "
                f"{client_max_age}s. The client is no longer connected; do not "
                f"wait for this result.",
            )
        )

    # Per-ghost isolation; see ``wake_sessions_needing_inference`` below
    # for the rationale.
    repaired: list[tuple[str, str]] = []
    for c, branch, error_text in repair_items:
        sid = c.session_id
        tcid = c.tool_call_id
        name = c.tool_name
        content = json.dumps({"error": error_text}, ensure_ascii=False)
        # Load each ghost's session account_id individually so the
        # cross-session sweeper (session_id=None) doesn't stamp empty
        # account_id onto repair events for real tenants.
        # Route through ``append_tool_result`` (services/sessions.py)
        # rather than a bare ``append_event``: its session-row lock +
        # ``find_tool_result_event`` dedup serialise concurrent
        # repairs.  Bare-append had a TOCTOU window between the
        # result-rows read above and the write here that admitted two
        # duplicate synthetic results for the same ``tool_call_id``
        # under concurrent sweep invocation, violating invariant #4
        # (tool-always-appends-EXACTLY-one result).
        try:
            sid_account_id = await sessions_service.load_session_account_id(pool, sid)
            async with pool.acquire() as conn:
                await sessions_service.append_tool_result(
                    conn,
                    account_id=sid_account_id,
                    session_id=sid,
                    tool_call_id=tcid,
                    content=content,
                    is_error=True,
                )
        except Exception:
            log.exception(
                "sweep.ghost_repair_failed",
                session_id=sid,
                tool_call_id=tcid,
                tool_name=name,
            )
            continue
        repaired.append((sid, tcid))
        # ``branch`` is the operational signal of #685: ops can grep this
        # log for ``branch=may_have_completed`` after a crash to triage
        # which recoveries carry side-effect risk vs which are safe-retry.
        log.info(
            "sweep.ghost_repaired",
            session_id=sid,
            tool_call_id=tcid,
            tool_name=name,
            branch=branch,
        )

    return repaired


def _was_dispatched(
    candidate: _Candidate,
    confirmed_ids: set[str],
    surface: _SweepAgentSurface,
) -> bool:
    """Determine whether a tool call was dispatched by the harness.

    A dispatched tool that has no result and no in-flight task is a
    ghost. A tool that was never dispatched (custom, or unconfirmed
    ``always_ask``) is legitimately waiting for the client or the user.

    Thin projection of the single-source disposition classifier (#1076): a call
    counts as dispatched UNLESS it is a client-executed ``custom`` tool or an
    ``always_ask`` confirmation that has NOT yet been satisfied. The
    ``confirmation_resolved`` bit is this call's id ∈ ``confirmed_ids``, so a
    confirmation-pending call projects to ``NEEDS_CONFIRM`` → not dispatched.

    This is where the historical drift lived: the previous body applied only
    ``resolve_permission`` (the tool's BASE permission) and **missed the
    arg-aware route refinement** that the dispatch and read paths both apply —
    so a route-gated ``always_ask`` ``http_request`` parked awaiting user
    confirmation was wrongly reported dispatched, and the ghost-repair branch
    fabricated an error result that killed the parked confirmation (the exact
    outcome ``_is_client_result_pending``'s docstring forbids). Routing through
    the single classifier closes that gap by construction: the refinement now
    exists in exactly one place. No ``mcp_server_map`` here — the sweep doesn't
    distinguish ``unknown_mcp`` (an unregistered MCP server resolves through the
    normal MCP ladder, preserving prior behavior).
    """
    from aios.harness.tool_disposition import ToolDisposition, classify_tool_call

    disposition = classify_tool_call(
        candidate.tool_name,
        candidate.arguments,
        surface,  # type: ignore[arg-type]  # duck-typed: classifier reads only .tools/.http_servers
        confirmation_resolved=candidate.tool_call_id in confirmed_ids,
    )
    return disposition not in (ToolDisposition.NEEDS_CONFIRM, ToolDisposition.CUSTOM)


def _is_client_result_pending(name: str, agent_tools: list[ToolSpec]) -> bool:
    """True if ``name`` is a CLIENT-result-pending tool.

    A client-result-pending tool is one the harness never dispatches because
    the *client* executes it and returns the result — the non-MCP,
    not-in-registry branch of :func:`_was_dispatched` (a custom tool, or a tool
    the model emitted under a bare name the harness can't resolve to a registry
    entry or an MCP toolset). When the client disconnects, such a call sits
    unresolved forever; the abandoned-client-call repair (#752) errors it past
    an age bound.

    Deliberately NARROW — it must EXCLUDE confirmation-pending calls
    (``always_ask`` registered tools, or non-``always_allow`` MCP tools, awaiting
    a ``tool_confirmed`` event). Those wait on the USER, not a client, and
    erroring them would kill a slow human-in-the-loop confirmation. Both
    excluded classes are reached only when ``is_mcp_tool_name`` is true or
    ``registry.has`` is true, so testing the negation of both is exactly the
    client-result-pending set.
    """
    from aios.models.agents import is_mcp_tool_name
    from aios.tools.registry import registry

    return not is_mcp_tool_name(name) and not registry.has(name)


# ─── sessions needing inference ──────────────────────────────────────────────


async def _errored_session_ids(
    conn: asyncpg.Connection[Any], *, session_id: str | None = None
) -> set[str]:
    """Session IDs currently in the derived ``errored`` state.

    See ``ERRORED_SESSIONS_SQL``. The sweep excludes these from both
    inference and ghost repair: an errored session is parked until a user
    message recovers it (mirrors the pre-derivation ``status = 'errored'``
    skip + the ``append_event`` recovery flip).
    """
    scope_clause = "AND s.id = $1" if session_id else ""
    params: list[Any] = [session_id] if session_id else []
    rows = await conn.fetch(ERRORED_SESSIONS_SQL.format(scope_clause=scope_clause), *params)
    return {r["session_id"] for r in rows}


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
            CANDIDATE_ROWS_SQL.format(scope_clause=scope_clause),
            *scope_params,
        )

        candidates = {r["session_id"] for r in candidate_rows}

        # Case (c) bypasses the batch filter — confirmed tools need dispatch.
        # The confirm-event age bound (#746) MUST match the dispatch resolver's
        # (``queries.list_confirmed_unresolved_tool_calls``) so detection and
        # dispatch resolve the identical condition (no wake-with-no-progress,
        # the #155 symptom).  ``$N`` is positional after ``scope_params``; bind
        # the setting so a weeks-stale confirmation is not surfaced for wake.
        confirmed_max_age_seconds = get_settings().confirmed_dispatch_max_age_seconds
        age_param = f"${len(scope_params) + 1}"
        confirmed_rows = await conn.fetch(
            CONFIRMED_ROWS_SQL.format(scope_clause=scope_clause, age_param=age_param),
            *scope_params,
            confirmed_max_age_seconds,
        )
        confirmed_sessions = {r["session_id"] for r in confirmed_rows}

        # Errored sessions are parked until a user message recovers them.
        # Derived from the event log rather than a denormalized status column
        # (subtracted in-process to keep the candidate/confirmed queries free
        # of an anti-join that the perf guard would flag as a SubPlan).
        errored = await _errored_session_ids(conn, session_id=session_id)

    candidates -= errored
    confirmed_sessions -= errored
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
        unreacted_rows = await conn.fetch(UNREACTED_ROWS_SQL, session_list)
        all_result_rows = await conn.fetch(ALL_RESULT_ROWS_SQL, session_list)
        all_asst_rows = await conn.fetch(ALL_ASST_ROWS_SQL, session_list)

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
        data = parse_jsonb(r["data"])
        grouped.setdefault(r["session_id"], []).append(data)
    return grouped


def _group_tool_call_ids(rows: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for r in rows:
        grouped.setdefault(r["session_id"], set()).add(r["tool_call_id"])
    return grouped


# ─── procrastinate stalled-job recovery ──────────────────────────────────────


async def reap_stalled_jobs(job_manager: Any) -> int:
    """Mark predecessor-owned in-flight procrastinate jobs as failed.

    This is a boot-only recovery step. It runs after the worker
    singleton advisory lock has handed off, proving the predecessor is
    gone, and before this process starts consuming jobs. At that point
    every ``doing`` job in procrastinate is orphaned by construction;
    there is no live in-process job for a heartbeat-age filter to
    protect.

    :meth:`procrastinate.manager.JobManager.get_stalled_jobs` is the
    blessed query for this state. Calling it with a zero-second
    heartbeat threshold returns every ``doing`` job, including rows
    with ``worker_id IS NULL`` and rows whose worker heartbeat is merely
    in the past. Failing those jobs releases their procrastinate locks
    so the startup wake sweep can re-enqueue the affected sessions in
    the same pass.

    Takes a ``job_manager`` rather than an ``App`` so tests can build
    a fresh manager pointed at the testcontainer DB without depending
    on the module-level ``procrastinate_app`` singleton (which fixes
    its connector at import time).

    Returns the number of jobs reaped.  Non-zero is a real signal
    that a worker died.
    """
    from procrastinate.jobs import Status

    stalled = list(await job_manager.get_stalled_jobs(seconds_since_heartbeat=0))
    for job in stalled:
        if job.id is None:
            continue
        await job_manager.finish_job_by_id_async(
            job_id=job.id,
            status=Status.FAILED,
            delete_job=False,
        )
    if stalled:
        log.warning(
            "sweep.reaped_stalled_jobs",
            count=len(stalled),
            ids=[j.id for j in stalled],
        )
    return len(stalled)


# ─── main entry point ────────────────────────────────────────────────────────


async def wake_sessions_needing_inference(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    *,
    session_id: str | None = None,
) -> SweepResult:
    """The main sweep function.

    1. Repairs ghosts (appends synthetic error results).
    2. Finds sessions needing inference.
    3. Defers procrastinate wakes for those sessions.

    Returns a :class:`SweepResult` carrying the repaired-ghost count and
    the number of procrastinate wakes deferred, so the tail-site
    ``sweep_end`` span can stamp both without unrolling the composition.
    """
    repaired = await find_and_repair_ghosts(pool, task_registry, session_id=session_id)
    woken = await find_sessions_needing_inference(pool, task_registry, session_id=session_id)
    # Per-session try/except: a transient failure on one session must
    # not strand the rest of the cross-session batch.  account_id is
    # loaded individually because the cross-session sweeper has none
    # in scope, and "" would leak an empty account_id onto the
    # ``wake_deferred`` event for a real tenant.
    woken_count = 0
    for sid in woken:
        try:
            sid_account_id = await sessions_service.load_session_account_id(pool, sid)
            await defer_wake(pool, sid, cause="sweep", account_id=sid_account_id)
        except Exception:
            log.exception("sweep.defer_wake_failed", session_id=sid)
            continue
        woken_count += 1
    return SweepResult(repaired_ghosts=len(repaired), woken_sessions=woken_count)
