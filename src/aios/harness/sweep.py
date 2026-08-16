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
    find_parked_servicer,
    find_unharvested_model_dispatch_parks,
    list_session_ids_with_unharvested_cancel_marker,
    session_active_predicate,
    session_errored_predicate,
)
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.jobs.app import defer_wake
from aios.logging import get_logger
from aios.services import sessions as sessions_service

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
    "session_has_pending_work",
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


# The zero-tool, zero-server surface used as the fallback for a candidate whose
# agent config could not be loaded (a race where the session/agent row is gone).
# A call classified against it is never dispatched — consistent with leaving an
# unresolvable call alone rather than fabricating work.
_EMPTY_SURFACE = _SweepAgentSurface(tools=[], http_servers=[])


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
# assistant-with-tool_calls events AT OR AFTER the floor, so the per-tcid
# candidate loop downstream is behaviorally unchanged for every batch that
# could still be open.
#
# Also bounded by the errored-session predicate (#897): an errored session is
# parked until a user message recovers it, so its open tool_calls are part of
# the terminal landing pad and must NOT be reaped.  This is the SOLE errored-skip
# for ghost repair — composed from the same single source
# ``session_errored_predicate`` that backs ``ERRORED_SESSIONS_SQL`` and the read
# path, all consuming the maintained scalar columns (migration 0066).
#
# ``s.open_tool_call_floor_seq`` (migration 0136, #1746) is a PROVEN lower
# bound on the oldest still-open tool_call's ``seq`` — never a heuristic. It is
# advanced ``GREATEST``-only, and ONLY by the ghost sweep itself (see
# ``_advance_open_tool_call_floor`` below), from the reconciliation this very
# function already computes. It is NEVER stamped by the write path off the
# ``open_tool_call_count == 0`` transition — that edge is NOT trustworthy (a
# dedup-skip decrement can drive the count to 0 with a genuinely-open sibling
# still outstanding; see ``decrement_open_tool_call_count``'s docstring). A
# fresh session (or one whose floor has never been advanced) has floor 0 —
# unbounded, today's exact behavior. Projects ``e.seq`` (to compute the new
# floor) and ``e.data->'tool_calls'`` (NOT full ``e.data`` — the candidate loop
# only reads ``tool_calls`` + ``created_at`` + ``seq``) rather than the whole
# JSONB payload, bounding bytes-per-row for the rare very-old legitimately-
# waiting call whose batch survives the floor.
GHOST_ASST_SQL = f"""
    SELECT e.session_id, e.seq, e.data->'tool_calls' AS tool_calls, e.created_at
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE s.archived_at IS NULL
       AND s.open_tool_call_count > 0
       AND NOT {session_errored_predicate("s")}
       AND e.kind = 'message'
       AND e.role = 'assistant'
       AND jsonb_array_length(COALESCE(NULLIF(e.data->'tool_calls', 'null'::jsonb), '[]'::jsonb)) > 0
       AND e.seq >= s.open_tool_call_floor_seq
       {{scope_clause}}
"""

# Deploy-ordering fail-closed fallback (#1746, Rollout section): ``aios-worker``
# boots straight into new code with no migration step of its own, so a naive
# single-image promote can serve this code before the post-deploy ``aios
# migrate`` has added ``open_tool_call_floor_seq``. Rather than let
# ``UndefinedColumnError`` abort the whole cross-session sweep (a fleet-wide
# wake outage), ``find_and_repair_ghosts`` catches it and re-issues these three
# floor-free twins — byte-identical to the pre-#1746 unbounded queries — so
# ghost repair degrades to "scan everything" (always correct, merely slower)
# instead of going dark.
_GHOST_ASST_SQL_UNBOUNDED_FALLBACK = f"""
    SELECT e.session_id, e.seq, e.data->'tool_calls' AS tool_calls, e.created_at
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

_GHOST_RESULT_ROWS_SQL_UNBOUNDED_FALLBACK = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role = 'tool'
"""

_GHOST_LIFECYCLE_SQL_UNBOUNDED_FALLBACK = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'lifecycle'
       AND e.data->>'event' = 'tool_confirmed'
       AND e.data->>'result' = 'allow'
"""

# ``GHOST_RESULT_ROWS_SQL`` / ``GHOST_LIFECYCLE_SQL`` — the tool-result and
# tool_confirmed-lifecycle counterparts of ``GHOST_ASST_SQL``, bounded by the
# SAME floor via a join to ``sessions``. Safe per invariant 2 (no false
# ghosts): a result/confirm event for an in-scope candidate is always appended
# AFTER its owning assistant batch, so its seq is strictly greater than the
# batch's — which is itself ``>= floor`` for anything ``GHOST_ASST_SQL``
# fetched. A row below the floor can only pair with an assistant batch that is
# ALSO below the floor, i.e. already fully resolved and irrelevant to this
# scan (dropped along with its batch, not "falsely resolved" — it was never a
# candidate in the first place).
GHOST_RESULT_ROWS_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role = 'tool'
       AND e.seq >= s.open_tool_call_floor_seq
"""

GHOST_LIFECYCLE_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'lifecycle'
       AND e.data->>'event' = 'tool_confirmed'
       AND e.data->>'result' = 'allow'
       AND e.seq >= s.open_tool_call_floor_seq
"""

# Per-session agent surface (tools + http_servers) for the disposition
# classifier. LEFT JOIN agent_versions to respect version pinning;
# ``http_servers`` is fetched alongside ``tools`` so the classifier can apply
# the arg-aware route refinement for ``http_request`` (#1076) — the same
# refinement the dispatch and read paths apply. Shared by ghost repair
# (:func:`find_and_repair_ghosts`) and the inference batch filter
# (:func:`_filter_incomplete_batches`) via :func:`_load_surfaces`.
AGENT_SURFACE_SQL = """
    SELECT s.id AS session_id,
           COALESCE(av.tools, a.tools) AS tools,
           COALESCE(av.http_servers, a.http_servers) AS http_servers
      FROM sessions s
      JOIN agents a ON a.id = s.agent_id
      LEFT JOIN agent_versions av
        ON av.agent_id = s.agent_id AND av.version = s.agent_version
     WHERE s.id = ANY($1::text[])
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

# Floored variant for the #253 preemption trigger — the SAME single-source
# predicate re-parameterized against the in-flight step's context watermark
# (``$2``): during a step, the committed ``last_reacted_seq`` is the *previous*
# step's watermark, so the unfloored form would re-admit the very stimuli the
# in-flight step is already reacting to. Always single-session (``$1``) — the
# preempt check runs from inside a step, never cross-session.
CANDIDATE_ROWS_FLOORED_SQL = f"""
    SELECT s.id AS session_id
      FROM sessions s
     WHERE s.archived_at IS NULL
       AND {session_active_predicate("s", stimulus_floor_param="$2")}
       AND s.id = $1
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
# of an old proposal is a fresh intent to dispatch (#746).  This is THIS query
# (the cross-session detector, which has no ``session_id`` equality to seek on)
# that is served by ``events_tool_confirmed_allow_recent_idx`` (0134) — a single
# ``created_at``-keyed partial index the now-sargable age clause prunes at,
# rather than heap-fetching every confirmed-allow row ever (#1740).  The
# per-session dispatch resolver (``queries.list_confirmed_unresolved_tool_calls``)
# still seeks ``events_tool_confirmed_allow_idx`` (0065) via its ``session_id``
# equality — both indexes are additive, each serving its own reader.  Kept as
# full standalone SQL (composed from, not replaced by, the fragment) so the
# perf guard can EXPLAIN the exact production text.  ``{age_param}`` survives
# the fragment to remain a ``str.format`` placeholder bound to the positional
# ``$N`` at call time.
CONFIRMED_ROWS_SQL = f"""
    SELECT DISTINCT lc.session_id
      FROM events lc
      JOIN sessions s ON s.id = lc.session_id
     WHERE s.archived_at IS NULL
       AND {confirmed_unresolved_predicate("lc", "{age_param}")}
       {{scope_clause}}
"""

# Floored variant for the #253 preemption trigger. ``lc.seq > $3`` bounds the
# confirmed arm to confirms the in-flight step could NOT have consumed: a step
# that reaches the model phase already resolved every confirm it saw at its
# confirmed-dispatch check (``_dispatch_confirmed_tools`` early-returns when it
# dispatches), so only a confirm sequenced past the step's context watermark
# represents new dispatchable work. Projects ``tool_call_id`` (not DISTINCT
# session_id) so the caller can additionally subtract already-dispatched calls
# — in-flight in the registry or bearing a ``tool_execute_start`` span — which
# the registry-blind predicate cannot see; without that subtraction a confirmed
# tool dispatched by a PRIOR step but still running would re-admit on every
# evaluation and thrash the preempt loop.
CONFIRMED_ROWS_FLOORED_SQL = f"""
    SELECT lc.session_id, lc.data->>'tool_call_id' AS tool_call_id
      FROM events lc
      JOIN sessions s ON s.id = lc.session_id
     WHERE s.archived_at IS NULL
       AND {confirmed_unresolved_predicate("lc", "$2")}
       AND lc.seq > $3
       AND lc.session_id = $1
"""

# The reaction watermark — ``MAX(COALESCE(reacting_to, seq))`` over assistant
# messages — lives in exactly ONE writable place: the ``last_reacted_seq``
# UPDATE in ``append_event`` (db/queries/events.py), seeded once by migration
# 0066's backfill. This gate consumes that maintained scalar directly via a
# JOIN; it does NOT recompute the watermark. Re-deriving the formula here (the
# pre-#1080 ``session_max_reacting`` CTE) re-introduces the #155-class drift
# the deletion in #1080 foreclosed — keep this an equality JOIN, not a CTE.
# Every unreacted tool result counts (matching the scalar gate's ``is_stimulus``
# in ``append_event``): a session with any tool result past its reaction
# watermark is a wake candidate.
_UNREACTED_ROWS_TEMPLATE = """
    SELECT e.session_id, e.role, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role <> 'assistant'
       AND e.seq > {watermark_expr}
"""

UNREACTED_ROWS_SQL = _UNREACTED_ROWS_TEMPLATE.format(watermark_expr="s.last_reacted_seq")

# Floored variant for the #253 preemption trigger — same rows, with the
# watermark raised to the in-flight step's context watermark (``$2``).
# ``GREATEST`` rather than a bare ``$2`` is belt-free self-documentation: the
# step's watermark is ≥ ``last_reacted_seq`` by construction (its context
# includes everything the previous assistant reacted to), so the two forms are
# equivalent; GREATEST states the invariant in the query text.
UNREACTED_ROWS_FLOORED_SQL = _UNREACTED_ROWS_TEMPLATE.format(
    watermark_expr="GREATEST(s.last_reacted_seq, $2)"
)

# ─── batch filter: bounded, payload-stripped fetches (#1729) ─────────────────
#
# ``_filter_incomplete_batches`` decides, per candidate session, whether the
# session's ONLY unreacted events are tool results from a batch whose sibling
# tools are still in-flight (→ not yet ready). Its actual need is tiny: the
# unreacted tool_call_ids (already fetched, seq-bounded by
# ``UNREACTED_ROWS_SQL``) → their OWNING assistant batches → whether each such
# batch's ids are fully covered by tool results.
#
# The pre-#1729 queries fetched the session's ENTIRE lifetime history — every
# ``role='tool'`` message ever (``ALL_RESULT_ROWS_SQL``) and every assistant
# message with tool_calls ever, WITH FULL ``data`` PAYLOAD (``ALL_ASST_ROWS_SQL``:
# 126 MB of JSONB observed on the largest session). All of it was pulled over
# the wire and JSON-decoded row-by-row on the worker event loop on EVERY full
# sweep (~2/min), a 16.6s-median pre-model tax that grows linearly and forever
# with session size. Neither query was seq-bounded or LIMITed.
#
# The two replacements below bound the scan to exactly what the filter inspects
# and strip the payload to the only field it reads (``tool_calls[].id``):
#
# * ``REFERENCED_ASST_BATCH_SQL`` — assistant batches OWNING an unreacted tcid.
#   The ``@>`` containment (``data->'tool_calls' @> $2``, one probe per unreacted
#   tcid, built as ``[{"id": tcid}]``) restricts to the handful of assistant
#   rows the unreacted results belong to — not the whole history. It projects
#   ``jsonb_path_query_array(data->'tool_calls', '$[*].id')`` (the id array)
#   rather than ``data``, so the 126 MB decode is gone even before bounding.
#
# * ``BATCH_RESULT_ROWS_SQL`` — tool results whose ``tool_call_id`` is one of the
#   ids belonging to those referenced batches (``= ANY($2)``), not every
#   ``role='tool'`` row the session has ever produced.
#
# Rows drop from ~65k (34k results + 30k assistants) to ~dozens, and the decoded
# bytes from 126 MB to a few id strings.
REFERENCED_ASST_BATCH_SQL = """
    SELECT e.session_id,
           jsonb_path_query_array(e.data->'tool_calls', '$[*].id') AS tool_call_ids
      FROM events e
     WHERE e.session_id = $1
       AND e.kind = 'message'
       AND e.role = 'assistant'
       AND e.data->'tool_calls' @> ANY($2::jsonb[])
"""

BATCH_RESULT_ROWS_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = $1
       AND e.kind = 'message'
       AND e.role = 'tool'
       AND e.data->>'tool_call_id' = ANY($2::text[])
"""

# ``ALL_RESULT_ROWS_SQL`` — every ``role='tool'`` result across a session set.
# Retained for GHOST REPAIR (``find_and_repair_ghosts``), where the session set
# is already bounded upstream by ``GHOST_ASST_SQL``'s ``open_tool_call_count > 0``
# gate (migration 0066) — a fully-resolved session contributes zero rows. The
# batch filter (#1729) no longer uses this for the has-unreacted path: it
# fetches results per session and bounded to specific batch ids via
# ``BATCH_RESULT_ROWS_SQL``. It IS reused by the dispatch-narrowing branch
# (#1710, ``OPEN_CANDIDATES_ASST_SQL`` below) for the same reason ghost repair
# reuses it: that branch's session set is already bounded to the handful of
# sids with no unreacted events and no in-flight task.
ALL_RESULT_ROWS_SQL = """
    SELECT e.session_id, e.data->>'tool_call_id' AS tool_call_id
      FROM events e
     WHERE e.session_id = ANY($1::text[])
       AND e.kind = 'message'
       AND e.role = 'tool'
"""

# ``OPEN_CANDIDATES_ASST_SQL`` — assistant-with-tool_calls events for the
# dispatch-narrowing branch of ``_filter_incomplete_batches`` (#1710). Scoped
# to the caller-supplied session set (already narrowed to sids with an empty
# unreacted set and no in-flight task — typically a handful per sweep) and
# further bounded by ``open_tool_call_count > 0``, mirroring ``GHOST_ASST_SQL``:
# a fully-resolved session contributes zero rows.
OPEN_CANDIDATES_ASST_SQL = """
    SELECT e.session_id, e.data
      FROM events e
      JOIN sessions s ON s.id = e.session_id
     WHERE e.session_id = ANY($1::text[])
       AND s.open_tool_call_count > 0
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

# ─── fast-path admission gate (#1659) ────────────────────────────────────────
#
# ``FAST_PATH_PENDING_WORK_SQL`` is the per-turn admission guard's cheap
# necessary-condition gate. It is a SINGLE PK-scoped lookup on ``sessions``
# (``id = $1``, served by ``sessions_pkey``) plus a lateral EXISTS over the tiny
# ``session_cancel_markers`` table — no CTEs, no cross-session materialization,
# no correlated ``NOT EXISTS`` over ``events``, and asymptotically O(1) in event
# count (the plan does not touch ``events`` at all).
#
# SAFETY (the wedge-class constraint, issue #1659 comment): this predicate is a
# proven **over-approximation** of ``find_sessions_needing_inference`` — it is
# TRUE whenever the full sweep could return this session, so a ``False`` result
# means there is *provably* no pending work. The caller may therefore ONLY
# early-out on ``False``; on ``True`` ("maybe work") it MUST fall through to the
# full ``find_sessions_needing_inference``. A wrong predicate is then bounded to
# an occasional *extra* full sweep (cheap), NEVER a missed wake (a wedged
# session).
#
# WHY IT IS AN OVER-APPROXIMATION — the full sweep returns a session iff:
#   (a/b) it is ``session_active_predicate``-true AND survives the incomplete-
#         batch filter (``_filter_incomplete_batches`` can only REMOVE, never
#         add — so ``active`` is a superset of the (a/b) contribution);
#   (c)   it has a confirmed-but-unresolved tool call. Such a call is an
#         assistant tool_call with no result → it contributes to
#         ``open_tool_call_count > 0`` and the session is non-errored (confirmed
#         is subtracted by ``errored``) → it is ALREADY covered by
#         ``session_active_predicate``; confirmed ⊆ active.
#   (cancel) it carries an unharvested cancel-marker — UNIONed BELOW the errored
#         subtraction, so it can fire on an idle/errored-parked session. This is
#         the ONE return path ``session_active_predicate`` does not subsume, so
#         it is OR-ed in explicitly via the marker EXISTS.
# Hence ``active OR has-unharvested-cancel-marker`` is TRUE on every session the
# full sweep could return: a proven over-approximation. The ``archived_at IS
# NULL`` fence matches both the sweep's candidate WHERE and the marker join.
FAST_PATH_PENDING_WORK_SQL = f"""
    SELECT
        s.archived_at IS NULL
        AND (
            {session_active_predicate("s")}
            OR EXISTS (
                SELECT 1 FROM session_cancel_markers m
                 WHERE m.session_id = s.id
                   AND m.harvested_at IS NULL
            )
        ) AS has_pending_work
      FROM sessions s
     WHERE s.id = $1
"""


async def session_has_pending_work(
    conn: asyncpg.Connection[Any],
    session_id: str,
) -> bool:
    """Cheap necessary-condition gate for the per-turn admission guard (#1659).

    Returns ``True`` when the session *may* need inference and ``False`` only
    when there is **provably** no pending work. A proven over-approximation of
    :func:`find_sessions_needing_inference` (see ``FAST_PATH_PENDING_WORK_SQL``):
    the caller may early-out on ``False`` but MUST fall through to the full sweep
    on ``True`` — so a wrong predicate costs at most an extra full sweep, never a
    missed wake.

    A single PK-scoped lookup on ``sessions`` (no CTEs, no ``events`` scan), so
    it does not grow with event history — the whole point of the fast path
    versus the multi-CTE ``find_sessions_needing_inference``.

    A missing session (``id`` not found) returns ``False``: no row → no work.
    """
    row = await conn.fetchrow(FAST_PATH_PENDING_WORK_SQL, session_id)
    if row is None:
        return False
    return bool(row["has_pending_work"])


# ─── shared surface loading ──────────────────────────────────────────────────


def _build_surfaces(agent_rows: list[Any]) -> dict[str, _SweepAgentSurface]:
    """Materialize ``AGENT_SURFACE_SQL`` rows into ``_SweepAgentSurface`` per
    session — the thin duck-typed ``Agent`` the disposition classifier reads.
    """
    from aios.models.agents import HttpServerSpec, load_tool_specs

    surface_by_session: dict[str, _SweepAgentSurface] = {}
    for r in agent_rows:
        tools_list = r["tools"]
        http_list = r["http_servers"]
        surface_by_session[r["session_id"]] = _SweepAgentSurface(
            tools=load_tool_specs(tools_list or []),
            http_servers=[HttpServerSpec.model_validate(h) for h in (http_list or [])],
        )
    return surface_by_session


async def _load_surfaces(
    conn: asyncpg.Connection[Any], session_ids: list[str]
) -> tuple[dict[str, _SweepAgentSurface], dict[str, set[str]]]:
    """Load the per-session agent surface and confirmed-``allow`` tool_call ids
    for ``session_ids``.

    Two batched queries (no N+1): ``AGENT_SURFACE_SQL`` (tools + http_servers,
    version-pinned) and ``GHOST_LIFECYCLE_SQL`` (``tool_confirmed``/``allow``
    events). Returns ``(surface_by_sid, confirmed_by_sid)``. The confirmed set
    is exactly the one :func:`_was_dispatched` consumes as ``confirmed_ids``, so
    both ghost repair and the inference filter resolve the identical
    confirmation-satisfied bit.
    """
    if not session_ids:
        return {}, {}

    agent_rows = await conn.fetch(AGENT_SURFACE_SQL, session_ids)
    surface_by_sid = _build_surfaces(agent_rows)

    lifecycle_rows = await conn.fetch(GHOST_LIFECYCLE_SQL, session_ids)
    confirmed_by_sid: dict[str, set[str]] = {}
    for r in lifecycle_rows:
        confirmed_by_sid.setdefault(r["session_id"], set()).add(r["tool_call_id"])

    return surface_by_sid, confirmed_by_sid


# ─── ghost repair ────────────────────────────────────────────────────────────

# The sweep-maintained floor advance (#1746). ``open_tool_call_floor_seq`` is
# written in EXACTLY this one statement — nowhere else in the codebase sets
# it (structurally guarded by
# ``tests/unit/test_open_tool_call_floor_seq_single_writer.py``). Bulk,
# set-based ``UPDATE ... FROM unnest(...)`` rather than one round trip per
# session: a single cross-session sweep pass can touch thousands of sessions
# and a per-row ``executemany`` would reintroduce a linear-in-session-count
# round-trip cost this whole feature exists to eliminate. ``GREATEST`` makes
# the write monotonic and race-safe: two concurrent sweeps computing bounds
# from consistent (possibly different) snapshots each produce a valid lower
# bound (invariant: oldest-open-call.seq is non-decreasing over time), and the
# max of two valid lower bounds is still a valid lower bound — so concurrent
# advances can never regress the floor below a value it already holds.
_ADVANCE_OPEN_TOOL_CALL_FLOOR_SQL = """
    UPDATE sessions
       SET open_tool_call_floor_seq = GREATEST(open_tool_call_floor_seq, v.floor_seq)
      FROM (
          SELECT * FROM unnest($1::text[], $2::bigint[]) AS t(session_id, floor_seq)
      ) v
     WHERE sessions.id = v.session_id
"""


async def _advance_open_tool_call_floor(
    pool: asyncpg.Pool[Any],
    session_ids: list[str],
    min_open_seq: dict[str, int],
) -> None:
    """Advance ``open_tool_call_floor_seq`` for every session with a computed bound.

    ``min_open_seq`` is populated in :func:`find_and_repair_ghosts` from the
    reconciliation it already performs — the seq of the oldest fetched
    assistant batch still carrying a no-result tcid, per session. Given a
    previously-valid floor, ``min_open_seq`` is well-defined for every session
    ``GHOST_ASST_SQL`` returned (the ``open_tool_call_count > 0`` gate
    guarantees a genuinely-open call exists somewhere, and a valid floor
    guarantees its batch's seq is ``>= floor`` — hence fetched). A session
    absent from ``min_open_seq`` (which invariant 0 says should not happen)
    is simply left unadvanced — safe, since a stale-low floor is never
    unsound, only a slower scan.

    A no-op cross-session round trip when ``min_open_seq`` is empty (every
    fetched batch's tcids all resolved via a race with a concurrent append,
    or ``session_ids`` itself is empty) — ``asyncpg`` executes ``UPDATE ...
    FROM unnest($1::text[], $2::bigint[])`` against two empty arrays and
    matches zero rows.
    """
    ids = [sid for sid in session_ids if sid in min_open_seq]
    if not ids:
        return
    floors = [min_open_seq[sid] for sid in ids]
    async with pool.acquire() as conn:
        await conn.execute(_ADVANCE_OPEN_TOOL_CALL_FLOOR_SQL, ids, floors)


async def find_and_repair_ghosts(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    *,
    session_id: str | None = None,
) -> list[tuple[str, str]]:
    """Recover ghost tool calls: re-park resumable tasks, error-repair the rest.

    A ghost is a tool_call_id from an assistant message where:

    - No tool-role result event exists in the log.
    - No asyncio task is in-flight (InflightToolRegistry).
    - The harness would have dispatched the tool (i.e. it's not a
      custom tool or an unconfirmed ``always_ask`` tool still waiting
      for client action).

    A ghost whose tool is a registered ``resumable`` builtin
    (:meth:`registry.resumable_tool_names`) is a **pure-await** that crash-recovery
    RE-PARKS rather than error-repairs (#1431): its
    servicer is re-derived from the durable edge (:func:`queries.find_parked_servicer`) and
    a fresh resume task is launched (a pure read of durable state) so the servicer's
    exactly-once answer lands in the original tool result. Only a resumable ghost whose edge
    is absent — the launch crashed before it was durable — falls through to a retryable
    ``launch_lost`` error. Every other ghost keeps the side-effect-conservative
    error-repair below.

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
    in_flight = inflight_tool_registry.all_in_flight_tool_call_ids()

    scope_clause = "AND e.session_id = $1" if session_id else ""
    scope_params: list[Any] = [session_id] if session_id else []

    # Rollout ordering fail-closed guard (#1746): ``aios-worker`` boots
    # straight into new code with no migration step of its own, so a naive
    # single-image promote can serve this code before the post-deploy ``aios
    # migrate`` has added ``open_tool_call_floor_seq``. Rather than let the
    # whole cross-session sweep abort with ``UndefinedColumnError`` (a
    # fleet-wide wake outage), degrade ALL THREE floor-bounded queries to
    # their pre-#1746 unbounded twins for this pass — never fail dark. The
    # floor is not advanced in this mode (``floor_column_present`` gates the
    # advance call below): with no reliable floor-bounded fetch, there is
    # nothing safe to derive a tighter bound from.
    floor_column_present = True

    async with pool.acquire() as conn:
        try:
            asst_rows = await conn.fetch(
                GHOST_ASST_SQL.format(scope_clause=scope_clause),
                *scope_params,
            )
        except asyncpg.UndefinedColumnError:
            log.warning("sweep.open_tool_call_floor_seq_missing_column_fallback")
            floor_column_present = False
            asst_rows = await conn.fetch(
                _GHOST_ASST_SQL_UNBOUNDED_FALLBACK.format(scope_clause=scope_clause),
                *scope_params,
            )

        if not asst_rows:
            return []

        session_ids = list({r["session_id"] for r in asst_rows})

        if floor_column_present:
            result_rows = await conn.fetch(GHOST_RESULT_ROWS_SQL, session_ids)
        else:
            result_rows = await conn.fetch(_GHOST_RESULT_ROWS_SQL_UNBOUNDED_FALLBACK, session_ids)
        results_by_session: dict[str, set[str]] = {}
        for r in result_rows:
            results_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

        if floor_column_present:
            lifecycle_rows = await conn.fetch(GHOST_LIFECYCLE_SQL, session_ids)
        else:
            lifecycle_rows = await conn.fetch(_GHOST_LIFECYCLE_SQL_UNBOUNDED_FALLBACK, session_ids)
        confirmed_by_session: dict[str, set[str]] = {}
        for r in lifecycle_rows:
            confirmed_by_session.setdefault(r["session_id"], set()).add(r["tool_call_id"])

    # First pass: find candidate ghosts (no result, no in-flight task).
    # We don't yet know their dispatch status — that requires agent config.
    # ``created_at`` is the assistant turn's emit time, carried so the
    # abandoned-client-call branch can age-bound off it (#752). Also tracks,
    # per session, ``min_open_seq`` — the seq of the OLDEST fetched batch that
    # still has an OPEN tcid (no result row) — the sweep-derived floor advance
    # (#1746).
    #
    # "Open" here means EXACTLY what ``open_tool_call_count`` means (no result
    # row), NOT the narrower "no result AND not in-flight" ghost-candidacy
    # test below. This distinction is load-bearing: an in-flight call has no
    # result yet, so it IS genuinely open — its batch must still anchor the
    # floor, because the in-flight task can crash later (after this floor
    # advance persists) and become a true ghost. Excluding an in-flight-only
    # batch from the floor computation would let the floor advance past a call
    # that has not actually resolved, reintroducing the exact permanent-wedge
    # class this redesign exists to prevent. This is also why ``min_open_seq``
    # is ALWAYS well-defined for a session ``GHOST_ASST_SQL`` returns: that
    # query's ``open_tool_call_count > 0`` gate guarantees (invariant 0's sound
    # side) at least one genuinely no-result tcid exists somewhere in the
    # session, and a valid floor guarantees its batch's seq is ``>= floor`` —
    # hence fetched. So there is no "no open batch found" branch to handle.
    candidates: list[_Candidate] = []
    min_open_seq: dict[str, int] = {}

    for row in asst_rows:
        sid = row["session_id"]
        tool_calls = row["tool_calls"] or []
        seq = row.get("seq")
        created_at = row["created_at"]
        existing_results = results_by_session.get(sid, set())
        session_in_flight = in_flight.get(sid, set())

        batch_has_open_tcid = False
        for tc in tool_calls:
            tcid = tc.get("id")
            if not tcid or tcid in existing_results:
                continue
            # No result row → genuinely open (regardless of in-flight status)
            # → this batch anchors the floor.
            batch_has_open_tcid = True
            if tcid in session_in_flight:
                # In-flight: genuinely open, but a live task is already
                # servicing it — not a ghost-repair candidate.
                continue
            function = tc.get("function") or {}
            name = function.get("name", "")
            candidates.append(
                _Candidate(sid, tcid, name, created_at, arguments=function.get("arguments"))
            )

        if batch_has_open_tcid and seq is not None:
            prior = min_open_seq.get(sid)
            if prior is None or seq < prior:
                min_open_seq[sid] = seq

    # Advance the floor for EVERY in-scope session, before the
    # ``not candidates`` early-return below — a session whose fetched batches
    # are all fully resolved must still advance its floor past them (#1746).
    # ``GREATEST``-only (never lowers the stored floor); skipped entirely when
    # the column was missing this pass — the fallback branch fetched from the
    # unbounded twins, so ``min_open_seq`` was derived without a floor
    # guarantee and is not safe to write back as one.
    if floor_column_present:
        await _advance_open_tool_call_floor(pool, session_ids, min_open_seq)

    if not candidates:
        return []

    # Second pass: load agent config only for sessions with candidates,
    # then classify each candidate into one of two repairable buckets
    # (dispatched ghost, or abandoned client-result-pending call past the age
    # bound). All other candidates are left alone (legitimately waiting).
    candidate_sids = list({c.session_id for c in candidates})
    async with pool.acquire() as conn:
        # ``AGENT_SURFACE_SQL`` LEFT JOINs agent_versions to respect version
        # pinning and fetches ``http_servers`` alongside ``tools`` so the
        # classifier can apply the arg-aware route refinement for
        # ``http_request`` (#1076) — the same refinement the dispatch and read
        # paths apply.
        agent_rows = await conn.fetch(AGENT_SURFACE_SQL, candidate_sids)

    agent_surface_by_session = _build_surfaces(agent_rows)

    # Abandoned-client-call bound (#752): a client-result-pending call whose
    # assistant turn is older than this is treated as abandoned. The cutoff is
    # computed once per sweep against ``created_at`` (the assistant turn's emit
    # time), consistent with the dispatched-ghost age semantics.
    client_max_age = get_settings().client_tool_call_max_age_seconds
    abandoned_cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=client_max_age)

    ghosts: list[_Candidate] = []
    abandoned: list[_Candidate] = []
    for c in candidates:
        confirmed = confirmed_by_session.get(c.session_id, set())
        surface = agent_surface_by_session.get(c.session_id, _EMPTY_SURFACE)
        if _was_dispatched(c, confirmed, surface):
            ghosts.append(c)
        elif _is_client_result_pending(c.tool_name, surface) and c.created_at < abandoned_cutoff:
            # Not dispatched AND client-result-pending AND older than the bound:
            # the client disconnected and will never return a result. Without
            # resolving it, the call's open_tool_call_count contribution keeps
            # the session a permanent wake candidate with no progress (#155 loop,
            # regression from #750). Confirmation-pending always_ask calls are
            # NOT client-result-pending, so they fall through here and are left
            # to wait on the user.
            abandoned.append(c)

    # #1431: a parked ``call_*`` ghost is a PURE-AWAIT, not a side-effectful tool — its
    # servicer is still running and will answer exactly once. Re-derive the servicer from
    # the durable edge (the ``tool_call_id`` the handler stamped onto ``caller``) and
    # RE-PARK it (a pure read), so the answer lands in the original tool result instead of
    # being orphaned by a synthetic error. Only when no edge exists (the launch crashed
    # before it was durable) does the call fall through to a retryable ``launch_lost`` error.
    # Lazy import: ``sweep`` ↔ ``tool_dispatch`` is a mutual-lazy-import pair (the
    # symmetric counterpart of ``_trigger_sweep``'s lazy ``sweep`` import). The tool
    # registry is the single source of truth for which builtins are pure-await
    # resumables (``resumable=True`` at registration); no separate name list.
    from aios.harness.tool_dispatch import relaunch_parked_task
    from aios.tools.registry import registry

    resumable_names = registry.resumable_tool_names()
    resumable = [c for c in ghosts if c.tool_name in resumable_names]
    ghosts = [c for c in ghosts if c.tool_name not in resumable_names]
    launch_lost: list[_Candidate] = []
    for c in resumable:
        # Per-ghost isolation, matching the error-repair + wake-defer loops below: this
        # runs cross-session (boot sweep), so one gone caller (``NotFoundError`` from
        # ``load_session_account_id``) or a transient DB error must NOT abort recovery of
        # the other tenants' ghosts. A gone caller has nothing to resume → log + skip.
        try:
            c_account_id = await sessions_service.load_session_account_id(pool, c.session_id)
            async with pool.acquire() as conn:
                handle = await find_parked_servicer(
                    conn,
                    caller_session_id=c.session_id,
                    tool_call_id=c.tool_call_id,
                    account_id=c_account_id,
                )
            if handle is None:
                launch_lost.append(c)
                continue
            servicer_kind, servicer_id, request_id, output_schema = handle
            relaunch_parked_task(
                pool,
                c.session_id,
                call={
                    "id": c.tool_call_id,
                    "function": {"name": c.tool_name, "arguments": c.arguments},
                },
                servicer_kind=servicer_kind,
                servicer_id=servicer_id,
                request_id=request_id,
                output_schema=output_schema,
                account_id=c_account_id,
            )
        except Exception:
            log.exception(
                "sweep.repark_failed", session_id=c.session_id, tool_call_id=c.tool_call_id
            )
            continue
        log.info(
            "sweep.task_reparked",
            session_id=c.session_id,
            tool_call_id=c.tool_call_id,
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
        )

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
        # it also fires for a crash in the window between the span commit and
        # the actual side-effectful invoke (an MCP auth resolve / parameter
        # validation dying mid-flight, or worker death). A cancel landing on the
        # span ``await`` is normally NOT among these — ``_tool_lifecycle`` resolves
        # it eagerly as ``cancelled`` (the body provably never ran). It only reaches
        # this branch if a *second* cancel interrupts that eager cleanup, where this
        # is the backstop. The remaining crash cases produce false "verify the
        # outcome" advice but never the dangerous false "safe to retry" that this
        # design eliminates. Tighter classification would need a second marker
        # before each tool's side-effectful call — deferred until the residual
        # over-pessimism is shown to matter.
        if (c.session_id, c.tool_call_id) in started:
            repair_items.append(
                (
                    c,
                    "may_have_completed",
                    "Tool dispatch was interrupted after execution began. "
                    "The tool may have completed and side effects may have "
                    "committed. Verify the outcome before retrying. "
                    "If the original call carried an Idempotency-Key header, "
                    "retrying the identical request with the same header "
                    "value is safe at providers that honor it; otherwise "
                    "verify the outcome out-of-band before retrying.",
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
    for c in launch_lost:
        # #1431: a resumable ``call_*`` ghost with no servicer edge — the launch crashed
        # before the request reached a servicer, so nothing is running to await. Safe to
        # retry (no servicer was created or served), unlike the may-have-completed branch.
        repair_items.append(
            (
                c,
                "launch_lost",
                "The task did not start before the worker restarted; "
                "nothing was launched. You may retry.",
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
        # under a concurrent sweep run, violating invariant #4
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


async def repark_stranded_model_dispatch(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str | None = None,
) -> int:
    """Re-park sessions stranded on a model-dispatch park whose harvest task was lost (#1635).

    The model-dispatch analog of the ``call_*`` re-park inside
    :func:`find_and_repair_ghosts`. The existing ghost scan only re-parks servicers backed
    by an open ``tool_call_id`` in a *persisted assistant message* — a model-dispatch park
    has neither (the assistant message is the bound run's OUTPUT, produced only after the
    park resolves), so a worker crash while parked strands the session: the run completes
    and writes its response, but nothing re-parks the outer session to consume it.

    This branch closes that gap. :func:`queries.find_unharvested_model_dispatch_parks`
    re-derives every session whose latest un-consumed park has no harvest event yet (the
    run's terminal state was never written back); for each, a fresh harvest task is
    launched via :func:`model_workflow.relaunch_model_dispatch_park`. That task is a pure
    read of the run's durable terminal state followed by one idempotent harvest append
    (dedup-guarded on ``run_id``), so it is safe to run even if the original task is in
    fact still alive — and ``relaunch_model_dispatch_park`` skips a key already in-flight
    in this worker, so a steady-state park (its live task running) is never double-parked.

    Returns the number of harvest tasks launched (the recovery count).
    """
    # Lazy import: ``sweep`` is imported widely at harness boot; ``model_workflow`` pulls
    # in the workflows service layer, so deferring the import keeps the import graph flat
    # (mirrors the lazy ``tool_dispatch`` import in ``find_and_repair_ghosts``).
    from aios.harness.model_workflow import relaunch_model_dispatch_park

    async with pool.acquire() as conn:
        stranded = await find_unharvested_model_dispatch_parks(conn, session_id=session_id)

    reparked = 0
    for sid, run_id, account_id in stranded:
        # Per-park isolation: cross-session at boot, so one transient failure must not
        # abort recovery of the others. ``relaunch_model_dispatch_park`` is synchronous
        # (it only spawns the task) — wrap defensively all the same.
        try:
            if relaunch_model_dispatch_park(pool, sid, run_id=run_id, account_id=account_id):
                reparked += 1
        except Exception:
            log.exception("sweep.model_dispatch_repark_failed", session_id=sid, run_id=run_id)
            continue
    if reparked:
        log.info("sweep.model_dispatch_reparked", count=reparked)
    return reparked


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


def _is_client_result_pending(name: str, surface: _SweepAgentSurface) -> bool:
    """True when the classifier says the call awaits a CLIENT result.

    Confirmation-pending calls classify as ``NEEDS_CONFIRM`` and remain parked
    for the user rather than being errored by abandoned-client-call repair.
    """
    from aios.harness.tool_disposition import ToolDisposition, classify_tool_call

    return (
        classify_tool_call(
            name,
            None,
            surface,  # type: ignore[arg-type]  # duck-typed: classifier reads .tools/.http_servers
            confirmation_resolved=False,
        )
        is ToolDisposition.CUSTOM
    )


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
    inflight_tool_registry: InflightToolRegistry,
    *,
    session_id: str | None = None,
    reacted_floor: int | None = None,
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

    ``reacted_floor`` (requires ``session_id``) is the #253 preemption
    trigger: the SAME predicate evaluated against the in-flight step's
    context watermark instead of the committed ``last_reacted_seq`` —
    "would this session be wake-eligible immediately after the current
    step finished?". See :func:`_find_needing_inference_floored` for the
    two floored-only restrictions.
    """
    if reacted_floor is not None:
        if session_id is None:
            raise ValueError("reacted_floor requires session_id")
        return await _find_needing_inference_floored(
            pool, inflight_tool_registry, session_id, reacted_floor
        )
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
        # C2: a non-archived session with an unharvested cancel-marker must run its cancel
        # leaf even when idle or errored-parked — it still owes a ``cancelled`` response. The
        # session-side analog of the run sweep's unharvested-cancel-signal clause; UNIONed
        # BELOW the errored subtraction so the park can't suppress the exit.
        cancel_marked = await list_session_ids_with_unharvested_cancel_marker(
            conn, session_id=session_id
        )

    candidates -= errored
    confirmed_sessions -= errored
    to_filter = candidates - confirmed_sessions
    if not to_filter:
        return confirmed_sessions | cancel_marked

    # Span the batch filter (#1729): ``sweep.batch_filter_start/end``. Before
    # #1729 this region fetched the session's entire assistant/tool-result
    # lifetime (126 MB observed) and decoded it on the event loop — a
    # multi-second pre-model stall that was invisible to every existing span
    # (``sweep.query_exec`` measured only the scalar-gate SQL, ~0.01s). Bracket
    # it so the next residual on this path is a query, not an archaeology dig
    # (same lesson as #1658/#1725). Only spanned on the per-step scoped path
    # (``session_id`` set), where an ``account_id`` is resolvable; the
    # cross-session sweep has no single session to stamp.
    if session_id is not None:
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        bf_start = await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {"event": "sweep.batch_filter_start", "candidate_count": len(to_filter)},
            account_id=account_id,
        )
        try:
            filtered = await _filter_incomplete_batches(pool, inflight_tool_registry, to_filter)
        finally:
            await sessions_service.append_event(
                pool,
                session_id,
                "span",
                {
                    "event": "sweep.batch_filter_end",
                    "start_id": bf_start.id,
                },
                account_id=account_id,
            )
    else:
        filtered = await _filter_incomplete_batches(pool, inflight_tool_registry, to_filter)

    return filtered | confirmed_sessions | cancel_marked


async def _find_needing_inference_floored(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    session_id: str,
    reacted_floor: int,
) -> set[str]:
    """The #253 preemption-trigger variant of the wake predicate.

    Same arms as :func:`find_sessions_needing_inference` — candidate gate,
    unreacted batch filter, confirmed-dispatch union, errored subtraction,
    cancel-marker union — with every seq-anchored comparison raised to the
    in-flight step's context watermark (the ``*_FLOORED_SQL`` constants, each
    composed from the same single-source fragment as its committed twin).
    Cancel markers need no floor: the step harvests them before building
    context, so any unharvested marker post-dates the watermark.

    Two floored-only restrictions, both erring toward NOT preempting (the
    inverse of the sweep's wedge-safety doctrine — a missed wake wedges a
    session forever, a missed preempt costs one stale model call that the
    already-queued wake supersedes next step):

    * The #1710 empty-unreacted dispatch-narrowing pass is skipped
      (``preempt_floor`` on :func:`_filter_incomplete_batches`): its
      wedge-safety argument assumes :func:`find_and_repair_ghosts` ran first,
      which only the composed sweep guarantees — and anything it admits was
      already true when the step's entry guard ran this predicate unfloored,
      so it cannot represent an *arriving* event.
    * The confirmed arm subtracts already-dispatched calls (in-flight in the
      registry, or bearing a durable ``tool_execute_start`` span) — see
      ``CONFIRMED_ROWS_FLOORED_SQL``.

    No span bracketing around the batch filter: this runs from the preempt
    watcher (potentially once per poll tick), and span appends here would
    both spam the log and re-trip the watcher's own ``last_event_seq`` gate.
    """
    confirmed_max_age_seconds = get_settings().confirmed_dispatch_max_age_seconds
    async with pool.acquire() as conn:
        candidate_rows = await conn.fetch(CANDIDATE_ROWS_FLOORED_SQL, session_id, reacted_floor)
        candidates = {r["session_id"] for r in candidate_rows}

        confirmed_rows = await conn.fetch(
            CONFIRMED_ROWS_FLOORED_SQL,
            session_id,
            confirmed_max_age_seconds,
            reacted_floor,
        )
        confirmed_tcids = {r["tool_call_id"] for r in confirmed_rows if r["tool_call_id"]}
        confirmed_tcids -= inflight_tool_registry.in_flight_tool_call_ids(session_id)
        if confirmed_tcids:
            span_rows = await conn.fetch(GHOST_SPAN_START_SQL, [session_id], list(confirmed_tcids))
            confirmed_tcids -= {r["tool_call_id"] for r in span_rows}
        confirmed_sessions = {session_id} if confirmed_tcids else set()

        errored = await _errored_session_ids(conn, session_id=session_id)
        cancel_marked = await list_session_ids_with_unharvested_cancel_marker(
            conn, session_id=session_id
        )

    candidates -= errored
    confirmed_sessions -= errored
    to_filter = candidates - confirmed_sessions
    if not to_filter:
        return confirmed_sessions | cancel_marked
    filtered = await _filter_incomplete_batches(
        pool, inflight_tool_registry, to_filter, preempt_floor=reacted_floor
    )
    return filtered | confirmed_sessions | cancel_marked


async def _filter_incomplete_batches(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    candidates: set[str],
    *,
    preempt_floor: int | None = None,
) -> set[str]:
    """Remove sessions whose only unreacted events are tool results from
    in-progress batches (where sibling tools are still in-flight).

    The unreacted set is fetched in ONE seq-bounded batched query
    (``UNREACTED_ROWS_SQL``, gated by ``seq > last_reacted_seq``). The assistant
    batches and their tool results are then fetched per session, but ONLY the
    batches that OWN an unreacted tool_call_id and ONLY the tool results for
    those batches' ids — via ``@>`` containment and ``= ANY`` rather than the
    pre-#1729 unbounded lifetime scans (which pulled every ``role='tool'`` row
    plus every assistant ``tool_calls`` payload — 126 MB observed — and decoded
    it on the event loop on every full sweep). See the query docstrings above.
    Sessions with no unreacted events (or whose unreacted set contains a user
    message) are decided without touching ``events`` at all — EXCEPT the
    dispatch-narrowing fetch below.

    Empty-unreacted / no-in-flight sessions are narrowed further: a session is
    woken from that branch ONLY if at least one of its open tool calls (no
    result, no in-flight task) was actually **dispatched** by the harness. A
    session parked purely on externally-executed work — a client ``custom`` call
    awaiting the client's result POST, or an ``always_ask`` call awaiting
    operator confirmation — is NOT dispatched, so it is left alone rather than
    re-fired on every sweep (a full paid model step with only a
    ``_PENDING_EXTERNAL`` placeholder as new context, #1710). This narrowing's
    supporting fetch (``OPEN_CANDIDATES_ASST_SQL``) is scoped to ONLY the sids
    that reach the empty-unreacted branch (typically a handful) and further
    bounded by ``open_tool_call_count > 0`` — it does not resurrect the
    pre-#1729 unbounded lifetime scan for the common (has-unreacted) path.

    **Wedge-safety invariant (fail toward waking too much, never too little):**
    a missed wake permanently wedges a months-long session; an extra wake costs
    one step. This narrowing is safe because :func:`find_and_repair_ghosts` runs
    BEFORE this filter (see the composed sweep): a genuinely-dispatched-but-
    taskless call (a crashed-worker ghost) is error-repaired first, so its
    synthesized result becomes an *unreacted* event and the session no longer
    reaches this empty-unreacted branch; confirmed-``allow`` calls are woken via
    the separate ``confirmed_sessions`` union that bypasses this filter. So by
    the time a session lands here with an open, result-less, taskless call, that
    call is either an unconfirmed ``always_ask`` or a client ``custom`` — both
    ``not _was_dispatched`` — and a crashed built-in ghost classifies as
    dispatched, so the narrowing never holds one back.

    Uses batched queries across all candidates (no N+1).
    """
    session_list = list(candidates)

    async with pool.acquire() as conn:
        if preempt_floor is not None:
            unreacted_rows = await conn.fetch(
                UNREACTED_ROWS_FLOORED_SQL, session_list, preempt_floor
            )
        else:
            unreacted_rows = await conn.fetch(UNREACTED_ROWS_SQL, session_list)
        unreacted_by_sid = _group_unreacted_rows(unreacted_rows)

        result: set[str] = set()
        # Sids whose unreacted set is empty and have no in-flight task —
        # deferred to a single batched dispatch-narrowing pass below (#1710)
        # rather than queried one-by-one inside this loop.
        empty_no_inflight: list[str] = []

        for sid in candidates:
            in_flight = inflight_tool_registry.in_flight_tool_call_ids(sid)
            unreacted = unreacted_by_sid.get(sid, [])

            if not unreacted:
                if not in_flight:
                    empty_no_inflight.append(sid)
                continue

            if any(role == "user" for role, _ in unreacted):
                result.add(sid)
                continue

            unreacted_tcids = {tcid for _, tcid in unreacted if tcid}
            if not unreacted_tcids:
                # Unreacted non-user events with no tool_call_id: pre-#1729 the
                # assistant loop ran but no batch intersected the (empty)
                # unreacted-tcid set, so the session was NOT admitted. Preserve
                # that — skip the (now pointless) bounded fetch and drop it.
                continue

            # Bounded fetch (#1729): only the assistant batches that OWN one of
            # this session's unreacted tool_call_ids, projected to their id
            # arrays (no payload). One containment probe per unreacted tcid.
            probes = [json.dumps([{"id": tcid}]) for tcid in unreacted_tcids]
            asst_rows = await conn.fetch(REFERENCED_ASST_BATCH_SQL, sid, probes)

            referenced_batches: list[set[str]] = []
            all_batch_ids: set[str] = set()
            for r in asst_rows:
                batch_ids = {tcid for tcid in (r["tool_call_ids"] or []) if tcid}
                if batch_ids & unreacted_tcids:
                    referenced_batches.append(batch_ids)
                    all_batch_ids |= batch_ids

            if not referenced_batches:
                continue

            # Bounded fetch (#1729): only the tool results for THESE batches'
            # ids, not the session's entire ``role='tool'`` history.
            result_rows = await conn.fetch(BATCH_RESULT_ROWS_SQL, sid, list(all_batch_ids))
            result_ids = {r["tool_call_id"] for r in result_rows if r["tool_call_id"]}

            for batch_ids in referenced_batches:
                if batch_ids <= result_ids:
                    result.add(sid)
                    break

        # Dispatch-narrowing pass (#1710): batched across ONLY the sids that
        # landed in the empty-unreacted/no-in-flight branch — not the whole
        # candidate set — so the common (has-unreacted) path never pays for
        # it. ``OPEN_CANDIDATES_ASST_SQL`` mirrors ``GHOST_ASST_SQL``'s
        # ``open_tool_call_count > 0`` bound.
        #
        # Skipped on the #253 floored path (``preempt_floor``): this branch's
        # wedge-safety argument assumes ghost repair ran first (true only in
        # the composed sweep), and anything it admits — a dispatched-but-
        # taskless open call — was already true when the step's entry guard
        # ran this predicate unfloored, so it cannot represent an *arriving*
        # event. Admitting it would preempt-loop on state the restarted step
        # cannot progress. False-negative-safe: no preempt → the step
        # completes → the composed sweep handles the ghost as today.
        if empty_no_inflight and preempt_floor is None:
            all_result_rows = await conn.fetch(ALL_RESULT_ROWS_SQL, empty_no_inflight)
            results_by_sid: dict[str, set[str]] = {}
            for r in all_result_rows:
                results_by_sid.setdefault(r["session_id"], set()).add(r["tool_call_id"])

            asst_rows = await conn.fetch(OPEN_CANDIDATES_ASST_SQL, empty_no_inflight)
            asst_by_sid: dict[str, list[dict[str, Any]]] = {}
            for r in asst_rows:
                asst_by_sid.setdefault(r["session_id"], []).append(r["data"])

            # Agent surface + confirmed-``allow`` ids for the dispatch
            # classifier. The classifier reads only ``.tools``/``.http_servers``
            # and the confirmed set matches the confirmed-dispatch path by
            # construction.
            surface_by_sid, confirmed_by_sid = await _load_surfaces(conn, empty_no_inflight)

            for sid in empty_no_inflight:
                in_flight = inflight_tool_registry.in_flight_tool_call_ids(sid)
                # Wake only if at least one open call was dispatched (#1710).
                # An empty ``open_calls`` (a stale counter — the maintained
                # ``open_tool_call_count > 0`` with no actual open assistant
                # tool_call) yields ``any(...) is False`` → not woken, which is
                # correct: there is no open call to react to, and a real
                # stimulus wakes the session via the normal unreacted path.
                open_calls = _open_candidates_for(sid, asst_by_sid, results_by_sid, in_flight)
                confirmed = confirmed_by_sid.get(sid, set())
                surface = surface_by_sid.get(sid, _EMPTY_SURFACE)
                if any(_was_dispatched(c, confirmed, surface) for c in open_calls):
                    result.add(sid)

    return result


def _open_candidates_for(
    sid: str,
    asst_by_sid: dict[str, list[dict[str, Any]]],
    results_by_sid: dict[str, set[str]],
    in_flight: set[str],
) -> list[_Candidate]:
    """Build the OPEN tool_calls for ``sid`` — assistant tool_calls with no
    tool-result and no in-flight task — as ``_Candidate`` objects the dispatch
    classifier (#1710) can inspect.

    Mirrors the ghost-candidate construction in :func:`find_and_repair_ghosts`.
    ``created_at`` is not carried on the batch-filter's assistant rows and the
    disposition classifier does not read it, so a placeholder is used. Only
    ``tool_name``/``arguments``/``tool_call_id`` drive ``_was_dispatched``.
    """
    result_ids = results_by_sid.get(sid, set())
    open_calls: list[_Candidate] = []
    for asst_data in asst_by_sid.get(sid, []):
        for tc in asst_data.get("tool_calls") or []:
            tcid = tc.get("id")
            if not tcid or tcid in result_ids or tcid in in_flight:
                continue
            function = tc.get("function") or {}
            open_calls.append(
                _Candidate(
                    session_id=sid,
                    tool_call_id=tcid,
                    tool_name=function.get("name", ""),
                    created_at=dt.datetime.now(dt.UTC),
                    arguments=function.get("arguments"),
                )
            )
    return open_calls


def _group_unreacted_rows(
    rows: list[Any],
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Group ``UNREACTED_ROWS_SQL`` rows by session_id into (role, tool_call_id)
    tuples. Projected-column counterpart of the raw per-row grouping done
    inline for ``OPEN_CANDIDATES_ASST_SQL`` (#1755) — the batch filter only
    ever reads these two fields for the unreacted set.
    """
    grouped: dict[str, list[tuple[str | None, str | None]]] = {}
    for r in rows:
        grouped.setdefault(r["session_id"], []).append((r["role"], r["tool_call_id"]))
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
    on the module-level ``aios.jobs.app`` singleton (which fixes
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
    inflight_tool_registry: InflightToolRegistry,
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
    repaired = await find_and_repair_ghosts(pool, inflight_tool_registry, session_id=session_id)
    # Crash-recovery for the model-dispatch park (#1635). A model-dispatch park is NOT
    # discoverable via the assistant-tool_call ghost scan above (it has no tool_call_id and
    # no persisted assistant message), so it needs its own re-park branch: re-launch the
    # harvest task for any session stranded with an unharvested park. Runs alongside ghost
    # repair every sweep (boot + periodic) — the harvest task it launches asynchronously
    # writes the harvest event and wakes the session, which then lands on the harvest fold.
    await repark_stranded_model_dispatch(pool, session_id=session_id)
    woken = await find_sessions_needing_inference(
        pool, inflight_tool_registry, session_id=session_id
    )
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
