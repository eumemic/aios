"""The open-obligations reminder (#1413) and the shared owed render (#1522).

An open **awaited** request edge (#1123 ``request_opened`` minus
``request_response``, ``awaited=true``) is an obligation the session must answer
with ``return``/``error``. Its only model-visible surface used to be a
render-time marker prepended to the *original* user message carrying
``metadata.request.request_id`` — which context windowing **erases** the moment
the conversation scrolls past it, exactly when the session has drifted far from
the ask and most needs the reminder.

This module renders the replacement: the **durable obligations reminder row**
(:func:`render_obligations_reminder`) the composer writes to the transcript when
an open obligation's original ask has left the window and the listing differs
from the one already in the window (``aios.harness.reminders``); plus the
one-line supersession (:data:`OBLIGATIONS_EMPTY_CONTENT`) written when the set
empties under a listing. Because the row is replayed byte-for-byte on every
later step, the render is a pure function of the open set — absolute
``opened_at`` timestamps, never a relative age. Obligations are a **distinct
plane** from channels — orthogonal to the request edge — so they live in their
own module rather than beside the channels listing.

The render is driven by :class:`~aios.models.sessions.Obligation` rows fetched
via :func:`aios.db.queries.sessions.get_open_obligations` (a full-log query, not
a slate-derived marker), so it survives windowing erasure of the original ask.
The same :func:`render_owed_entry` projection feeds the quiescence-attempt nudge
and the ``list_obligations`` tool, so every surface agrees on what is owed.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aios.models.sessions import Obligation

# Max obligations rendered as full lines; beyond this a ``+K more`` marker
# collapses the rest (mirrors ``trace_max_nodes``). Keeps the reserved row
# budget bounded REGARDLESS of obligation count — without a cap an unbounded
# count inflates the reserved budget until ``read_windowed_events`` raises
# ``ValueError`` (no budget for events) → step crash. Issue C adds the
# complementary per-session open-goal admission cap.
MAX_RENDERED_OBLIGATIONS = 10

# Requests up to this size remain verbatim in the reminder. Beyond it the
# reminder cannot re-render the task in full without blowing the reserved
# budget (see ``max_obligations_reminder_local``), so it renders a bounded,
# neutral pointer instead (:data:`_REMINDER_ABRIDGED`). The cap is load-bearing
# (#2080/#2071): an unbounded task in a persisted row would inflate every later
# window.
_TASK_MAX = 8192

# The genuinely-unavailable marker: NO content was ever persisted on the frame
# (pre-#1413), so there is nothing to abridge and nothing to point at. #2080's
# fail-loud property is correct on every surface: an absent task cannot be
# recovered from anywhere, so improvising one is the exact hazard.
_TASK_UNAVAILABLE = "[TASK CONTENT UNAVAILABLE — return an error; do not infer the task]"

# What an oversized task renders as on every reminder surface — the durable
# reminder row, the quiescence nudge and the ``list_obligations`` projection.
# All are REMINDERS about a request that is, by construction, still open, and
# all are persisted or persisted-adjacent (the row and the nudge sit in the
# window for as long as it keeps them), so ordering a refusal is never
# justified: the surface cannot prove the original is GONE, and #2080's own
# evidence is that a model obeys a trailing refuse-order — emitting one
# re-parks exactly the oversized-payload sessions #2221 exists to unpark.
#
# Deliberately carries NO content prefix: a plausible-looking prefix of a
# possibly-unrecoverable task is precisely #2080's improvisation hazard (a
# sub-agent read a prefix and invented the rest). A bare char-count pointer
# refuses BOTH failure modes — it never orders a refusal, and it offers nothing
# to improvise from — and keeps the durable row small regardless of task size.
_REMINDER_ABRIDGED = (
    "[TASK ABRIDGED IN THIS REMINDER — {chars} characters, over this reminder's "
    "{cap}-character budget. This is a reminder, not the task: the original request "
    "message (this request_id) carries it in full. If that message is not in your "
    "context, retrieve it with search_events before acting; do not infer the task "
    "from this line, and do not refuse on account of it.]"
)

# Max chars of a rendered ``output_schema`` contract (#1522). Kept narrower
# than the task budget because a schema is a structural contract and usually
# legitimately needs a few keys/types to be useful) — still a HARD cap so a large
# persisted schema can't blow the reserved row budget. A schema longer than this
# is JSON-serialised then elided to this width + an ellipsis. The bound is what
# keeps ``max_obligations_reminder_local`` a correct upper bound for the
# schema-bearing render: per-entry schema cost is capped REGARDLESS of the real
# schema size, so the reserved budget never overflows ``read_windowed_events``.
_SCHEMA_MAX = 240

_HEADER = "━━━ Open obligations (answer with return/error) ━━━"


def _origin_label(obligation: Obligation, *, session_id: str) -> str:
    """The ``[origin]`` label for an obligation's caller.

    ``api`` | ``session`` | ``run`` come straight off the trusted ``caller.kind``;
    a ``session`` caller that is the session ITSELF is a self-goal (#1414) and
    renders as ``self``. An unknown/absent kind renders bare so the line never
    crashes on a malformed frame.
    """
    if obligation.caller_kind == "session" and obligation.caller_id == session_id:
        return "self"
    return obligation.caller_kind or "?"


def _reminder_content(summary: str | None) -> str:
    """Render a task for a reminder surface.

    The one task renderer behind :func:`render_owed_entry`, and so behind
    every consumer — the durable obligations reminder row, the
    quiescence-attempt nudge and the ``list_obligations`` tool. None of
    them can prove the original ask is GONE (the reminder row is written
    only when the ask has left the window, but it then sits in the
    transcript for as long as the window keeps it, past any later
    re-read of the original), so none of them may order a refusal.

    * ``summary is None`` -> the loud unavailable marker. Nothing was
      ever persisted on the frame, so there is no task to point AT; #2080's
      fail-loud property is correct here and is preserved verbatim.
    * within :data:`_TASK_MAX` -> verbatim, byte-for-byte.
    * oversized -> a NEUTRAL bounded pointer (:data:`_REMINDER_ABRIDGED`) with no
      refusal imperative and no content prefix.

    The invariant: **a surface that cannot establish the original is GONE must not
    order a refusal.** These surfaces are reminders about a still-open request, so
    the honest render is "this line is abridged; the request message has the whole
    thing" — never "return an error".
    """
    if summary is None:
        return _TASK_UNAVAILABLE
    if len(summary) > _TASK_MAX:
        return _REMINDER_ABRIDGED.format(chars=len(summary), cap=_TASK_MAX)
    return summary


def _render_schema(output_schema: dict[str, Any] | None) -> str | None:
    """A bounded, single-line preview of an obligation's ``output_schema`` contract
    (#1522), or ``None`` when the request demands no schema.

    JSON-serialises the schema (compact separators, sorted keys for stability) and
    **elides** it to :data:`_SCHEMA_MAX` chars + an ellipsis. A large persisted
    schema can therefore
    NEVER inflate the rendered row past a fixed per-entry bound, which is what
    keeps :func:`max_obligations_reminder_local` a correct upper bound (no
    ``read_windowed_events`` budget overflow). Newlines are flattened so the
    contract stays a single render line.
    """
    if not output_schema:
        return None
    text = json.dumps(output_schema, separators=(",", ":"), sort_keys=True, default=str)
    text = text.replace("\n", " ")
    if len(text) > _SCHEMA_MAX:
        text = text[:_SCHEMA_MAX] + "…"
    return text


def render_owed_entry(obligation: Obligation, *, session_id: str) -> dict[str, Any]:
    """The shared per-obligation owed-read-model entry (#1522) — the ONE place the
    "outstanding obligation + its contract" projection is formatted.

    Both contract-bearing consumers feed from this:

    * the **quiescence-attempt surfacing** (#1514, folded here) — "you're trying to
      stop, here is what you owe and in what format" — joins these entries into the
      nudge content via :func:`render_owed_listing`; and
    * the **``list_obligations`` PULL tool** — returns these entries directly as its
      JSON result rows.

    Each entry carries ``request_id``, ``caller_kind`` (the trusted frame kind),
    ``origin`` (``api``/``session``/``run`` plus ``self`` for a #1414 self-goal),
    the task in ``summary``, the absolute ``opened_at`` (ISO-8601 UTC — never a
    relative age: the durable reminder row built from this render must be
    byte-stable across wall-clock time, or its change-gate would churn every
    minute), and the **bounded** ``output_schema`` contract (elided to
    :data:`_SCHEMA_MAX`; ``None`` when the request demands no schema). The schema
    bound is what lets the render stay within its reserved upper bound.

    ``summary`` renders through :func:`_reminder_content` (#2221 round 2):
    every consumer of this projection is a REMINDER surface — the durable
    reminder row and the quiescence nudge both persist their render to the
    event log, and none can establish that the original ask has been
    evicted at the time the model reads it. An oversized task therefore
    renders as a neutral bounded pointer with no refusal imperative; only a
    genuinely absent ``summary`` (nothing on the frame) still fails loud.
    """
    return {
        "request_id": obligation.request_id,
        "caller_kind": obligation.caller_kind or "",
        "origin": _origin_label(obligation, session_id=session_id),
        # #2221 round 2: a refuse-order from a persisted reminder about a
        # STILL-OPEN request re-parks the very sessions #2221 unparks.
        "summary": _reminder_content(obligation.summary),
        "opened_at": obligation.opened_at.astimezone(UTC).isoformat(timespec="seconds"),
        "output_schema": _render_schema(obligation.output_schema),
    }


def _owed_listing_line(entry: dict[str, Any]) -> str:
    """One human-readable line built from a :func:`render_owed_entry` row —
    ``request_id``, ``[origin]``, optional quoted summary, the absolute
    ``opened_at``, and (when present) the bounded ``output_schema`` contract."""
    summary = entry["summary"]
    summary_clause = f' "{summary}"' if summary else ""
    line = (
        f"• {entry['request_id']} [{entry['origin']}]{summary_clause} (opened {entry['opened_at']})"
    )
    schema = entry["output_schema"]
    if schema:
        line += f"\n    expected output_schema: {schema}"
    return line


def render_owed_listing(
    obligations: list[Obligation],
    *,
    session_id: str,
    header: str,
) -> str:
    """The shared **contract-bearing** owed render (#1522) used by the
    quiescence-attempt surfacing (consumer (a), folding #1514).

    A header line, then one entry per open obligation **oldest-first** (the caller
    fetches them ``ORDER BY req.seq ASC``) drawn from :func:`render_owed_entry`:
    each line shows ``request_id``, ``[origin]`` (incl. ``self``), task content,
    age, **and the bounded ``output_schema`` contract** — the format the session
    must produce to answer. Capped at :data:`MAX_RENDERED_OBLIGATIONS` entries +
    a ``+K more`` marker so the rendered size stays bounded regardless of count;
    each schema is :data:`_SCHEMA_MAX`-elided so a large contract can't blow the
    budget either.
    """
    lines = [header]
    rendered = obligations[:MAX_RENDERED_OBLIGATIONS]
    for ob in rendered:
        lines.append(_owed_listing_line(render_owed_entry(ob, session_id=session_id)))
    remaining = len(obligations) - len(rendered)
    if remaining > 0:
        lines.append(f"…(+{remaining} more)")
    return "\n".join(lines)


# The durable obligations reminder written when the LAST open obligation is
# answered while a listing is still in the window — so the stale listing is
# visibly superseded once, in the transcript, rather than lingering as the last
# word on what the session owes. A fixed string: written at most once per
# non-empty→empty transition.
OBLIGATIONS_EMPTY_CONTENT = (
    "━━━ Open obligations ━━━\n(none — every request this session owed has been answered)"
)

# Upper bound (local approx_tokens units) reserved in the window budget for the
# empty one-liner above, priced with the adjacent-user separator pre-pay. Paid
# unconditionally on every session, like the omission-marker reserve: whether
# a listing is in the window — the only thing that decides if the one-liner is
# written — is unknowable at prelude time. ``TestObligationsEmptyReserve`` pins
# the render under this bound.
OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL = 64


def render_obligations_reminder(obligations: list[Obligation], *, session_id: str) -> str:
    """The durable obligations reminder: the shared contract-bearing owed render
    under the tail header. Age-free by construction (``opened_at`` is absolute),
    so the SAME open set renders to the SAME bytes in every build — the
    property the reminder's change-gate and the prompt-prefix cache both need.
    Never called on an empty set (the empty transition writes
    :data:`OBLIGATIONS_EMPTY_CONTENT`)."""
    return render_owed_listing(obligations, session_id=session_id, header=_HEADER)


def max_obligations_reminder_local(obligations: list[Obligation]) -> int:
    """Worst-case local-token cost of :func:`render_obligations_reminder` for
    THIS step, computed at windowing time from the already-fetched open set (the
    real count, capped at :data:`MAX_RENDERED_OBLIGATIONS` + the ``+K more``
    marker; each task through :func:`_reminder_content`'s bound; each schema
    :data:`_SCHEMA_MAX`-elided). Priced with the adjacent-user separator, like
    every user row. Returns 0 on an empty set — the empty one-liner has its own
    unconditional reserve, :data:`OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL`."""
    if not obligations:
        return 0
    from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
    from aios.harness.tokens import approx_tokens

    # session_id="" keeps the origin label bare ("self" never widens the bound
    # vs. the literal caller_kind).
    return approx_tokens(
        [
            {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
            {"role": "user", "content": render_obligations_reminder(obligations, session_id="")},
        ]
    )
