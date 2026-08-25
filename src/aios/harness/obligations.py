"""Tail-injected obligations block (#1413).

An open **awaited** request edge (#1123 ``request_opened`` minus
``request_response``, ``awaited=true``) is an obligation the session must answer
with ``return``/``error``. Its only model-visible surface used to be a
render-time marker prepended to the *original* user message carrying
``metadata.request.request_id`` — which context windowing **erases** the moment
the conversation scrolls past it, exactly when the session has drifted far from
the ask and most needs the reminder.

This module renders the always-on replacement: an **ephemeral, rebuilt-each-step,
last-user-role** block listing every open obligation, appended after
:func:`~aios.harness.context.build_messages` so per-step mutations never bust the
prompt-prefix cache (the load-bearing property mirrored from
:func:`~aios.harness.channels.build_channels_tail_block`). Obligations are a
**distinct plane** from channels — orthogonal to the request edge — so they live
in their own module rather than co-located with the channels tail.

The render is driven by :class:`~aios.models.sessions.Obligation` rows fetched
via :func:`aios.db.queries.sessions.get_open_obligations` (a full-log query, not
a slate-derived marker), so it survives windowing erasure of the original ask.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aios.harness.context import EPHEMERAL_TAIL_KEY

if TYPE_CHECKING:
    from aios.models.sessions import Obligation

# Max obligations rendered as full lines; beyond this a ``+K more`` marker
# collapses the tail (mirrors ``trace_max_nodes``). Keeps the reserved tail
# budget bounded REGARDLESS of obligation count — without a cap an unbounded
# count inflates the reserved budget until ``read_windowed_events`` raises
# ``ValueError`` (no budget for events) → step crash. Issue C adds the
# complementary per-session open-goal admission cap.
MAX_RENDERED_OBLIGATIONS = 10

# Requests up to this size remain verbatim in the always-on reminder. Beyond it
# the reminder cannot re-render the task in full without blowing the reserved tail
# budget (see ``max_obligations_block_local``), so it renders one of two things —
# and WHICH one turns on whether the original ask is still in the context window:
#
# * still in the window (#2221) -> a bounded preview + a pointer to the intact
#   original. The task is NOT lost; it is sitting earlier in this same prompt, so
#   an instruction to refuse it is simply false. Emitting one made every task over
#   this cap structurally unbuildable: the child read the whole task, then read a
#   trailing order to refuse it, and obeyed.
# * genuinely gone (evicted, or no summary on the frame) -> the loud refuse marker
#   (#2080). Here the task really is unrecoverable from context, and a
#   plausible-looking prefix is what let a sub-agent improvise cross-repo writes.
#
# The cap itself is load-bearing and unchanged; only the behaviour AT the cap
# depends on presence.
_TASK_MAX = 8192
_TASK_TRUNCATED = "[TASK TRUNCATED — return an error; do not act on or infer the missing task]"

# The genuinely-unavailable marker: NO content was ever persisted on the frame
# (pre-#1413), so there is nothing to abridge and nothing to point at. Shared by
# BOTH renderers — the tail block and the reminder surfaces — because #2080's
# fail-loud property is correct on every surface: an absent task cannot be
# recovered from anywhere, so improvising one is the exact hazard. Hoisted to a
# module constant so the two paths can never drift on this wording.
_TASK_UNAVAILABLE = "[TASK CONTENT UNAVAILABLE — return an error; do not infer the task]"

# Chars of an oversized-but-present task echoed as a bounded preview beside the
# pointer. Strictly smaller than :data:`_TASK_MAX`, so the abridged render is
# never fatter than the at-cap VERBATIM render the reserved budget already
# permits — the fence stays where #2080/#2071 put it.
_TASK_PREVIEW = 2048

# What an oversized task renders as on a surface that CANNOT establish whether the
# original ask is still reachable (#2221 round 2): the quiescence nudge and the
# ``list_obligations`` projection. Both are REMINDERS about a request that is, by
# construction, still open — so ordering a refusal there is never justified:
#
# * the surface cannot prove the original is GONE, and
# * a refusal order on the nudge path is strictly WORSE than the tail-block bug
#   this issue opened for. The tail block is ephemeral and rebuilt each step; the
#   nudge is written to the event log as a DURABLE user message (up to
#   ``REQUEST_NUDGE_BUDGET`` times) and then sits in the window permanently. #2080's
#   own evidence is that a model obeys a trailing refuse-order, so emitting one here
#   re-parks exactly the oversized-payload sessions #2221 exists to unpark.
#
# Deliberately carries NO content prefix. On the tail block a preview is safe
# because presence is ESTABLISHED — the full task is intact earlier in the same
# prompt, so the preview is redundant signal, not a substitute. Here presence is
# unknown, and a plausible-looking prefix of an unrecoverable task is precisely
# #2080's improvisation hazard (a sub-agent read a prefix and invented the rest).
# A bare char-count pointer refuses BOTH failure modes: it never orders a refusal,
# and it offers nothing to improvise from. It also keeps the durable nudge event
# small regardless of task size.
_REMINDER_ABRIDGED = (
    "[TASK ABRIDGED IN THIS REMINDER — {chars} characters, over this reminder's "
    "{cap}-character budget. This is a reminder, not the task: the original request "
    "message carries it in full. Do not refuse on account of this line.]"
)

# Max chars of a rendered ``output_schema`` contract (#1522). Kept narrower
# than the task budget because a schema is a structural contract and usually
# legitimately needs a few keys/types to be useful) — still a HARD cap so a large
# persisted schema can't blow the reserved tail budget. A schema longer than this
# is JSON-serialised then elided to this width + an ellipsis. The bound is what
# keeps ``max_obligations_block_local`` a correct upper bound for the
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


def _format_age(opened_at: datetime, now: datetime) -> str:
    """A terse ``<age>`` string (``3s`` / ``5m`` / ``2h`` / ``4d``) for the
    ``(open <age>)`` clause. Coarse-grained on purpose — the block is a reminder,
    not a stopwatch — and never negative (clamped at 0)."""
    delta = now - opened_at
    secs = int(delta.total_seconds())
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def _request_content(summary: str | None, *, original_present: bool = False) -> str:
    """Render a verbatim task, an abridged one, or a loud marker.

    ``original_present`` is the caller's answer to "is the ORIGINAL request user
    message still in the rendered context window?" — knowable only from the
    post-windowing slate, so it is threaded in rather than guessed here.

    * ``summary is None`` -> the unavailable marker. No content exists on the
      frame (pre-#1413), so there is nothing to abridge (#2080 fail-loud).
    * within :data:`_TASK_MAX` -> verbatim, byte-for-byte.
    * oversized AND the original is still in the window -> a bounded preview plus
      a pointer to the intact original (#2221). The task is not lost, so a refuse
      instruction here is FALSE and — being the last user-role content the model
      reads — gets obeyed over the real task sitting earlier in the same prompt.
    * oversized AND the original is gone -> the refuse marker (#2080). Now the
      task really is unrecoverable, and a plausible prefix is the exact failure
      that let a sub-agent improvise cross-repo writes.

    Defaults to ``False`` so any caller that does NOT know the window state keeps
    today's conservative fail-loud behaviour; only a caller holding the slate can
    opt into the softer render.
    """
    if summary is None:
        return _TASK_UNAVAILABLE
    if len(summary) > _TASK_MAX:
        if not original_present:
            return f"{_TASK_TRUNCATED} (received {len(summary)} characters; limit {_TASK_MAX})"
        return (
            f"[TASK ABRIDGED IN THIS REMINDER — the full task is intact in the original "
            f"request message earlier in this context; {len(summary)} characters, reminder "
            f"budget {_TASK_MAX}. Do not refuse; read the original message above.]\n"
            f"{summary[:_TASK_PREVIEW]}"
        )
    return summary


def _reminder_content(summary: str | None) -> str:
    """Render a task for a **reminder surface that cannot establish presence**.

    The sibling of :func:`_request_content` for the two consumers of
    :func:`render_owed_entry` — the quiescence-attempt nudge and the
    ``list_obligations`` tool. Neither holds a post-windowing slate, so neither can
    answer "is the original ask still in the context?" the way
    :func:`~aios.harness.step_context.compose_step_context` can.

    This is a DISTINCT function rather than a presence flag on
    :func:`_request_content` on purpose. A boolean threaded through these callers
    could only ever be a *guess* at presence, and a guess that says "present" is the
    false-reassurance hazard #2080 exists to prevent. The surface class is a static
    fact about the caller, not a runtime claim about the window, so it is encoded
    in WHICH renderer the caller picks.

    * ``summary is None`` -> the loud unavailable marker, unchanged. Nothing was
      ever persisted on the frame, so there is no task to point AT; #2080's
      fail-loud property is correct here and is preserved verbatim.
    * within :data:`_TASK_MAX` -> verbatim, byte-for-byte (identical to the tail).
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
    NEVER inflate the rendered tail past a fixed per-entry bound, which is what
    keeps :func:`max_obligations_block_local` a correct upper bound (no
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


def _obligation_line(
    obligation: Obligation,
    *,
    session_id: str,
    now: datetime,
    present_request_ids: frozenset[str] = frozenset(),
) -> str:
    """One render line for an obligation, oldest-first ordering applied by caller.

    The literal ``request_id`` comes first (copy-pasteable; the id the model
    echoes to ``return``/``error``), followed by origin, age, and the task.
    An oversized task is abridged (preview + pointer) when its original ask is
    still in the window, and replaced by a loud marker when it is not — see
    :func:`_request_content`. A plausible prefix is never shown WITHOUT either the
    pointer or the marker.
    """
    origin = _origin_label(obligation, session_id=session_id)
    request_content = _request_content(
        obligation.summary,
        original_present=obligation.request_id in present_request_ids,
    )
    age = _format_age(obligation.opened_at, now)
    return f"• {obligation.request_id} [{origin}] (open {age}) verbatim task: {request_content}"


def build_obligations_tail_block(
    obligations: list[Obligation],
    *,
    session_id: str,
    now: datetime | None = None,
    present_request_ids: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Ephemeral per-step listing of every open awaited obligation (#1413).

    Clone of :func:`~aios.harness.channels.build_channels_tail_block` in spirit:
    a ``{role:"user"}`` dict appended after :func:`build_messages` so per-step
    mutation never busts the prompt-prefix cache (render-only tail blocks never
    enter ``cumulative_tokens``, so they never rewrite an earlier message).

    Header line, then one line per obligation **oldest-first** (the caller already
    fetches them ``ORDER BY req.seq ASC``): the literal ``request_id``, an
    ``[origin]`` label (``api``|``session``|``run``, plus ``self`` for a #1414
    self-goal), ``(open <age>)``, and the verbatim task (or, past the render
    budget, an abridged preview + pointer when the original ask is still in the
    window, else a loud marker). The block is
    capped at :data:`MAX_RENDERED_OBLIGATIONS` lines + a ``+K more`` marker so the
    reserved tail budget stays bounded regardless of obligation count.

    ``present_request_ids`` is the set of ``request_id``s whose ORIGINAL request
    user message survived windowing into this step's slate (#2221). The composer
    computes it from the post-windowing events; it is the ONLY input that can
    distinguish "oversized but recoverable from context" (abridge + point) from
    "oversized and genuinely gone" (fail loud). Empty by default, which keeps the
    conservative #2080 marker for any caller that cannot know.

    Returns ``None`` on an empty set (zero tail, zero tokens).
    """
    if not obligations:
        return None
    if now is None:
        now = datetime.now(UTC)
    lines = [_HEADER]
    rendered = obligations[:MAX_RENDERED_OBLIGATIONS]
    for ob in rendered:
        lines.append(
            _obligation_line(
                ob, session_id=session_id, now=now, present_request_ids=present_request_ids
            )
        )
    remaining = len(obligations) - len(rendered)
    if remaining > 0:
        lines.append(f"…(+{remaining} more)")
    return {"role": "user", "content": "\n".join(lines), EPHEMERAL_TAIL_KEY: True}


def render_owed_entry(obligation: Obligation, *, session_id: str, now: datetime) -> dict[str, Any]:
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
    the task in ``summary``, a terse ``age``, and the **bounded**
    ``output_schema`` contract (elided to :data:`_SCHEMA_MAX`; ``None`` when the
    request demands no schema). The schema bound is what lets the surfacing render
    stay within :func:`max_obligations_block_local`'s upper bound.

    ``summary`` renders through :func:`_reminder_content`, NOT
    :func:`_request_content` (#2221 round 2). Both consumers above are REMINDER
    surfaces: they run without a post-windowing slate, so neither can establish
    that the original ask has been evicted — and the nudge consumer persists its
    render as a durable user event. An oversized task therefore renders as a
    neutral bounded pointer with no refusal imperative; only a genuinely absent
    ``summary`` (nothing on the frame) still fails loud. The tail block keeps
    :func:`_request_content`, which CAN prove presence and so can safely show a
    preview.
    """
    return {
        "request_id": obligation.request_id,
        "caller_kind": obligation.caller_kind or "",
        "origin": _origin_label(obligation, session_id=session_id),
        # #2221 round 2: the REMINDER renderer, not ``_request_content``. Both
        # consumers of this projection (the quiescence nudge, ``list_obligations``)
        # are reminders about a STILL-OPEN request on a surface that cannot see the
        # post-windowing slate, so neither can establish that the original ask is
        # gone — and the nudge writes its render to the event log DURABLY. Emitting
        # a refuse-order from here re-parks the very sessions #2221 unparks, through
        # a more permanent door than the ephemeral tail block ever had.
        "summary": _reminder_content(obligation.summary),
        "age": _format_age(obligation.opened_at, now),
        "output_schema": _render_schema(obligation.output_schema),
    }


def _owed_listing_line(entry: dict[str, Any]) -> str:
    """One human-readable line for the quiescence-attempt surfacing, built from a
    :func:`render_owed_entry` row — ``request_id``, ``[origin]``, optional quoted
    summary, age, and (when present) the bounded ``output_schema`` contract."""
    summary = entry["summary"]
    summary_clause = f' "{summary}"' if summary else ""
    line = f"• {entry['request_id']} [{entry['origin']}]{summary_clause} (open {entry['age']})"
    schema = entry["output_schema"]
    if schema:
        line += f"\n    expected output_schema: {schema}"
    return line


def render_owed_listing(
    obligations: list[Obligation],
    *,
    session_id: str,
    header: str,
    now: datetime | None = None,
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
    if now is None:
        now = datetime.now(UTC)
    lines = [header]
    rendered = obligations[:MAX_RENDERED_OBLIGATIONS]
    for ob in rendered:
        lines.append(_owed_listing_line(render_owed_entry(ob, session_id=session_id, now=now)))
    remaining = len(obligations) - len(rendered)
    if remaining > 0:
        lines.append(f"…(+{remaining} more)")
    return "\n".join(lines)


def max_obligations_block_local(obligations: list[Obligation]) -> int:
    """Worst-case local-token cost of :func:`build_obligations_tail_block`.

    Called at windowing time. Unlike the channels tail (whose actual content is
    unknown pre-windowing, so it synthesizes a fattest-line bound), the obligation
    set is **already fetched** by ``compute_step_prelude``, so this bounds from the
    REAL obligations — the real count (capped at :data:`MAX_RENDERED_OBLIGATIONS`
    + the ``+K more`` marker line) and each rendered task (verbatim through
    :data:`_TASK_MAX`, then either abridged or marker-replaced). Strictly tighter
    than a synthetic max; the produced tail at
    send time is guaranteed ≤ this bound, so reserving it never overshoots
    ``window_max``.

    **Presence is unknowable here, so the bound assumes the FATTER branch.** This
    runs at windowing time — BEFORE the slate exists — so it cannot know which
    oversized tasks will render abridged (#2221: preview + pointer, ~
    :data:`_TASK_PREVIEW` chars) versus marker-replaced (#2080: a short fixed
    string). The abridged branch is strictly fatter, so the bound is computed with
    every request treated as present. Costing the marker branch instead would
    UNDER-RESERVE — measured 516 reserved against a 3240-token real render for 10
    oversized obligations — and an under-reserved tail is exactly the
    ``read_windowed_events`` budget overflow (→ step crash) the cap exists to
    prevent.

    Returns 0 on an empty set (the block is ``None`` and nothing is appended).
    """
    if not obligations:
        return 0
    from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
    from aios.harness.tokens import approx_tokens

    # Render with a fixed ``now`` so the age clause has a stable (worst-case-ish)
    # width — ``4d`` etc. are all <= a handful of chars; the count/summary
    # dominate the bound. session_id="" keeps the origin label bare ("self" never
    # widens the bound vs. the literal caller_kind). Every request is treated as
    # PRESENT so the bound covers the fatter abridged render (see docstring).
    block = build_obligations_tail_block(
        obligations,
        session_id="",
        now=datetime(1970, 1, 1, tzinfo=UTC),
        present_request_ids=frozenset(ob.request_id for ob in obligations),
    )
    if block is None:
        return 0
    # The tail is user-role and lands after the log's final message; when that
    # message is also user-role, ``merge_adjacent_user_messages`` concatenates
    # them. Reserving an assistant-separator's worth keeps the budget a
    # conservative upper bound either way (the proven channels-block path).
    return approx_tokens(
        [
            {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
            block,
        ]
    )
