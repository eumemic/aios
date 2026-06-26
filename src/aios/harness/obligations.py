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

# Summary truncation length; matches the 60-char preview the channels tail uses
# and the store-side ``_obligation_summary`` budget.
_SUMMARY_MAX = 60

# Max chars of a rendered ``output_schema`` contract (#1522). Analogous to the
# 60-char summary cap, but wider because a schema is a structural contract (it
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


def _truncate_summary(summary: str | None) -> str:
    if not summary:
        return ""
    s = summary.replace("\n", " ").strip()
    if len(s) > _SUMMARY_MAX:
        s = s[:_SUMMARY_MAX] + "…"
    return s


def _render_schema(output_schema: dict[str, Any] | None) -> str | None:
    """A bounded, single-line preview of an obligation's ``output_schema`` contract
    (#1522), or ``None`` when the request demands no schema.

    JSON-serialises the schema (compact separators, sorted keys for stability) and
    **elides** it to :data:`_SCHEMA_MAX` chars + an ellipsis — the schema-side
    analogue of the 60-char summary cap. A large persisted schema can therefore
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


def _obligation_line(obligation: Obligation, *, session_id: str, now: datetime) -> str:
    """One render line for an obligation, oldest-first ordering applied by caller.

    Shape: ``• <request_id> [origin] "<summary>" (open <age>)`` — the literal
    ``request_id`` first (copy-pasteable; the id the model echoes to
    ``return``/``error``), then origin, an optional quoted summary, and age.
    """
    origin = _origin_label(obligation, session_id=session_id)
    summary = _truncate_summary(obligation.summary)
    summary_clause = f' "{summary}"' if summary else ""
    age = _format_age(obligation.opened_at, now)
    return f"• {obligation.request_id} [{origin}]{summary_clause} (open {age})"


def build_obligations_tail_block(
    obligations: list[Obligation],
    *,
    session_id: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Ephemeral per-step listing of every open awaited obligation (#1413).

    Clone of :func:`~aios.harness.channels.build_channels_tail_block` in spirit:
    a ``{role:"user"}`` dict appended after :func:`build_messages` so per-step
    mutation never busts the prompt-prefix cache (render-only tail blocks never
    enter ``cumulative_tokens``, so they never rewrite an earlier message).

    Header line, then one line per obligation **oldest-first** (the caller already
    fetches them ``ORDER BY req.seq ASC``): the literal ``request_id``, an
    ``[origin]`` label (``api``|``session``|``run``, plus ``self`` for a #1414
    self-goal), a ``<=60``-char quoted summary, and ``(open <age>)``. The block is
    capped at :data:`MAX_RENDERED_OBLIGATIONS` lines + a ``+K more`` marker so the
    reserved tail budget stays bounded regardless of obligation count.

    Returns ``None`` on an empty set (zero tail, zero tokens).
    """
    if not obligations:
        return None
    if now is None:
        now = datetime.now(UTC)
    lines = [_HEADER]
    rendered = obligations[:MAX_RENDERED_OBLIGATIONS]
    for ob in rendered:
        lines.append(_obligation_line(ob, session_id=session_id, now=now))
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
    a ``<=60``-char ``summary``, a terse ``age``, and the **bounded**
    ``output_schema`` contract (elided to :data:`_SCHEMA_MAX`; ``None`` when the
    request demands no schema). The schema bound is what lets the surfacing render
    stay within :func:`max_obligations_block_local`'s upper bound.
    """
    return {
        "request_id": obligation.request_id,
        "caller_kind": obligation.caller_kind or "",
        "origin": _origin_label(obligation, session_id=session_id),
        "summary": _truncate_summary(obligation.summary),
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
    each line shows ``request_id``, ``[origin]`` (incl. ``self``), quoted summary,
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
    + the ``+K more`` marker line) and each real summary (re-truncated to the
    render width). Strictly tighter than a synthetic max; the produced tail at
    send time is guaranteed ≤ this bound, so reserving it never overshoots
    ``window_max``.

    Returns 0 on an empty set (the block is ``None`` and nothing is appended).
    """
    if not obligations:
        return 0
    from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
    from aios.harness.tokens import approx_tokens

    # Render with a fixed ``now`` so the age clause has a stable (worst-case-ish)
    # width — ``4d`` etc. are all <= a handful of chars; the count/summary
    # dominate the bound. session_id="" keeps the origin label bare ("self" never
    # widens the bound vs. the literal caller_kind).
    block = build_obligations_tail_block(
        obligations, session_id="", now=datetime(1970, 1, 1, tzinfo=UTC)
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
