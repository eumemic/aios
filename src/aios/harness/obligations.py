"""Obligation formatting helpers.

An open **awaited** request edge (#1123 ``request_opened`` minus
``request_response``, ``awaited=true``) is an obligation the session must answer
with ``return``/``error``.

#1413 introduced an always-on, per-step tail-injected block that rendered every
open obligation on EVERY step. #1514 **removed** that per-step injection: an
outstanding obligation is only *decision-relevant* when the agent tries to stop,
so surfacing it on every step was a continuous context-token tax. The
outstanding-task list AND each task's acceptance contract (``output_schema``) are
now surfaced ONLY at the quiescence attempt, in the quiescence-guard nudge
(:func:`aios.services.sessions.append_assistant_and_guard_quiescence`). No
per-step block is rendered anywhere.

What remains here are the small obligation **formatting helpers** still used by
the goal-management tool surface (``list_goals`` renders a terse ``age``) — the
render-time block builders and their windowing token-bound are gone with the
per-step injection they served.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aios.models.sessions import Obligation

# Summary truncation length; matches the 60-char preview the channels tail uses
# and the store-side ``_obligation_summary`` budget.
_SUMMARY_MAX = 60


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
    """A terse ``<age>`` string (``3s`` / ``5m`` / ``2h`` / ``4d``).

    Coarse-grained on purpose — a reminder, not a stopwatch — and never negative
    (clamped at 0). Used by ``list_goals`` to render an open self-goal's age.
    """
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
