"""Chunked stable-prefix context windowing.

This is the load-bearing algorithm aios uses to fit a session's event log into
a model's context window without invalidating the prompt prefix cache on every
turn.

The naive sliding window ("keep the last N events") shifts the cutoff on every
new event, which destroys prefix caching at every turn. Instead, we keep the
cutoff *monotonic non-decreasing* in the total token count and let it advance
in discrete chunks. Within a chunk, every new event is appended to a stable
prefix → cache hits. The cutoff jumps forward by ``(max - min)`` at "snap"
points, which happen rarely.

Concretely, given a per-agent ``min_tokens`` / ``max_tokens`` (defaults
50k / 150k):

* As long as the conversation fits in ``max_tokens``, return everything.
* When the total exceeds ``max_tokens``, drop the oldest events in
  ``(max - min)``-token chunks. The included size oscillates between just
  above ``min_tokens`` (right after a snap) and ``max_tokens`` (right before
  the next snap).
* Within a single chunk, the cutoff is constant — every new turn just appends
  to a stable prefix, so prompt prefix caching keeps hitting until the next
  snap.

Both bounds describe RETAINED CONVERSATION, but the per-request prelude
(system prompt + tool schemas + reserves) is asymmetric between them, and
the asymmetry is load-bearing (issue #2289):

* ``max_tokens`` bounds the *sent prompt*, so the prelude's cost must be
  subtracted from it — the retained slate gets whatever is left over.
* ``min_tokens`` bounds the *retained history* only. Subtracting the
  prelude from it too would reinterpret the floor as a floor on the whole
  prompt, which a fat tool prelude satisfies by itself with zero history —
  the incident that produced #2289 (121 tool schemas ≈ 99k effective
  overhead against a 50k/150k band drove the floor to 0, so every snap
  emptied the window to a single event and the agent was left talking to
  its own "history has scrolled out of view" notice).

The floor is instead *clamped* to a fraction of the events budget, so an
unaffordable band degrades to "less history than asked for" rather than
"no history" — see ``_WINDOW_FLOOR_MAX_FRACTION`` in
:mod:`aios.db.queries.events` and :class:`WindowFloor` below.

Note: a ``context_overflow`` safety check (idle the session when
windowed content still exceeds ``max_tokens * 1.5``) is planned but
not yet implemented.

The live windowing path is the SQL ``read_windowed_events``
(:mod:`aios.db.queries.events`); this module now only carries the
window's result shape (``WindowedEvents`` / ``WindowOmission``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from aios.models.events import Event


@dataclass(frozen=True, slots=True)
class WindowOmission:
    """Facts about the transcript a window omits (issue #738) — the
    inputs to the head omission marker.

    ``began_at`` is the ``created_at`` of the session's first message
    event; ``omitted_messages`` counts omitted user+assistant events
    only (tool results would dominate tool-heavy sessions without
    conveying conversational depth).

    Cache-stability rationale (canonical home — other sites reference
    this): both fields are pure functions of the drop boundary over the
    immutable log, so the marker rendered from them is byte-identical
    within a snap chunk and re-renders exactly when the boundary moves —
    a snap, when the head changes anyway.  Producer:
    :func:`~aios.db.queries.read_windowed_events`; consumer:
    :func:`~aios.harness.context.build_messages`.
    """

    began_at: datetime
    omitted_messages: int


@dataclass(frozen=True, slots=True)
class WindowFloor:
    """How the retained-history floor resolved for one windowed read (#2289).

    ``configured`` is the ``window_min`` the caller asked for; ``effective``
    is the floor the snap math actually ran with, after clamping it to a
    fraction of ``events_window_max`` (the events budget left once
    ``overhead_effective`` — the prelude's cost in the same effective space —
    is subtracted from ``window_max``).

    ``outcome`` is a kind rather than an ``is_clamped`` flag so the two
    regimes stay nameable at every call site: ``"honored"`` means the
    configured floor survived intact, ``"clamped"`` means the agent's
    configured band cannot deliver it against the current prelude and the
    read degraded deliberately.  ``"clamped"`` is an operator signal — the
    band or the tool surface wants attention — not an error: the prelude
    grows whenever an upstream MCP server adds tools, so failing hard here
    would brick every session in the fleet at wake time.

    Producer: :func:`~aios.db.queries.events.read_windowed_events`;
    consumer: :func:`~aios.harness.loop.run_session_step`, which stamps
    these onto the ``read_window_end`` span and warns on ``"clamped"``.
    """

    configured: int
    effective: int
    events_window_max: int
    overhead_effective: int
    outcome: Literal["honored", "clamped"]


@dataclass(frozen=True, slots=True)
class WindowedEvents:
    """A context window over a session log: the retained trailing slate,
    plus facts about what the drop boundary excluded.  ``omission`` is
    ``None`` when nothing is excluded — the whole transcript fits, or an
    oversized first event straddles the boundary.

    ``floor`` carries how the retained-history floor resolved (#2289).  It is
    ``None`` only on the pre-backfill fallback path, where no snap arithmetic
    runs at all, and for callers that construct the shape directly."""

    events: list[Event]
    omission: WindowOmission | None
    floor: WindowFloor | None = None
