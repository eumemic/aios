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
class WindowedEvents:
    """A context window over a session log: the retained trailing slate,
    plus facts about what the drop boundary excluded.  ``omission`` is
    ``None`` when nothing is excluded — the whole transcript fits, or an
    oversized first event straddles the boundary."""

    events: list[Event]
    omission: WindowOmission | None
