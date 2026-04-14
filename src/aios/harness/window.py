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
not yet implemented.  ``select_window`` is pure: same input, same
output, no side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from aios.harness.tokens import tokens_to_drop as _tokens_to_drop


def select_window[T](
    events: Sequence[T],
    *,
    min_tokens: int,
    max_tokens: int,
    token_counter: Callable[[T], int],
) -> list[T]:
    """Select the trailing window of ``events`` that fits the chunked policy.

    Parameters
    ----------
    events:
        Chronologically-ordered sequence of message events.
    min_tokens:
        Lower bound for the included window size.
    max_tokens:
        Upper bound for the included window size.
    token_counter:
        Pure function returning the token count for a single event. The harness
        passes a memoized counter; tests can pass a trivial one.

    Returns
    -------
    list[T]
        A trailing slice of ``events`` whose cumulative token count is between
        ``min_tokens`` and ``max_tokens`` (inclusive of the upper bound,
        exclusive of zero on the lower side when total ≤ max_tokens).
    """
    if min_tokens < 1 or max_tokens < 1:
        raise ValueError("min_tokens and max_tokens must be positive")
    if min_tokens >= max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) must be strictly less than max_tokens ({max_tokens})"
        )

    if not events:
        return []

    # First pass: compute the cumulative-tokens-through-event-i array.
    cumulative: list[tuple[T, int]] = []
    running = 0
    for evt in events:
        running += token_counter(evt)
        cumulative.append((evt, running))

    total = running
    drop = _tokens_to_drop(total, window_min=min_tokens, window_max=max_tokens)
    if drop == 0:
        return list(events)
    return [evt for evt, cum in cumulative if cum > drop]


def cumulative_tokens[T](
    events: Sequence[T],
    *,
    token_counter: Callable[[T], int],
) -> int:
    """Sum the token counts of every event in ``events``.

    Convenience for callers that need the total (e.g., to apply the
    ``> max * 1.5`` safety check after windowing).
    """
    return sum(token_counter(e) for e in events)
