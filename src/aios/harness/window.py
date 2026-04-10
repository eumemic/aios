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

The safety case: if a single event is so large that even a windowed sequence
still exceeds ``max_tokens * 1.5``, the harness will emit a context_overflow
lifecycle event and idle the session. That check is intentionally outside this
function — ``select_window`` is pure: same input, same output, no side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


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

    # Cheap path: everything fits.
    if total <= max_tokens:
        return list(events)

    # Snap math. We want the cutoff (`tokens_to_drop`) to be a step function
    # of `total`, advancing in chunks of (max - min). After the first snap
    # (overshoot in (0, max-min]), tokens_to_drop = (max - min). After the
    # second snap (overshoot in (max-min, 2*(max-min)]), it's 2*(max - min).
    overshoot = total - max_tokens
    chunk = max_tokens - min_tokens
    snaps = (overshoot + chunk - 1) // chunk  # ceil(overshoot / chunk)
    tokens_to_drop = snaps * chunk

    return [evt for evt, cum in cumulative if cum > tokens_to_drop]


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
