"""Token counting helpers.

Two estimators live here:

* :func:`approx_tokens` — fast, dependency-free ``len // 4`` estimate
  used by the cumulative-tokens column and context windowing.  Suitable
  for boundary decisions where ±10 % accuracy is fine.

* :func:`token_count_for_event` — precise LiteLLM-backed counter with
  a process-local LRU cache keyed by event id.

Both are importable from any layer (no internal aios dependencies
beyond the ``Event`` model type hint).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from aios.models.events import Event


# ─── cheap estimator (no external deps) ────────────────────────────────────


def approx_tokens(message: dict[str, Any]) -> int:
    """Rough token count for a chat-completions message dict.

    Counts characters in ``content`` plus ``tool_calls[].function.{name,
    arguments}`` and divides by 4.  Returns at least 1.

    This is the single source of truth for the ``cumulative_tokens``
    column on the events table and for the chunked-window boundary
    computation.  If the formula changes, run the backfill script to
    recompute stored values.
    """
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    total_chars = len(content)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        total_chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return max(1, total_chars // 4)


# ─── snap boundary math ───────────────────────────────────────────────────


def tokens_to_drop(total: int, *, window_min: int, window_max: int) -> int:
    """Compute how many tokens to drop from the front of a context window.

    Uses the chunked snap policy from :mod:`aios.harness.window`:
    drop in ``(max - min)``-sized chunks so the cutoff advances
    monotonically and prefix caching stays stable within a chunk.

    Returns 0 when the total fits within ``window_max``.
    """
    if total <= window_max:
        return 0
    overshoot = total - window_max
    chunk = window_max - window_min
    snaps = (overshoot + chunk - 1) // chunk  # ceil division
    return snaps * chunk


# ─── precise litellm counter ───────────────────────────────────────────────


@lru_cache(maxsize=10_000)
def _token_count_cached(event_id: str, payload_repr: str, model: str) -> int:
    """LRU-cached token counter keyed by ``(event_id, payload_repr, model)``.

    The ``payload_repr`` is a hash-stable string representation of the
    message; we include it in the cache key as a defensive check, even
    though events are immutable, so a corrupted re-emission can't poison
    the cache.
    """
    # litellm.token_counter accepts a list of messages and returns a token count.
    return int(litellm.token_counter(model=model, messages=[_payload_from_repr(payload_repr)]))


def _payload_from_repr(payload_repr: str) -> dict[str, object]:
    import json

    result: dict[str, object] = json.loads(payload_repr)
    return result


def token_count_for_event(event: Event, *, model: str) -> int:
    """Return the token count of a single message-kind event for ``model``.

    Non-message events return 0 — they don't appear in the chat-completions
    request the harness builds, so they don't consume context budget.
    """
    if event.kind != "message":
        return 0

    import json

    # Stable repr keyed by event id is enough; the payload is included as a
    # secondary key only to make the cache resilient to bugs in event id
    # uniqueness, not because we expect events to mutate.
    payload_repr = json.dumps(event.data, sort_keys=True)
    return _token_count_cached(event.id, payload_repr, model)
