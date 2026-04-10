"""Token counting helper.

Wraps :func:`litellm.token_counter` with a process-local LRU cache keyed by
event id. Events are immutable, so the cache entries are permanent for the
lifetime of the process. The harness re-counts only events it hasn't seen
before, so the per-turn cost stays O(new events since last turn) instead of
O(full session history).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import litellm

if TYPE_CHECKING:
    from aios.models.events import Event


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
