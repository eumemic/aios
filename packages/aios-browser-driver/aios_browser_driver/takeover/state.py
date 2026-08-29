"""The takeover admission gate and the standing-takeover record.

The gate is the barrier between agent actions and a human takeover. Its
contract is the delicate part (jarbot#106 §5.6, red-team folds):

* ``admit`` is SYNCHRONOUS — a closed-check and a count-increment with no
  await between — so an action either fully enters before the gate closes or
  is refused. ``close_and_drain`` sets ``closed`` (synchronously) BEFORE
  awaiting the in-flight count to reach zero, so no action can slip in after
  the drain begins. Together these mean the epoch rotates only once every
  admitted action has finished — a human's input can never land under agent
  control, nor an agent action under the human's.
* On a drain that times out, the caller reopens the gate and FAILS the open;
  the epoch never rotates while actions are in flight.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aios_browser_driver.takeover.injector import InputInjector
    from aios_browser_driver.takeover.screencast import Screencast

# LRU depth of the closed-takeover handback cache — a redriven ``takeover_close``
# for an already-closed grant replays its cached handback instead of erroring.
_REPLAY_CACHE_MAX = 8


class AdmissionGate:
    """Serializes agent actions against a takeover. Open: actions ``admit``.
    Closing: new actions are refused and the in-flight ones drain to zero
    before the caller rotates the epoch."""

    def __init__(self) -> None:
        self._closed = False
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def closed(self) -> bool:
        return self._closed

    def admit(self) -> bool:
        """Synchronously admit one action, or refuse if the gate is closed.
        No await between the check and the increment — the drain cannot race
        an admission in."""
        if self._closed:
            return False
        self._active += 1
        self._idle.clear()
        return True

    def release(self) -> None:
        # Paired one-to-one with a successful admit() in the caller's finally,
        # so _active never goes negative.
        self._active -= 1
        if self._active == 0:
            self._idle.set()

    async def close_and_drain(self, drain_timeout_s: float) -> bool:
        """Close the gate and wait for in-flight actions to finish. Returns
        True if drained within ``drain_timeout_s``, False otherwise (the caller
        then reopens and fails the open)."""
        self._closed = True
        if self._active == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), drain_timeout_s)
            return True
        except TimeoutError:
            return False

    def reopen(self) -> None:
        self._closed = False


@dataclass
class Takeover:
    """The one standing takeover. ``opened_at``/``last_input`` are wall-clock
    seconds; the watchdog folds in the heartbeat marker's mtime for liveness."""

    grant_id: str
    session_id: str
    epoch: int
    opened_at: float
    screencast: Screencast
    injector: InputInjector
    target: dict[str, Any]
    signed_in_at_open: list[str] = field(default_factory=list)
    input_task: asyncio.Task[None] | None = None
    last_input: float = 0.0
    last_seq: int = 0  # highest input-spool seq applied (per this takeover)

    def liveness(self, marker_mtime: float) -> float:
        return max(self.opened_at, self.last_input, marker_mtime)

    def is_unclaimed(self, marker_mtime: float) -> bool:
        """No input line AND no heartbeat-marker touch since it opened — the
        ack was produced but no viewer ever attached."""
        return self.last_input == 0.0 and marker_mtime <= self.opened_at


class ReplayCache:
    """Closed-grant → handback, bounded FIFO. Lets a redriven close of an
    already-finalized grant replay the real handback. Grant ids are single-use
    and a redrive lands right after its own close — well inside the window — so
    insertion order suffices; no access-recency bookkeeping is needed."""

    def __init__(self, maxlen: int = _REPLAY_CACHE_MAX) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._maxlen = maxlen

    def put(self, grant_id: str, handback: dict[str, Any]) -> None:
        self._store[grant_id] = handback
        while len(self._store) > self._maxlen:
            del self._store[next(iter(self._store))]  # evict oldest insertion

    def get(self, grant_id: str) -> dict[str, Any] | None:
        return self._store.get(grant_id)


def now() -> float:
    return time.time()
