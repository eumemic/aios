"""Per-account step admission gate — execution-time wake fairness (issue #418).

Problem: procrastinate's ``worker_concurrency`` is one global cap with no notion
of tenant. Account A bursting 100 session wakes fills every slot and account B's
single wake waits behind all of them.

Mechanism (the issue's Option A, enforced at EXECUTION time per the 2026-09-30
seat constraint — not an enqueue-time priority demotion, which #2485 showed both
freezes a stale priority at insert (F1) and races the count against the enqueue
(F2)):

* ``run_session_step`` calls :meth:`AccountStepGate.try_admit` before it emits
  ``step_start``. Admission is **non-blocking**: a blocking ``async with
  semaphore`` would keep the procrastinate job slot occupied while waiting, so
  a bursting account would fill every slot with running + blocked handlers and
  fairness would never materialise.
* A refused wake is re-deferred with a short delay and the handler returns,
  freeing the procrastinate slot at once. Because the re-deferred job is
  *scheduled in the future*, procrastinate's fetch skips it until the backoff
  elapses — the bursting account's surplus jobs quiesce instead of
  busy-looping, and the free slots go to other accounts' ready jobs. Each
  surplus job bounces at most once per backoff interval, which bounds the
  ``wake_deferred`` write amplification.
* **No starvation within an account** (the #2485 F1 property): the first refusal
  of a session records it in a per-account FIFO of waiters. While waiters
  exist, a free slot is reserved for the oldest ``free`` waiters — a newly
  submitted wake from the same account cannot jump ahead of a session that has
  been waiting longer, no matter how fast fresh wakes arrive. A waiter that
  never comes back (session archived, job lost) lapses after
  ``stale_after_seconds`` so it cannot wedge the account.
* Priority tiering is untouched: the gate never edits procrastinate
  ``priority``. A re-defer goes through ``defer_wake``, which re-derives the
  foreground/background tier from lineage, so a foreground wake stays
  foreground and fairness never demotes a human's message across tiers.

Locus of the meter (process-local vs Postgres): the gate is process-local. That
is exact under the ratified deployment shape — one worker per database
(advisory-lock singleton in ``harness/worker.py``, shard-by-DB per #921) — so
the per-account cap is a per-database cap. If several workers ever share one
database, each enforces the cap independently and the effective per-account cap
becomes ``N * account_step_concurrency_max``; fairness weakens but nothing
deadlocks. The migration path is local: replace this class's body with a
Postgres-backed counter and leave the call site in ``loop.py`` alone. No schema
change is involved.

Memory: an account's entry is dropped as soon as it has no running steps and no
waiters, so the map only holds currently-busy accounts.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _AccountState:
    active: int = 0
    # session_id -> last-seen monotonic time, insertion-ordered by FIRST refusal.
    waiters: OrderedDict[str, float] = field(default_factory=OrderedDict)


class AccountStepGate:
    """Non-blocking, FIFO-fair, per-account cap on concurrently running steps."""

    def __init__(
        self,
        max_per_account: int,
        *,
        stale_after_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_per_account < 1:
            raise ValueError("max_per_account must be >= 1")
        self.max_per_account = max_per_account
        self.stale_after_seconds = stale_after_seconds
        self._clock = clock
        self._accounts: dict[str, _AccountState] = {}

    def try_admit(self, account_id: str, session_id: str) -> bool:
        """Admit ``session_id`` to run a step now, or record it as a waiter.

        Returns ``True`` when a slot was taken — the caller MUST call
        :meth:`release` exactly once when the step finishes (on every path).
        Returns ``False`` when the account is at its cap or the free slots are
        reserved for longer-waiting sessions of the same account.
        """
        now = self._clock()
        state = self._accounts.setdefault(account_id, _AccountState())
        self._expire_stale(state, now)
        free = self.max_per_account - state.active
        if free > 0:
            if session_id in state.waiters:
                ahead = list(state.waiters).index(session_id)
            else:
                ahead = len(state.waiters)
            if ahead < free:
                state.waiters.pop(session_id, None)
                state.active += 1
                return True
        # Refused: keep (or take) a place in line; refresh liveness only.
        state.waiters[session_id] = now
        return False

    def release(self, account_id: str) -> None:
        state = self._accounts.get(account_id)
        if state is None or state.active <= 0:
            return
        state.active -= 1
        self._maybe_forget(account_id, state)

    def active(self, account_id: str) -> int:
        state = self._accounts.get(account_id)
        return state.active if state is not None else 0

    def waiting(self, account_id: str) -> list[str]:
        state = self._accounts.get(account_id)
        return list(state.waiters) if state is not None else []

    def tracked_accounts(self) -> int:
        return len(self._accounts)

    def _expire_stale(self, state: _AccountState, now: float) -> None:
        stale = [s for s, seen in state.waiters.items() if now - seen > self.stale_after_seconds]
        for s in stale:
            del state.waiters[s]

    def _maybe_forget(self, account_id: str, state: _AccountState) -> None:
        self._expire_stale(state, self._clock())
        if state.active == 0 and not state.waiters:
            self._accounts.pop(account_id, None)
