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
  been waiting longer, no matter how fast fresh wakes arrive.
* **A waiter never loses its place to the clock.** Wall-clock time since a
  waiter was last fetched is NOT evidence it is gone: while every worker slot is
  busy (exactly the burst condition this gate exists for, and steps routinely
  outlast any short window — ``HARNESS_STEP_TIMEOUT_S`` is minutes) nothing is
  fetched and no live waiter can re-arrive. An earlier revision expired waiters
  after ``stale_after_seconds`` of silence; under saturation that erased the
  whole FIFO and let fresh wakes overtake the originals indefinitely. Now
  ``stale_after_seconds`` only marks a waiter as *unconfirmed*; a waiter is
  dropped solely when DURABLE state says it is gone — the session is archived
  or it no longer has a queued/running ``wake_session`` job
  (:meth:`AccountStepGate.unconfirmed_waiters_ahead` +
  :meth:`AccountStepGate.reconcile_waiters`, driven by ``run_session_step``).
  The check runs only when a free slot is being withheld for unconfirmed
  waiters, and a confirmed-live waiter is not re-checked for another window, so
  the query rate stays bounded.
* **The FIFO is tier-aware** (#2593 F1). Procrastinate fetches ``(priority
  DESC, id ASC)``, and ``defer_wake`` puts a workflow fan-out's wakes in the
  background tier (``-10``) under a user's foreground wakes (``0``). The gate
  orders waiters the same way: a wake's place in line is behind every waiter of
  a HIGHER tier and behind OLDER waiters of its OWN tier only. Lower-tier
  waiters never hold a free slot away from a higher-tier wake, so a single
  account's background fan-out cannot delay that account's own foreground
  message. FIFO no-starvation still holds within each tier. A lower tier can
  wait behind a continuous higher tier, which is the policy procrastinate's
  fetch already applies on master. The gate never edits procrastinate
  ``priority``: a re-defer goes through ``defer_wake``, which re-derives the
  tier from lineage, so fairness never moves a wake to another tier.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

#: procrastinate's default job priority, which ``defer_wake`` uses for foreground.
DEFAULT_PRIORITY = 0


@dataclass
class _Waiter:
    # Last-CONFIRMED-live monotonic time (re-arrival or durable check). Never an
    # expiry deadline.
    seen: float
    # Procrastinate priority tier of the waiter's latest wake (higher runs first).
    priority: int


@dataclass
class _AccountState:
    active: int = 0
    # session_id -> waiter, insertion-ordered by FIRST refusal (FIFO within a tier).
    waiters: OrderedDict[str, _Waiter] = field(default_factory=OrderedDict)


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

    def known_priority(self, account_id: str, session_id: str) -> int | None:
        """The tier recorded for ``session_id`` as a waiter, or ``None``.

        A waiter's tier is the priority of the wake job its last refusal
        enqueued (see :meth:`set_waiter_priority`), i.e. exactly the tier
        procrastinate fetched the re-deferred job at."""
        state = self._accounts.get(account_id)
        waiter = state.waiters.get(session_id) if state is not None else None
        return waiter.priority if waiter is not None else None

    def needs_priority(self, account_id: str, session_id: str) -> bool:
        """``True`` when admitting ``session_id`` depends on a tier the gate does
        not know yet: the account has free capacity, there are waiters to order
        against, and ``session_id`` is not one of them. Elsewhere the caller can
        skip the lookup: with no waiters the wake is admitted whatever its tier;
        at the cap it is refused whatever its tier, and the refusal records the
        tier the re-deferred job was really enqueued at
        (:meth:`set_waiter_priority`); a known waiter's tier is already recorded.
        """
        state = self._accounts.get(account_id)
        if state is None or not state.waiters or session_id in state.waiters:
            return False
        return state.active < self.max_per_account

    def set_waiter_priority(self, account_id: str, session_id: str, priority: int) -> None:
        """Record the tier a refused wake was actually re-enqueued at."""
        state = self._accounts.get(account_id)
        waiter = state.waiters.get(session_id) if state is not None else None
        if waiter is not None:
            waiter.priority = priority

    def try_admit(
        self, account_id: str, session_id: str, *, priority: int = DEFAULT_PRIORITY
    ) -> bool:
        """Admit ``session_id`` to run a step now, or record it as a waiter.

        ``priority`` is the wake's procrastinate priority tier (higher runs
        first). Returns ``True`` when a slot was taken — the caller MUST call
        :meth:`release` exactly once when the step finishes (on every path).
        Returns ``False`` when the account is at its cap or the free slots are
        reserved for sessions ahead of this one: higher-tier waiters, or older
        waiters of the same tier. Lower-tier waiters never reserve a slot
        against it.
        """
        now = self._clock()
        state = self._accounts.setdefault(account_id, _AccountState())
        free = self.max_per_account - state.active
        if free > 0 and len(self._ahead(state, session_id, priority)) < free:
            state.waiters.pop(session_id, None)
            state.active += 1
            return True
        # Refused: keep (or take) a place in line; refresh liveness and tier. The
        # FIFO position is never lost to elapsed time (see module docstring).
        waiter = state.waiters.get(session_id)
        if waiter is None:
            state.waiters[session_id] = _Waiter(seen=now, priority=priority)
        else:
            waiter.seen = now
            waiter.priority = priority
        return False

    def unconfirmed_waiters_ahead(
        self, account_id: str, session_id: str, *, priority: int = DEFAULT_PRIORITY
    ) -> list[str]:
        """Waiters holding a free slot away from ``session_id`` whose liveness has
        not been confirmed within ``stale_after_seconds``.

        Empty unless the account has free capacity that is being withheld from
        ``session_id`` for waiters ahead of it — the only situation where a gone
        waiter costs anything. The caller checks these against durable state and
        reports back via :meth:`reconcile_waiters`; they are NOT dropped here.
        """
        state = self._accounts.get(account_id)
        if state is None:
            return []
        free = self.max_per_account - state.active
        if free <= 0:
            return []
        now = self._clock()
        return [
            sid
            for sid in self._ahead(state, session_id, priority)
            if now - state.waiters[sid].seen > self.stale_after_seconds
        ]

    def reconcile_waiters(
        self, account_id: str, *, checked: Iterable[str], live: Iterable[str]
    ) -> None:
        """Apply a durable liveness check: of ``checked``, those in ``live`` keep
        their place (and are marked confirmed now); the rest are gone and drop."""
        state = self._accounts.get(account_id)
        if state is None:
            return
        now = self._clock()
        live_set = set(live)
        for sid in checked:
            if sid not in state.waiters:
                continue
            if sid in live_set:
                state.waiters[sid].seen = now
            else:
                del state.waiters[sid]
        self._maybe_forget(account_id, state)

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

    @staticmethod
    def _ahead(state: _AccountState, session_id: str, priority: int) -> list[str]:
        """Waiters that procrastinate's ``(priority DESC, id ASC)`` order would run
        before a wake of ``session_id`` at ``priority``: every waiter of a higher
        tier, plus same-tier waiters that started waiting earlier. A newcomer is
        behind every same-tier waiter."""
        out: list[str] = []
        before_self = True
        for sid, waiter in state.waiters.items():
            if sid == session_id:
                before_self = False
                continue
            if waiter.priority > priority or (waiter.priority == priority and before_self):
                out.append(sid)
        return out

    def _maybe_forget(self, account_id: str, state: _AccountState) -> None:
        if state.active == 0 and not state.waiters:
            self._accounts.pop(account_id, None)
