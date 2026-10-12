"""Degradation telemetry for snapshot chain accounting (#2538 N3).

The #2349/#2388 chain-cost triggers fail SAFE when ``docker history`` cannot be
parsed or the shared base cannot be inspected: the dead-history flatten is
disabled and the budget falls back to what is provable. Fail-safe is also
fail-SILENT — a fleet-wide unparseable history (a Docker upgrade that changes
the ``{{.Size}}`` rendering, say) would quietly turn the dead-history trigger
off everywhere, visible only as a stream of per-image warnings.

This module makes the degradation observable:

* a Prometheus counter ``aios_sandbox_chain_accounting_degraded_total{reason}``
  (when ``prometheus_client`` is importable) plus always-on in-process tallies;
* two alarm triggers, each logging an error-level
  ``sandbox.chain_accounting_degraded_alarm`` with ``alarm_event=True`` (the
  same alert shape as the worker watchdogs):

  - a CONSECUTIVE streak keyed per ``(reason, subject)`` — the subject is the
    image id (or base ref) whose probe degraded. Only a successful FRESH
    measurement of that same subject resets it. A snapshot probes several
    images (base, parent, committed), so a single process-global streak was
    reset by sibling successes and one image's persistent failure never
    alarmed (#2545 F1). Alarms once per streak.
  - a per-reason sliding WINDOW over the last :data:`WINDOW` fresh probe
    outcomes. A fleet-wide fault in which every NEW image degrades never
    builds a long per-image streak (each image is probed only a couple of
    times), but it dominates the window. Alarms when at least
    :data:`WINDOW_ALERT_DEGRADED` of the window degraded; re-arms only after
    the degraded share falls to :data:`WINDOW_REARM_DEGRADED` or below.

  Chain-CACHE hits are not measurements and must not be reported here: a
  cached base read proves nothing about whether the probe works today.

Corroboration only: nothing here gates snapshot behaviour, and a metrics-sink
failure must never propagate into the snapshot verb.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from typing import Literal

from aios.logging import get_logger

log = get_logger("aios.sandbox.snapshot_telemetry")

Reason = Literal["history_unparseable", "history_unavailable", "base_unmeasurable"]

# Consecutive degraded observations of ONE subject (with no fresh success of
# that subject in between) that raise the alarm. Low enough that a persistent
# fault alarms within a handful of snapshots; high enough that a single
# transient daemon hiccup does not.
ALERT_STREAK = 5

# Sliding window over fresh probe outcomes, per reason.
WINDOW = 20
WINDOW_ALERT_DEGRADED = 10
WINDOW_REARM_DEGRADED = 5

# Bound on tracked per-subject streaks (subjects are content-addressed ids).
_MAX_SUBJECTS = 1024

try:  # pragma: no cover - exercised only where prometheus_client is installed
    from prometheus_client import Counter as _PromCounter

    _DEGRADED = _PromCounter(
        "aios_sandbox_chain_accounting_degraded_total",
        "Snapshot chain-accounting probes that degraded to a fail-safe fallback.",
        ["reason"],
    )
except Exception:  # pragma: no cover
    _DEGRADED = None

_lock = threading.Lock()
_totals: dict[str, int] = {}
_streaks: dict[tuple[str, str], int] = {}
_windows: dict[str, deque[bool]] = {}
_window_latched: set[str] = set()


def _window_push(reason: str, degraded: bool) -> tuple[bool, int]:
    """Append an outcome; return ``(alarm_now, degraded_in_window)``. Lock held."""
    window = _windows.get(reason)
    if window is None:
        window = _windows[reason] = deque(maxlen=WINDOW)
    window.append(degraded)
    count = sum(window)
    if reason in _window_latched:
        if count <= WINDOW_REARM_DEGRADED:
            _window_latched.discard(reason)
        return False, count
    if count >= WINDOW_ALERT_DEGRADED:
        _window_latched.add(reason)
        return True, count
    return False, count


def _alarm(reason: str, **fields: object) -> None:
    with contextlib.suppress(Exception):
        log.error(
            "sandbox.chain_accounting_degraded_alarm",
            alarm_event=True,
            reason=reason,
            **fields,
        )


def record_degraded(reason: Reason, subject: str = "", **context: object) -> None:
    """Count one degraded FRESH probe of ``subject``; alarm on a per-subject
    streak or on a degraded-dominated window."""
    key = (reason, subject)
    with _lock:
        _totals[reason] = _totals.get(reason, 0) + 1
        if key not in _streaks and len(_streaks) >= _MAX_SUBJECTS:
            _streaks.pop(next(iter(_streaks)))
        streak = _streaks.get(key, 0) + 1
        _streaks[key] = streak
        window_alarm, window_count = _window_push(reason, True)
    if _DEGRADED is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _DEGRADED.labels(reason=reason).inc()
    if streak == ALERT_STREAK:
        _alarm(reason, trigger="streak", subject=subject, consecutive=streak, **context)
    if window_alarm:
        _alarm(
            reason,
            trigger="window",
            subject=subject,
            degraded_in_window=window_count,
            window=WINDOW,
            **context,
        )


def record_measured(subject: str = "", *reasons: Reason) -> None:
    """A successful FRESH measurement of ``subject`` ends ITS streak for
    ``reasons`` (never another subject's). Do not call on cache hits."""
    with _lock:
        for reason in reasons:
            _streaks.pop((reason, subject), None)
            _window_push(reason, False)


def degraded_total(reason: Reason) -> int:
    with _lock:
        return _totals.get(reason, 0)


def degraded_streak(reason: Reason, subject: str | None = None) -> int:
    """Current streak of ``subject``; with no subject, the longest for ``reason``."""
    with _lock:
        if subject is not None:
            return _streaks.get((reason, subject), 0)
        return max((v for (r, _s), v in _streaks.items() if r == reason), default=0)


def reset() -> None:
    """Test hook: clear all in-process tallies."""
    with _lock:
        _totals.clear()
        _streaks.clear()
        _windows.clear()
        _window_latched.clear()
