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
* a per-reason CONSECUTIVE streak, reset by any successful measurement. When a
  streak reaches :data:`ALERT_STREAK` an error-level
  ``sandbox.chain_accounting_degraded_alarm`` is logged with
  ``alarm_event=True`` (the same alert shape as the worker watchdogs) — once
  per streak, so a persistent fault alarms once rather than flooding.

Corroboration only: nothing here gates snapshot behaviour, and a metrics-sink
failure must never propagate into the snapshot verb.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Literal

from aios.logging import get_logger

log = get_logger("aios.sandbox.snapshot_telemetry")

Reason = Literal["history_unparseable", "history_unavailable", "base_unmeasurable"]

# Consecutive degraded observations (with no success in between) that raise the
# alarm. Low enough that a fleet-wide fault alarms within a handful of
# snapshots; high enough that a single transient daemon hiccup does not.
ALERT_STREAK = 5

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
_streaks: dict[str, int] = {}


def record_degraded(reason: Reason, **context: object) -> None:
    """Count one degraded probe; alarm when the consecutive streak hits the threshold."""
    with _lock:
        _totals[reason] = _totals.get(reason, 0) + 1
        streak = _streaks.get(reason, 0) + 1
        _streaks[reason] = streak
    if _DEGRADED is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _DEGRADED.labels(reason=reason).inc()
    if streak == ALERT_STREAK:
        with contextlib.suppress(Exception):
            log.error(
                "sandbox.chain_accounting_degraded_alarm",
                alarm_event=True,
                reason=reason,
                consecutive=streak,
                **context,
            )


def record_measured(*reasons: Reason) -> None:
    """A successful measurement ends the consecutive streak for ``reasons``."""
    with _lock:
        for reason in reasons:
            _streaks[reason] = 0


def degraded_total(reason: Reason) -> int:
    with _lock:
        return _totals.get(reason, 0)


def degraded_streak(reason: Reason) -> int:
    with _lock:
        return _streaks.get(reason, 0)


def reset() -> None:
    """Test hook: clear all in-process tallies."""
    with _lock:
        _totals.clear()
        _streaks.clear()
