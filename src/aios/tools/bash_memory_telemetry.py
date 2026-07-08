"""Telemetry for the bash memory-mount reconcile phase (#1748).

The #1733 zero-inference-gap profiler measures the event-loop gap directly
and will see the on-loop hash vanish once the stat-prefilter ships — but
nothing in that profile tells you whether a LATER edit re-introduced an
on-loop read/hash, nor whether the ``asyncio.to_thread`` offload is
saturating the default thread pool under concurrent load. This module is the
corroboration layer for both:

* A per-call **candidate-read counter** — the number of files actually read
  and hashed by one ``reconcile_memory_mounts`` call. The all-unchanged path
  MUST report 0; a regression that silently reverts to "hash everything"
  shows up here as a counter spike, ideally caught before it ever shows up as
  an inference-gap regression.
* Per-phase duration samples (``before_scan``, ``after_scan``, ``db_phase``)
  so the "moved off the event loop" claim is a gated, measured property, not
  an eyeballed one.

Nothing here is load-bearing for correctness — this module NEVER gates
reconcile behavior, only observes it. A metrics-sink failure must never wedge
a bash call, matching the pattern in
:mod:`aios.retirements.telemetry`.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

# ── Optional Prometheus sink (the #1324 metrics surface) ──────────────────────
try:  # pragma: no cover - exercised only where prometheus_client is installed
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Histogram as _PromHistogram

    _CANDIDATE_READS = _PromCounter(
        "bash_memory_reconcile_candidate_reads_total",
        "Candidate file reads/hashes performed across all bash memory-reconcile calls.",
    )
    _PHASE_SECONDS = _PromHistogram(
        "bash_memory_reconcile_phase_seconds",
        "Duration of a named bash memory-reconcile phase.",
        ["phase"],
    )
except Exception:  # pragma: no cover - the common path in this repo today
    _CANDIDATE_READS = None
    _PHASE_SECONDS = None


# ── In-process tallies (always maintained) ────────────────────────────────────

_lock = threading.Lock()
_candidate_read_total = 0
_last_candidate_read_count = 0
_phase_totals: dict[str, float] = {}
_phase_counts: dict[str, int] = {}


def record_candidate_reads(count: int) -> None:
    """Record the number of candidate reads/hashes performed by one reconcile call.

    The all-unchanged path MUST report 0 — a regression that re-hashes
    everything shows up as a counter spike before it shows up as an
    event-loop-gap regression in the #1733 profiler.
    """
    global _candidate_read_total, _last_candidate_read_count
    with _lock:
        _candidate_read_total += count
        _last_candidate_read_count = count
    if _CANDIDATE_READS is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _CANDIDATE_READS.inc(count)


def last_candidate_read_count() -> int:
    """The candidate-read count from the most recent :func:`record_candidate_reads` call."""
    with _lock:
        return _last_candidate_read_count


def total_candidate_reads() -> int:
    """Cumulative candidate reads across the process lifetime."""
    with _lock:
        return _candidate_read_total


def record_phase_duration(phase: str, duration_s: float) -> None:
    """Record one duration sample for a named reconcile phase.

    Phases: ``before_scan`` (the pre-exec stat walk), ``after_scan`` (the
    post-exec stat walk + candidate reads, run via ``asyncio.to_thread``),
    ``db_phase`` (the on-loop create/update/delete calls). Corroboration
    only — never gates behavior.
    """
    with _lock:
        _phase_totals[phase] = _phase_totals.get(phase, 0.0) + duration_s
        _phase_counts[phase] = _phase_counts.get(phase, 0) + 1
    if _PHASE_SECONDS is not None:  # pragma: no cover
        with contextlib.suppress(Exception):
            _PHASE_SECONDS.labels(phase=phase).observe(duration_s)


def phase_totals() -> dict[str, float]:
    """Cumulative duration per phase across the process lifetime."""
    with _lock:
        return dict(_phase_totals)


def phase_counts() -> dict[str, int]:
    """Cumulative sample count per phase across the process lifetime."""
    with _lock:
        return dict(_phase_counts)


@contextmanager
def timed_phase(phase: str) -> Iterator[None]:
    """Context manager that records one duration sample for ``phase`` on exit.

    Wraps the body in a monotonic-clock timer; never raises on its own (a
    metrics failure inside :func:`record_phase_duration`'s Prometheus sink is
    already swallowed there), and re-raises whatever the body raised.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        record_phase_duration(phase, time.monotonic() - start)


def reset() -> None:
    """Test-only: clear all in-process tallies."""
    global _candidate_read_total, _last_candidate_read_count
    with _lock:
        _candidate_read_total = 0
        _last_candidate_read_count = 0
        _phase_totals.clear()
        _phase_counts.clear()
