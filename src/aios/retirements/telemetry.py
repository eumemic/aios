"""Retirement read-tolerance telemetry — corroboration only, NEVER the gate.

When the ``mode="before"`` read-tolerance validator (``models/agents.py``) maps a
persisted legacy tool ``type`` to its canonical successor, it fires a *tolerance
hit*: evidence that an unmigrated row is still being read. This module records
those fires as **corroboration / telemetry only**:

* a per-token Prometheus counter ``retirement_tolerance_hits{token=...}``, and
* the wall-clock ``last_seen`` of the most recent fire per token.

Nothing here is load-bearing. The gate that decides whether a token is *tolerated*
is the registry's ``contract_rev IS NULL`` predicate (see
:func:`aios.retirements.registry.tolerated_rename_map`); a teardown decision is
made from the registry + boot-gate, never from these counters. Reading or
clearing these values must never change validation behavior.

#1324 metrics surface
---------------------
The epic wires these counters through the shared Prometheus surface introduced in
#1324. That surface is an *optional* sink here: if ``prometheus_client`` (the
backing library the #1324 surface uses) is importable, each fire increments a
real ``Counter``; otherwise the in-process tallies below are the sole record.
Either way the in-process ``hits``/``last_seen`` snapshots are always maintained,
so this module is self-contained and testable without the metrics deployment —
and a fire is never silently lost if the surface is absent.
"""

from __future__ import annotations

import contextlib
import threading
from datetime import UTC, datetime

# ── Optional Prometheus sink (the #1324 metrics surface) ──────────────────────
#
# The #1324 surface is built on ``prometheus_client``. We bind to it lazily and
# tolerate its absence: the validator must never fail to load a row just because
# the metrics deployment is missing. A tolerance hit is recorded in-process
# regardless; the Counter is incremented in addition when the library is present.
try:  # pragma: no cover - exercised only where prometheus_client is installed
    from prometheus_client import Counter as _PromCounter

    _RETIREMENT_TOLERANCE_HITS = _PromCounter(
        "retirement_tolerance_hits",
        "Read-tolerance fires per retired token (corroboration only; NEVER a gate).",
        ["token"],
    )
except Exception:  # pragma: no cover - the common path in this repo today
    _RETIREMENT_TOLERANCE_HITS = None


# ── In-process tallies (always maintained) ────────────────────────────────────

_lock = threading.Lock()
_hits: dict[str, int] = {}
_last_seen: dict[str, datetime] = {}


def record_tolerance_hit(token: str, *, now: datetime | None = None) -> None:
    """Record one read-tolerance fire for ``token`` — corroboration only.

    Increments ``retirement_tolerance_hits{token}`` (Prometheus, when the #1324
    surface is present) and stamps ``last_seen[token]`` with the fire time. This
    is telemetry: callers MUST NOT branch validation/teardown on its result, and
    this function MUST NOT raise — a metrics failure can never wedge a read.

    ``now`` is injectable for deterministic tests; it defaults to the current UTC
    time.
    """

    stamp = now or datetime.now(UTC)
    with _lock:
        _hits[token] = _hits.get(token, 0) + 1
        _last_seen[token] = stamp
    if _RETIREMENT_TOLERANCE_HITS is not None:  # pragma: no cover
        # Telemetry is never load-bearing: swallow any metrics-sink error so a
        # broken surface can never wedge a read.
        with contextlib.suppress(Exception):
            _RETIREMENT_TOLERANCE_HITS.labels(token=token).inc()


def tolerance_hits(token: str) -> int:
    """In-process count of tolerance fires for ``token`` (corroboration only)."""

    with _lock:
        return _hits.get(token, 0)


def last_seen(token: str) -> datetime | None:
    """Wall-clock time of the most recent tolerance fire for ``token``, or ``None``."""

    with _lock:
        return _last_seen.get(token)


def reset() -> None:
    """Clear the in-process tallies. Test helper; has no effect on the gate."""

    with _lock:
        _hits.clear()
        _last_seen.clear()
