"""#780 — per-wake budget scaling: the pure ``_scaled_seconds`` contract.

Exact values (not just monotonicity): the constant itself is load-bearing — the
integration rescue test only proves the scale exceeds interpreter spawn cost, so
this table is what pins 30s/MiB and the 600s cap.
"""

from __future__ import annotations

import pytest

from aios.workflows.host_launcher import (
    DEADLINE_SECONDS_PER_INIT_MIB,
    MAX_SCALED_SECONDS,
    _scaled_seconds,
)

_MIB = 1024 * 1024


def test_scaled_seconds_exact_values() -> None:
    assert _scaled_seconds(30.0, 0) == 30.0  # empty INIT → base
    assert _scaled_seconds(30.0, _MIB) == 60.0  # +30s per MiB
    assert _scaled_seconds(0.01, 2 * _MIB) == pytest.approx(60.01)
    assert _scaled_seconds(2.0, 100) == pytest.approx(2.0, abs=0.01)  # tiny INIT ≈ base


def test_scaled_seconds_is_capped() -> None:
    # The 64MiB frame cap would otherwise imply a ~32min budget — the cap bounds
    # cancel latency and deploy drain instead.
    assert _scaled_seconds(30.0, 64 * _MIB) == MAX_SCALED_SECONDS
    assert _scaled_seconds(30.0, 1024 * _MIB) == MAX_SCALED_SECONDS


def test_constants_sane() -> None:
    # The cap must dominate the base + a generous real-world INIT, or scaling is dead.
    assert MAX_SCALED_SECONDS > 30.0 + 5 * DEADLINE_SECONDS_PER_INIT_MIB
