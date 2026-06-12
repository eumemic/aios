"""The sandbox re-dispatch horizon derivation (#988, sweep Option 1).

``bash`` rides the ``tool`` capability, so the sweep's ``tool`` stale-clause is the
backstop that re-dispatches a bash call whose worker crashed before signalling. The
horizon must exceed the maximum wall-clock a live bash exec can occupy — the bash
ceiling PLUS provisioning slack — so the sweep never re-drives a still-running exec.
"""

from __future__ import annotations

from aios.workflows.sweep import (
    SANDBOX_PROVISIONING_SLACK_SECONDS,
    _sandbox_redispatch_horizon,
)


def test_horizon_floors_at_300_for_default_ceiling() -> None:
    # The 120s default ceiling: 120 + 180 = 300 — the floor preserves the original
    # tool stale value, so common-case behaviour is unchanged.
    assert _sandbox_redispatch_horizon(120) == 300.0


def test_horizon_scales_with_a_raised_ceiling() -> None:
    # An operator who raises the bash ceiling widens the horizon to match (ceiling
    # + slack), so the two never drift apart and a slow exec is never re-driven.
    assert _sandbox_redispatch_horizon(600) == 600 + SANDBOX_PROVISIONING_SLACK_SECONDS


def test_horizon_never_below_floor() -> None:
    # A tiny ceiling still clamps up to the 300s floor.
    assert _sandbox_redispatch_horizon(10) == 300.0
