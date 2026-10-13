"""A wake that raised is not throughput for the dead-man (#2541 review F1)."""

from __future__ import annotations

from typing import Any

import pytest

from aios.harness.production_watchdogs import ThroughputDeadMan, WakeThroughputProbe
from aios.harness.throughput import CompletionCounter


class _StuckConn:
    """One claimed (``doing``) wake that never finishes."""

    async def fetchval(self, query: str, *args: object, **kwargs: object) -> int:
        return 1


@pytest.mark.asyncio
async def test_failing_wakes_do_not_mask_a_stuck_claimed_wake() -> None:
    counter = CompletionCounter()
    probe = WakeThroughputProbe(counter)
    alarms: list[tuple[str, dict[str, Any]]] = []
    dead_man = ThroughputDeadMan(
        threshold_seconds=600,
        rate_limit_seconds=60,
        alarm=lambda name, detail: alarms.append((name, detail)),
    )
    conn = _StuckConn()

    for tick in range(1, 4):
        with pytest.raises(RuntimeError), counter.track():
            raise RuntimeError("wake step failed")
        claimed, completed = await probe.sample(conn)
        dead_man.observe(claimed=claimed, completed=completed, now=tick * 600)

    assert any(name == "throughput_dead_man" for name, _ in alarms), (
        "a stuck claimed wake must alarm even while other wakes keep failing"
    )
    assert counter.failed == 3


@pytest.mark.asyncio
async def test_successful_wakes_still_reset_the_stall_timer() -> None:
    counter = CompletionCounter()
    probe = WakeThroughputProbe(counter)
    alarms: list[str] = []
    dead_man = ThroughputDeadMan(
        threshold_seconds=600, rate_limit_seconds=60, alarm=lambda n, d: alarms.append(n)
    )
    conn = _StuckConn()
    for tick in range(1, 4):
        with counter.track():
            pass
        claimed, completed = await probe.sample(conn)
        dead_man.observe(claimed=claimed, completed=completed, now=tick * 600)
    assert alarms == []
