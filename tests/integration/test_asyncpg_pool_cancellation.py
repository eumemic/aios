"""Phase-complete probe for the asyncpg cancellation hypothesis in #1975.

This is deliberately a diagnostic invariant test, not a claim that the incident
has been reproduced.  Every cancellation is synchronized inside asyncpg's
acquire/query/release machinery; an unreached phase is a harness failure.
"""

from __future__ import annotations

import asyncio
import contextvars
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

import asyncpg
import asyncpg.pool
import pytest

from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]

_POOL_SIZE = 16
_INCIDENT_HOLDERS = 9
_STORMS_PER_PHASE = 112
_TASK_BOUND = 5.0
_PHASES = ("post_query", "release", "acquire")
_phase: contextvars.ContextVar[str | None] = contextvars.ContextVar("asyncpg_phase", default=None)


@dataclass
class _Ticket:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    proceed: asyncio.Event = field(default_factory=asyncio.Event)


class _Probe:
    current: ClassVar[_Probe | None] = None

    def __init__(self) -> None:
        self.census: Counter[str] = Counter()
        self.tickets: dict[str, list[_Ticket]] = {name: [] for name in _PHASES}

    async def checkpoint(self, name: str) -> None:
        ticket = _Ticket()
        self.tickets[name].append(ticket)
        self.census[name] += 1
        ticket.entered.set()
        await ticket.proceed.wait()

    async def wait_for_hits(self, name: str, before: int, count: int) -> list[_Ticket]:
        async with asyncio.timeout(_TASK_BOUND):
            while len(self.tickets[name]) < before + count:  # noqa: ASYNC110 -- dynamic tickets
                await asyncio.sleep(0)
        tickets = self.tickets[name][before : before + count]
        await asyncio.gather(*(ticket.entered.wait() for ticket in tickets))
        return tickets


class _InstrumentedConnection(asyncpg.Connection):  # type: ignore[misc]
    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: float | None = None,  # noqa: ASYNC109 -- asyncpg override signature
    ) -> Any:
        value = await super().fetchval(query, *args, column=column, timeout=timeout)
        # super().fetchval has completed protocol execution and materialized the
        # result, but this override has not delivered it to the caller yet.
        if _phase.get() == "post_query" and "phase_probe" in query:
            probe = _Probe.current
            assert probe is not None
            await probe.checkpoint("post_query")
        return value


async def _holder(pool: asyncpg.Pool[Any], phase: str) -> None:
    token = _phase.set(phase)
    try:
        async with pool.acquire() as connection:
            if phase != "acquire":
                await connection.fetchval("SELECT 1 /* phase_probe */")
    finally:
        _phase.reset(token)


async def _pool_census(
    pool: asyncpg.Pool[Any], observer: asyncpg.Connection[Any]
) -> dict[str, int]:
    holders = pool._holders  # this test explicitly audits asyncpg internals
    return {
        "size": pool.get_size(),
        "idle": pool.get_idle_size(),
        "queue": pool._queue.qsize(),
        "connected": sum(holder._con is not None for holder in holders),
        "proxies": sum(holder._proxy is not None for holder in holders),
        "server": await observer.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1",
            "aios-1975-phase-probe",
        ),
    }


async def _assert_recovered(
    pool: asyncpg.Pool[Any], observer: asyncpg.Connection[Any], tasks: list[asyncio.Task[None]]
) -> None:
    async with asyncio.timeout(_TASK_BOUND):
        await asyncio.gather(*tasks, return_exceptions=True)
        # Shielded Pool.release continues after cancellation of its caller.
        while pool.get_idle_size() != _POOL_SIZE:  # noqa: ASYNC110 -- shielded releases
            await asyncio.sleep(0)
    assert all(task.done() for task in tasks), "a cancelled coroutine remained parked"

    # Cancellation during setup legitimately closes that holder's connection.
    # Exercise every slot concurrently so lazy pool reconnection must restore
    # physical capacity before taking the leak census.
    acquired: list[asyncpg.pool.PoolConnectionProxy] = []
    async with asyncio.timeout(_TASK_BOUND):
        acquired = list(await asyncio.gather(*(pool.acquire() for _ in range(_POOL_SIZE))))
        await asyncio.gather(*(pool.release(connection) for connection in acquired))
    assert await _pool_census(pool, observer) == {
        "size": _POOL_SIZE,
        "idle": _POOL_SIZE,
        "queue": _POOL_SIZE,
        "connected": _POOL_SIZE,
        "proxies": 0,
        "server": _POOL_SIZE,
    }


@pytest.mark.asyncio
async def test_phase_complete_asyncpg_cancellation_probe(
    migrated_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explore 3,024 instrumented incident-shaped cancellations without leaks."""
    probe = _Probe()
    _Probe.current = probe
    original_release = asyncpg.pool.PoolConnectionHolder.release
    original_acquire = asyncpg.pool.PoolConnectionHolder.acquire

    async def instrumented_acquire(holder: Any) -> Any:
        # Pool._acquire has removed this holder from its queue, but holder.acquire
        # has not yet attached a proxy: this is the incident's mid-acquire await.
        if _phase.get() == "acquire":
            await probe.checkpoint("acquire")
        return await original_acquire(holder)

    async def instrumented_release(
        holder: Any,
        timeout: float | None,  # noqa: ASYNC109 -- patched asyncpg signature
    ) -> None:
        if _phase.get() == "release":
            # The caller's cancellation causes Pool.release to spawn a second,
            # shielded release task. Do not instrument that cleanup twice.
            _phase.set(None)
            await probe.checkpoint("release")
        await original_release(holder, timeout)

    monkeypatch.setattr(asyncpg.pool.PoolConnectionHolder, "acquire", instrumented_acquire)
    monkeypatch.setattr(asyncpg.pool.PoolConnectionHolder, "release", instrumented_release)

    pool = await asyncpg.create_pool(
        migrated_db_url,
        min_size=_POOL_SIZE,
        max_size=_POOL_SIZE,
        connection_class=_InstrumentedConnection,
        server_settings={"application_name": "aios-1975-phase-probe"},
    )
    assert pool is not None
    observer = await asyncpg.connect(migrated_db_url)
    rng = random.Random(1975)
    try:
        assert (await _pool_census(pool, observer))["server"] == _POOL_SIZE
        for phase in _PHASES:
            for _storm in range(_STORMS_PER_PHASE):
                before = len(probe.tickets[phase])
                tasks = [
                    asyncio.create_task(_holder(pool, phase), name=f"phase-probe-{phase}-{index}")
                    for index in range(_INCIDENT_HOLDERS)
                ]
                tickets = await probe.wait_for_hits(phase, before, _INCIDENT_HOLDERS)
                # Task.cancel(), while the target await is active, mirrors the
                # step-timeout source. Random order/jitter explores scheduling.
                rng.shuffle(tasks)
                for task in tasks:
                    await asyncio.sleep(rng.random() / 100_000)
                    task.cancel()
                for ticket in tickets:
                    ticket.proceed.set()
                await _assert_recovered(pool, observer, tasks)

        expected = _STORMS_PER_PHASE * _INCIDENT_HOLDERS
        assert probe.census == Counter({phase: expected for phase in _PHASES})
        assert sum(probe.census.values()) == 3024
    finally:
        _Probe.current = None
        await observer.close()
        async with asyncio.timeout(_TASK_BOUND):
            await pool.close()
