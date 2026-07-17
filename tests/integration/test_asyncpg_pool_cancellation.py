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
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import asyncpg.pool
import pytest

from aios.harness import loop, runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]

_POOL_SIZE = 16
_INCIDENT_HOLDERS = 9
_STORMS_PER_PHASE = 112
_TASK_BOUND = 5.0
_PHASES = ("post_query", "release", "acquire")
_INCIDENT_PHASES = ("incident_http", "incident_stream")
_INCIDENT_WAITERS = 13
_INCIDENT_STORMS_PER_PHASE = 8
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


class _SlowNonDB:
    """Sleep-backed HTTP/model-stream double synchronized after transaction entry."""

    def __init__(self, probe: _Probe, phase: str) -> None:
        self.probe = probe
        self.phase = phase
        self.entered_count = 0
        self.ready_count = 0
        self.all_ready = asyncio.Event()
        self.all_entered = asyncio.Event()
        self.armed = asyncio.Event()

    async def _slow_await(self) -> None:
        self.probe.census[self.phase] += 1
        self.entered_count += 1
        if self.entered_count == _POOL_SIZE:
            self.all_entered.set()
        await asyncio.sleep(60)

    async def http_post(self) -> None:
        """Mock an HTTP POST whose transport is a slow asyncio sleep."""
        await self._slow_await()

    async def stream(self) -> Any:
        """Mock a model response that blocks before its first streamed chunk."""
        await self._slow_await()
        yield b"unreachable-until-timeout"


async def _incident_holder(pool: asyncpg.Pool[Any], slow: _SlowNonDB) -> None:
    async with pool.acquire() as connection, connection.transaction():
        slow.ready_count += 1
        if slow.ready_count == _POOL_SIZE:
            slow.all_ready.set()
        # Arm only after all 16 transactions are holding all 16 pool slots and
        # the 13 incident-shaped pool waiters have actually queued.
        await slow.armed.wait()
        if slow.phase == "incident_http":
            await asyncio.wait_for(slow.http_post(), timeout=0.05)
        else:
            stream = slow.stream()
            try:
                await asyncio.wait_for(anext(stream), timeout=0.05)
            finally:
                await stream.aclose()


async def _queued_witness(pool: asyncpg.Pool[Any], acquired: Counter[str]) -> None:
    async with pool.acquire() as connection:
        await connection.fetchval("SELECT 1")
        acquired["waiters"] += 1


@pytest.mark.asyncio
async def test_incident_call_graph_timeout_under_saturated_pool(
    migrated_db_url: str,
) -> None:
    """Probe timeout cancellation of non-DB awaits under a held transaction."""
    probe = _Probe()
    pool = await asyncpg.create_pool(
        migrated_db_url,
        min_size=_POOL_SIZE,
        max_size=_POOL_SIZE,
        server_settings={"application_name": "aios-1975-phase-probe"},
    )
    assert pool is not None
    observer = await asyncpg.connect(migrated_db_url)
    try:
        for phase in _INCIDENT_PHASES:
            for storm in range(_INCIDENT_STORMS_PER_PHASE):
                slow = _SlowNonDB(probe, phase)
                holders = [
                    asyncio.create_task(
                        _incident_holder(pool, slow),
                        name=f"incident-{phase}-{storm}-{index}",
                    )
                    for index in range(_POOL_SIZE)
                ]
                async with asyncio.timeout(_TASK_BOUND):
                    # Every holder has acquired a connection and opened a
                    # transaction before it reaches this mocked non-DB await.
                    await slow.all_ready.wait()
                assert pool.get_idle_size() == 0

                acquired: Counter[str] = Counter()
                waiters = [
                    asyncio.create_task(
                        _queued_witness(pool, acquired),
                        name=f"incident-waiter-{phase}-{storm}-{index}",
                    )
                    for index in range(_INCIDENT_WAITERS)
                ]
                async with asyncio.timeout(_TASK_BOUND):
                    while len(pool._queue._getters) < _INCIDENT_WAITERS:  # noqa: ASYNC110 -- waiter census
                        await asyncio.sleep(0)

                slow.armed.set()
                async with asyncio.timeout(_TASK_BOUND):
                    await slow.all_entered.wait()
                await _assert_recovered(pool, observer, holders + waiters)
                assert acquired["waiters"] == _INCIDENT_WAITERS, (
                    "queued waiters did not eventually acquire"
                )

                # wait_for must have delivered TimeoutError through every open
                # transaction; a success or unrelated exception is a bad probe.
                results = [task.exception() for task in holders]
                assert all(isinstance(result, TimeoutError) for result in results)

        expected = _INCIDENT_STORMS_PER_PHASE * _POOL_SIZE
        for phase in _INCIDENT_PHASES:
            assert probe.census[phase] == expected
            assert probe.census[phase] > 0
    finally:
        await observer.close()
        async with asyncio.timeout(_TASK_BOUND):
            await pool.close()


@pytest.mark.asyncio
async def test_real_run_session_step_timeout_wrapper_releases_saturated_pool(
    migrated_db_url: str,
) -> None:
    """Drive the production job-level timeout wrapper at the incident pressure.

    Unlike the distilled call-graph probe above, cancellation here is created by
    ``run_session_step``'s real ``asyncio.wait_for`` and crosses its nested
    timeout/error/finally/unregister/step-end control flow.  The body seam is
    replaced only so all sixteen steps can be synchronized at the two observed
    non-DB frames without constructing paid model calls or OAuth secrets.
    """
    pool = await asyncpg.create_pool(
        migrated_db_url,
        min_size=_POOL_SIZE,
        max_size=_POOL_SIZE,
        server_settings={"application_name": "aios-1975-phase-probe"},
    )
    assert pool is not None
    observer = await asyncpg.connect(migrated_db_url)
    previous_pool = runtime.pool
    previous_registry = runtime.inflight_tool_registry
    runtime.pool = pool
    runtime.inflight_tool_registry = InflightToolRegistry()
    event_seq = 0

    async def append_event(*args: Any, **kwargs: Any) -> Any:
        nonlocal event_seq
        event_seq += 1
        return SimpleNamespace(id=f"evt_{event_seq}", seq=event_seq)

    async def load_account(*args: Any, **kwargs: Any) -> str:
        return "acc_1975"

    async def timeout_result(*args: Any, **kwargs: Any) -> None:
        return None

    try:
        for phase in _INCIDENT_PHASES:
            probe = _Probe()
            for storm in range(_INCIDENT_STORMS_PER_PHASE):
                slow = _SlowNonDB(probe, phase)

                async def real_timeout_body(
                    *args: Any, _slow: _SlowNonDB = slow, _phase_name: str = phase, **kwargs: Any
                ) -> Any:
                    # This is the exact connection/transaction/non-DB-await
                    # shape; the outer production wrapper supplies cancellation.
                    async with pool.acquire() as connection, connection.transaction():
                        _slow.ready_count += 1
                        if _slow.ready_count == _POOL_SIZE:
                            _slow.all_ready.set()
                        await _slow.armed.wait()
                        if _phase_name == "incident_http":
                            await _slow.http_post()
                        else:
                            stream = _slow.stream()
                            try:
                                await anext(stream)
                            finally:
                                await stream.aclose()

                with (
                    mock.patch.object(loop, "HARNESS_STEP_TIMEOUT_S", 0.05),
                    mock.patch.object(loop, "_run_session_step_body", real_timeout_body),
                    mock.patch.object(
                        sessions_service,
                        "load_live_session_account_id",
                        load_account,
                    ),
                    mock.patch.object(sessions_service, "append_event", append_event),
                    mock.patch.object(loop, "_handle_step_timeout", timeout_result),
                    mock.patch.object(loop, "defer_wake", new=AsyncMock()),
                ):
                    holders = [
                        asyncio.create_task(
                            loop.run_session_step(f"sess_1975_{phase}_{storm}_{index}"),
                            name=f"real-step-{phase}-{storm}-{index}",
                        )
                        for index in range(_POOL_SIZE)
                    ]
                    async with asyncio.timeout(_TASK_BOUND):
                        await slow.all_ready.wait()
                    assert pool.get_idle_size() == 0

                    acquired: Counter[str] = Counter()
                    waiters = [
                        asyncio.create_task(
                            _queued_witness(pool, acquired),
                            name=f"real-step-waiter-{phase}-{storm}-{index}",
                        )
                        for index in range(_INCIDENT_WAITERS)
                    ]
                    async with asyncio.timeout(_TASK_BOUND):
                        while len(pool._queue._getters) < _INCIDENT_WAITERS:  # noqa: ASYNC110
                            await asyncio.sleep(0)
                    slow.armed.set()
                    async with asyncio.timeout(_TASK_BOUND):
                        await slow.all_entered.wait()
                    await _assert_recovered(pool, observer, holders + waiters)
                    assert all(task.exception() is None for task in holders)
                    assert acquired["waiters"] == _INCIDENT_WAITERS

            assert probe.census[phase] == _INCIDENT_STORMS_PER_PHASE * _POOL_SIZE
    finally:
        runtime.pool = previous_pool
        runtime.inflight_tool_registry = previous_registry
        await observer.close()
        async with asyncio.timeout(_TASK_BOUND):
            await pool.close()
