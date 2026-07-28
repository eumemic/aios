"""Regression for #2008 notify-pool ownership at model awaits."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import asyncpg
import pytest

from aios.harness import completion
from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]
POOL_SIZE = 16
WAITERS = 13
STORMS = 8
TASK_BOUND = 5.0
PHASES = ("incident_http", "incident_stream")


class Probe:
    def __init__(self) -> None:
        self.census: Counter[str] = Counter()


class SlowNonDB:
    def __init__(self, probe: Probe, phase: str) -> None:
        self.probe = probe
        self.phase = phase
        self.entered_count = 0
        self.all_entered = asyncio.Event()

    async def _slow_await(self) -> None:
        self.probe.census[self.phase] += 1
        self.entered_count += 1
        if self.entered_count == POOL_SIZE:
            self.all_entered.set()
        await asyncio.sleep(60)

    async def http_post(self) -> None:
        await self._slow_await()


async def pool_census(pool: asyncpg.Pool[Any], observer: asyncpg.Connection[Any]) -> dict[str, int]:
    return {
        "size": pool.get_size(),
        "idle": pool.get_idle_size(),
        "queue": pool._queue.qsize(),
        "connected": sum(h._con is not None for h in pool._holders),
        "proxies": sum(h._proxy is not None for h in pool._holders),
        "server": await observer.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1",
            "aios-1975-completion-probe",
        ),
    }


async def assert_recovered(
    pool: asyncpg.Pool[Any], observer: asyncpg.Connection[Any], tasks: list[asyncio.Task[None]]
) -> None:
    async with asyncio.timeout(TASK_BOUND):
        await asyncio.gather(*tasks, return_exceptions=True)
        while pool.get_idle_size() != POOL_SIZE:  # noqa: ASYNC110
            await asyncio.sleep(0)
    assert await pool_census(pool, observer) == {
        "size": POOL_SIZE,
        "idle": POOL_SIZE,
        "queue": POOL_SIZE,
        "connected": POOL_SIZE,
        "proxies": 0,
        "server": POOL_SIZE,
    }


class _BlockedModelStream:
    """LiteLLM response double; only the external stream boundary is mocked."""

    def __init__(self, slow: SlowNonDB) -> None:
        self.slow = slow
        self.closed = False

    def __aiter__(self) -> _BlockedModelStream:
        return self

    async def __anext__(self) -> Any:
        await self.slow._slow_await()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


async def _held_capacity_witness(
    pool: asyncpg.Pool[Any], acquired: Counter[str], release: asyncio.Event
) -> None:
    async with pool.acquire() as connection:
        await connection.fetchval("SELECT 1")
        acquired["capacity"] += 1
        await release.wait()


@pytest.mark.asyncio
async def test_completion_external_awaits_do_not_hold_notify_connections(
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise ``stream_litellm`` at both real external-await boundaries.

    This is the #2008 regression: before that change, ``stream_litellm``
    checked out one notify connection around its entire chunk loop.  Therefore
    the ``idle == size`` assertion below failed in the stream phase (all 16
    slots were held while the model had emitted no chunk).  Only
    ``litellm.acompletion`` -- the external HTTP/model boundary -- is replaced;
    connection ownership remains production ``completion.py`` code.
    """
    probe = Probe()
    pool = await asyncpg.create_pool(
        migrated_db_url,
        min_size=POOL_SIZE,
        max_size=POOL_SIZE,
        server_settings={"application_name": "aios-1975-completion-probe"},
    )
    assert pool is not None
    observer = await asyncpg.connect(migrated_db_url)
    responses: list[_BlockedModelStream] = []
    try:
        for phase in PHASES:
            for storm in range(STORMS):
                slow = SlowNonDB(probe, phase)

                async def fake_acompletion(
                    *,
                    _phase_name: str = phase,
                    _slow: SlowNonDB = slow,
                    **_kwargs: Any,
                ) -> _BlockedModelStream:
                    if _phase_name == "incident_http":
                        await _slow.http_post()
                    response = _BlockedModelStream(_slow)
                    responses.append(response)
                    return response

                monkeypatch.setattr(completion.litellm, "acompletion", fake_acompletion)
                calls = [
                    asyncio.create_task(
                        completion.stream_litellm(
                            completion.LlmRequest(
                                messages=[{"role": "user", "content": "probe"}],
                                session_id=f"sess_1975_{phase}_{storm}_{index}",
                            ),
                            model="openai/probe-model",
                            pool=pool,
                        ),
                        name=f"completion-{phase}-{storm}-{index}",
                    )
                    for index in range(POOL_SIZE)
                ]
                async with asyncio.timeout(TASK_BOUND):
                    await slow.all_entered.wait()

                # Decisive pre/post-#2008 evidence: an external HTTP or model
                # stream await owns no notify checkout.  The old chunk-loop
                # scope made this 0 in incident_stream instead of 16.
                assert pool.get_idle_size() == POOL_SIZE
                assert sum(holder._proxy is not None for holder in pool._holders) == 0

                # Prove the advertised capacity is usable while every model call
                # remains parked, rather than relying only on pool counters.
                acquired: Counter[str] = Counter()
                release_witnesses = asyncio.Event()
                witnesses = [
                    asyncio.create_task(
                        _held_capacity_witness(pool, acquired, release_witnesses),
                        name=f"completion-capacity-{phase}-{storm}-{index}",
                    )
                    for index in range(WAITERS)
                ]
                async with asyncio.timeout(TASK_BOUND):
                    while acquired["capacity"] != WAITERS:  # noqa: ASYNC110
                        await asyncio.sleep(0)
                assert pool.get_idle_size() == POOL_SIZE - WAITERS

                for call in calls:
                    call.cancel()
                release_witnesses.set()
                await assert_recovered(pool, observer, calls + witnesses)
                assert all(call.cancelled() for call in calls)

        expected = STORMS * POOL_SIZE
        assert probe.census == Counter({phase: expected for phase in PHASES})
        assert all(response.closed for response in responses)
    finally:
        await observer.close()
        async with asyncio.timeout(TASK_BOUND):
            await pool.close()
