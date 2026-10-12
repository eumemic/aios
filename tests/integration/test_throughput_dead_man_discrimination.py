"""Throughput dead-man discrimination against the REAL procrastinate substrate (#2541).

The dead-man once counted ``procrastinate_events`` rows of type ``succeeded``.
The production worker runs with ``delete_jobs="successful"``, which deletes a
successful job (and, by cascade, its events) instead of writing ``succeeded`` —
so ``completed`` was structurally zero and the detector alarmed on any job that
stayed ``doing`` for the threshold, even on a worker finishing jobs every second.

These tests run a real procrastinate worker, with the production
``delete_jobs`` setting and the real ``harness.wake_session`` task body, against
the migrated procrastinate schema — no hand-made fixture rows:

* busy-and-progressing (one long wake parked, others completing) must NOT alarm;
* claimed-and-frozen (one long wake parked, nothing completing) MUST alarm.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import asyncpg
import pytest
from procrastinate import PsycopgConnector

import aios.harness.tasks  # noqa: F401  — registers @app.task handlers on the shared app
from aios.harness.production_watchdogs import ThroughputDeadMan, WakeThroughputProbe
from aios.harness.worker import WORKER_DELETE_JOBS
from aios.jobs.app import QUEUE_SESSIONS, _sync_dsn
from aios.jobs.app import app as procrastinate_app

pytestmark = pytest.mark.integration

_THRESHOLD = 600.0
_LONG = "sess_long_running"


@pytest.fixture
async def worker_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Connection[Any], Callable[[], Awaitable[None]]]]:
    """A real worker on the ``sessions`` queue + an inspector connection.

    ``run_session_step`` is patched: the long session parks on an event, every
    other session returns at once. Yields ``(inspector, release_long)``.
    """
    release = asyncio.Event()

    async def fake_step(session_id: str, **_: Any) -> None:
        if session_id == _LONG:
            await release.wait()

    connector = PsycopgConnector(conninfo=_sync_dsn(migrated_db_url))
    await connector.open_async()
    inspector = await asyncpg.connect(migrated_db_url)
    try:
        with (
            procrastinate_app.replace_connector(connector),
            mock.patch("aios.harness.loop.run_session_step", fake_step),
        ):
            worker = asyncio.create_task(
                procrastinate_app.run_worker_async(
                    queues=[QUEUE_SESSIONS],
                    concurrency=4,
                    wait=True,
                    install_signal_handlers=False,
                    fetch_job_polling_interval=0.05,
                    delete_jobs=WORKER_DELETE_JOBS,
                )
            )

            async def release_long() -> None:
                release.set()

            try:
                yield inspector, release_long
            finally:
                release.set()
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
    finally:
        await inspector.close()
        await connector.close_async()


async def _defer(session_id: str) -> None:
    await procrastinate_app.configure_task(
        "harness.wake_session",
        queue=QUEUE_SESSIONS,
        lock=session_id,
        queueing_lock=session_id,
        allow_unknown=True,
    ).defer_async(session_id=session_id, cause="message")


async def _until(predicate: Callable[[], Awaitable[bool]]) -> None:
    """Poll the DB (no in-process signal exists for procrastinate row state)."""
    for _ in range(500):
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached within 10s")


async def _doing(conn: asyncpg.Connection[Any]) -> int:
    return int(
        await conn.fetchval("SELECT count(*) FROM procrastinate_jobs WHERE status = 'doing'")
    )


async def _remaining(conn: asyncpg.Connection[Any], session_id: str) -> int:
    return int(
        await conn.fetchval(
            "SELECT count(*) FROM procrastinate_jobs WHERE lock = $1 AND status IN ('todo', 'doing')",
            session_id,
        )
    )


async def test_busy_progressing_worker_does_not_alarm(
    worker_env: tuple[asyncpg.Connection[Any], Callable[[], Awaitable[None]]],
) -> None:
    inspector, _release = worker_env
    alarm = MagicMock()
    dead_man = ThroughputDeadMan(threshold_seconds=_THRESHOLD, rate_limit_seconds=1, alarm=alarm)
    probe = WakeThroughputProbe()

    await _defer(_LONG)
    await _until(functools.partial(_has_doing, inspector))
    claimed, completed = await probe.sample(inspector)
    assert claimed == 1
    dead_man.observe(claimed=claimed, completed=completed, now=0)

    # Many ticks past the threshold, a short wake completes in every interval
    # while the long one stays claimed.
    for tick in range(1, 6):
        sid = f"sess_short_{tick}"
        await _defer(sid)
        await _until(functools.partial(_done, inspector, sid))
        claimed, completed = await probe.sample(inspector)
        assert claimed == 1, "the long wake is still claimed"
        assert completed >= 1, "the completed short wake must be counted"
        dead_man.observe(claimed=claimed, completed=completed, now=tick * _THRESHOLD)

    # Regression pin: the substrate genuinely writes no ``succeeded`` event under
    # the production delete setting — the signal the old detector relied on.
    assert (
        await inspector.fetchval(
            "SELECT count(*) FROM procrastinate_events WHERE type = 'succeeded'"
        )
        == 0
    )
    alarm.assert_not_called()


async def test_claimed_and_frozen_worker_alarms(
    worker_env: tuple[asyncpg.Connection[Any], Callable[[], Awaitable[None]]],
) -> None:
    inspector, _release = worker_env
    alarm = MagicMock()
    dead_man = ThroughputDeadMan(threshold_seconds=_THRESHOLD, rate_limit_seconds=1, alarm=alarm)
    probe = WakeThroughputProbe()

    await _defer(_LONG)
    await _until(functools.partial(_has_doing, inspector))
    for tick, now in enumerate((0.0, 1.0, _THRESHOLD + 2)):
        claimed, completed = await probe.sample(inspector)
        assert (claimed, completed) == (1, 0), f"tick {tick}"
        dead_man.observe(claimed=claimed, completed=completed, now=now)

    alarm.assert_called_once()
    assert alarm.call_args.args[0] == "throughput_dead_man"


async def _has_doing(conn: asyncpg.Connection[Any]) -> bool:
    return await _doing(conn) >= 1


async def _done(conn: asyncpg.Connection[Any], session_id: str) -> bool:
    return await _remaining(conn, session_id) == 0
