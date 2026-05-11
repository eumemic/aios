"""E2E tests for the stalled-procrastinate-job reaper.

A worker process can die mid-job (laptop sleep, OOM, ungraceful shutdown)
leaving the row at ``status='doing'``. Procrastinate's heartbeat lease
(``procrastinate_workers.last_heartbeat``) is what we lean on to detect
this — :meth:`procrastinate.manager.JobManager.get_stalled_jobs` returns
``doing`` jobs whose worker has been silent for longer than the
threshold (we use 60s).

These tests pin the lease semantics end-to-end against a real
procrastinate schema so a refactor can't quietly break recovery.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from procrastinate import PsycopgConnector
from procrastinate.manager import JobManager

from aios.harness.sweep import reap_stalled_jobs
from tests.conftest import needs_docker


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    p = await create_pool(get_settings().db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def job_manager(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """A fresh ``JobManager`` pointed at the testcontainer DB.

    The module-level ``procrastinate_app`` singleton fixes its
    connector at import time — by the time the fixture runs, that
    connector is already bound to whatever settings were active at
    import (typically the dev DB or nothing). We build a fresh
    connector so the manager talks to the same DB the rest of the
    test fixtures are seeding into.
    """
    conninfo = aios_env["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://")
    connector = PsycopgConnector(conninfo=conninfo)
    await connector.open_async()
    try:
        yield JobManager(connector=connector)
    finally:
        await connector.close_async()


async def _insert_worker(conn: Any, *, stale_seconds: int = 0) -> int:
    """Insert a procrastinate_workers row.

    ``stale_seconds`` controls the heartbeat age — 0 means fresh,
    >60 means past our reap threshold.
    """
    return int(
        await conn.fetchval(
            """
            INSERT INTO procrastinate_workers (last_heartbeat)
            VALUES (NOW() - ($1 || ' second')::interval)
            RETURNING id
            """,
            str(stale_seconds),
        )
    )


async def _insert_job(
    conn: Any,
    *,
    status: str,
    worker_id: int | None,
    lock: str = "lk-test",
    queueing_lock: str | None = "lk-test",
) -> int:
    """Insert a minimal procrastinate_jobs row.

    Defaults match an aios wake job (per-session lock); each test
    overrides only what it cares about.
    """
    return int(
        await conn.fetchval(
            """
            INSERT INTO procrastinate_jobs
                (queue_name, task_name, priority, lock, queueing_lock, args, status, worker_id, attempts)
            VALUES ('sessions', 'harness.wake_session', 0, $1, $2, '{}'::jsonb, $3, $4, 0)
            RETURNING id
            """,
            lock,
            queueing_lock,
            status,
            worker_id,
        )
    )


@needs_docker
class TestReapStalledJobs:
    async def test_reaps_doing_with_null_worker_id(self, pool: Any, job_manager: Any) -> None:
        """``worker_id IS NULL`` is procrastinate's "worker dropped" state.

        After procrastinate prunes a stalled worker from
        ``procrastinate_workers``, any job it had ``doing`` ends up
        with ``worker_id = NULL`` — genuinely orphaned and should be
        reaped.
        """
        async with pool.acquire() as conn:
            jid = await _insert_job(conn, status="doing", worker_id=None)
            try:
                reaped = await reap_stalled_jobs(job_manager)
                row = await conn.fetchrow("SELECT status FROM procrastinate_jobs WHERE id=$1", jid)
                assert reaped >= 1
                assert row["status"] == "failed"
            finally:
                await conn.execute("DELETE FROM procrastinate_jobs WHERE id=$1", jid)

    async def test_reaps_doing_with_stale_worker_heartbeat(
        self, pool: Any, job_manager: Any
    ) -> None:
        """A worker whose heartbeat is older than the threshold is stalled.

        This is the lease semantics: live workers extend their
        heartbeat every ``update_heartbeat_interval``; if the worker
        dies, the heartbeat goes stale, and ``get_stalled_jobs``
        finds the orphan even before procrastinate's prune deletes
        the row.
        """
        async with pool.acquire() as conn:
            wid = await _insert_worker(conn, stale_seconds=120)
            jid = await _insert_job(conn, status="doing", worker_id=wid)
            try:
                reaped = await reap_stalled_jobs(job_manager)
                row = await conn.fetchrow("SELECT status FROM procrastinate_jobs WHERE id=$1", jid)
                assert reaped >= 1
                assert row["status"] == "failed"
            finally:
                await conn.execute("DELETE FROM procrastinate_jobs WHERE id=$1", jid)
                await conn.execute("DELETE FROM procrastinate_workers WHERE id=$1", wid)

    async def test_leaves_doing_with_live_worker_alone(self, pool: Any, job_manager: Any) -> None:
        """A live worker's in-flight job MUST NOT be reaped.

        The safety case — under normal load, jobs sit at
        ``status='doing'`` while their worker is processing. The
        lease check on ``last_heartbeat`` distinguishes legitimate
        live work from genuinely abandoned jobs.
        """
        async with pool.acquire() as conn:
            wid = await _insert_worker(conn, stale_seconds=0)
            jid = await _insert_job(conn, status="doing", worker_id=wid)
            try:
                await reap_stalled_jobs(job_manager)
                row = await conn.fetchrow("SELECT status FROM procrastinate_jobs WHERE id=$1", jid)
                assert row["status"] == "doing"
            finally:
                await conn.execute("DELETE FROM procrastinate_jobs WHERE id=$1", jid)
                await conn.execute("DELETE FROM procrastinate_workers WHERE id=$1", wid)

    async def test_ignores_other_statuses(self, pool: Any, job_manager: Any) -> None:
        """Only ``status='doing'`` is candidate for reaping.

        ``todo`` jobs without a worker are normal queued state.
        ``succeeded`` / ``failed`` are terminal. Reaping any of
        these would corrupt history.
        """
        async with pool.acquire() as conn:
            todo_id = await _insert_job(conn, status="todo", worker_id=None)
            done_id = await _insert_job(
                conn,
                status="succeeded",
                worker_id=None,
                lock="lk-test-done",
                queueing_lock=None,
            )
            try:
                await reap_stalled_jobs(job_manager)
                rows = await conn.fetch(
                    "SELECT id, status FROM procrastinate_jobs WHERE id = ANY($1::bigint[])",
                    [todo_id, done_id],
                )
                statuses = {r["id"]: r["status"] for r in rows}
                assert statuses[todo_id] == "todo"
                assert statuses[done_id] == "succeeded"
            finally:
                await conn.execute(
                    "DELETE FROM procrastinate_jobs WHERE id = ANY($1::bigint[])",
                    [todo_id, done_id],
                )

    async def test_returns_zero_when_no_stalled_jobs(self, pool: Any, job_manager: Any) -> None:
        """Steady state: no stalled jobs, return 0."""
        reaped = await reap_stalled_jobs(job_manager)
        assert reaped == 0
