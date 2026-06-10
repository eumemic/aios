"""High-fidelity leak repro driving the full SSE stack to TCP disconnect.

Spins up real uvicorn + real httpx; opens N SSE streams; abruptly drops
each TCP connection (simulating client crash / network blip); polls
``pg_stat_activity`` to confirm the api-side Postgres backends are
reaped.

The production leak only manifests under the full stack: sse-starlette
runs the response generator inside an anyio task group, and on
``http.disconnect`` it cancels the group's scope.  Under anyio's
scope-cancellation, every subsequent ``await`` in the cancelled scope
re-raises ``CancelledError`` immediately — so ``await conn.close()`` in
the listener's finally never completes, asyncpg never sends ``Terminate``
or aborts the transport, and the Postgres backend lingers until TCP
keepalive reaps it (~2h on most distros).

Cleanup via synchronous ``conn.terminate()`` (``transport.abort()``
directly, no await) closes the socket regardless of cancellation state.
This test guards against any future regression that re-introduces an
async cleanup step in the listener helpers' finally blocks.

## Why the backend count is filtered by ``application_name`` (issue #894)

Counting *all* of the database's client backends made this test racy under
``pytest -n`` (xdist): other workers' app pools and listener connections,
plus this server's own pool churning on backfill, moved the global count
non-deterministically. The fix tags every dedicated listener connection
with a per-instance Postgres ``application_name`` —
``aios-listener:<instance_id>``, set in
:func:`aios.db.listen._connect_listener` — and counts ONLY backends
carrying that exact label. The in-process ``live_aios_server`` shares
``get_settings()`` with this test, so both observe the same ``instance_id``
and therefore the same label. Filtering makes the count immune to other
xdist workers and to this server's own app pool. With a clean,
instance-scoped count we can assert FULL reap back to ``baseline`` (every
opened stream's listener backend is gone) rather than the old
``peak - n_streams`` heuristic, which tolerated pool noise and so detected
leaks more weakly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import httpx
import pytest

from aios.config import get_settings
from aios.db.pool import listener_application_name
from tests.conftest import needs_docker
from tests.e2e.conftest import live_aios_server, wait_for_predicate
from tests.helpers.db import count_active_backends
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.docker


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Real-uvicorn aios server.  Sessions-router defer_wake is mocked so
    the SSE backfill query and live tail behave identically to production
    without dragging in a procrastinate worker.
    """
    async with live_aios_server(
        defer_wake_patches=(
            "aios.api.routers.sessions.defer_wake",
            "aios.services.inbound.defer_wake",
        ),
    ) as url:
        yield url


async def _backend_states(db_url: str) -> list[dict[str, Any]]:
    """Snapshot all client backends with state + last query (failure-only diagnostic)."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT pid, state, query, "
            "  EXTRACT(EPOCH FROM (now() - backend_start)) AS age_s, "
            "  EXTRACT(EPOCH FROM (now() - state_change)) AS state_age_s, "
            "  wait_event_type, wait_event "
            "FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid() "
            "ORDER BY backend_start ASC"
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _seed_session(db_url: str) -> str:
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)
    try:
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_test_stub", prefix="sseleak"
        )
        return session.id
    finally:
        await pool.close()


@needs_docker
class TestSseDisconnectLeak:
    async def test_tcp_close_mid_stream_does_not_leak_pg_conns(
        self,
        live_server: str,
        aios_env: dict[str, str],
        migrated_db_url: str,
    ) -> None:
        """Open N SSE streams, drop TCP abruptly, assert conns return to baseline."""
        api_key = aios_env["AIOS_API_KEY"]
        session_id = await _seed_session(migrated_db_url)
        app_name = listener_application_name(get_settings().instance_id)

        # Baseline AFTER session is seeded so the seed-helper pool's
        # lifecycle is out of the way.  Filtered by this instance's listener
        # application_name so the count is immune to other xdist workers and
        # the app pool (issue #894).
        await asyncio.sleep(0.2)
        baseline = await count_active_backends(migrated_db_url, application_name=app_name)

        n_streams = 5
        clients: list[httpx.AsyncClient] = []
        consumer_tasks: list[asyncio.Task[Any]] = []

        async def _consume_one(client: httpx.AsyncClient) -> None:
            """Open the SSE stream and sit on the response until cancelled."""
            async with client.stream(
                "GET",
                f"/v1/sessions/{session_id}/stream",
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                resp.raise_for_status()
                async for _line in resp.aiter_lines():
                    pass

        try:
            for _ in range(n_streams):
                c = httpx.AsyncClient(base_url=live_server, timeout=30.0)
                clients.append(c)
                consumer_tasks.append(asyncio.create_task(_consume_one(c)))

            # Wait for the SSE backends to attach.  Each open stream adds
            # exactly one tagged ``listen_for_events`` backend; pool /
            # backfill connections carry a different application_name and are
            # excluded from this filtered count.  Poll until we see at least
            # n_streams new tagged listener backends.
            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                peak = await count_active_backends(migrated_db_url, application_name=app_name)
                if peak >= baseline + n_streams:
                    break
                if asyncio.get_running_loop().time() > deadline:
                    pytest.fail(
                        f"only {peak - baseline}/{n_streams} backends observed "
                        f"during peak (baseline={baseline}, peak={peak})"
                    )
                await asyncio.sleep(0.1)

            # Hard TCP close: abruptly close each httpx client mid-stream.
            # Closing the client tears down its transport, sending FIN
            # to the server.  sse-starlette's ``_listen_for_disconnect``
            # task should pick up the ``http.disconnect`` message and
            # cancel the response task group.
            for c in clients:
                await c.aclose()

            # Cancel any consumer tasks still hanging on aiter_lines.
            for t in consumer_tasks:
                t.cancel()
            for t in consumer_tasks:
                with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                    await t

        finally:
            # Make sure clients/tasks are fully torn down for the final
            # assertion to be meaningful.
            for c in clients:
                with contextlib.suppress(Exception):
                    await c.aclose()
            for t in consumer_tasks:
                t.cancel()
            for t in consumer_tasks:
                with contextlib.suppress(BaseException):
                    await t

        # After cleanup, every dedicated listener backend opened by the N
        # streams must be reaped.  Because the count is filtered to this
        # instance's listener application_name, the app pool's idle conns
        # don't show up — so we can assert FULL reap back to ``baseline``
        # (the measured floor) rather than the old ``peak - n_streams``
        # heuristic.
        async def _reaped_to_baseline() -> bool:
            return (
                await count_active_backends(migrated_db_url, application_name=app_name) <= baseline
            )

        with contextlib.suppress(AssertionError):
            await wait_for_predicate(_reaped_to_baseline, max_wait_s=8.0, interval_s=0.1)
        cur = await count_active_backends(migrated_db_url, application_name=app_name)
        if cur > baseline:
            # Unfiltered diagnostic: ALL client backends, not just tagged
            # listeners — so this snapshot won't match the filtered counts
            # above.  Printed only on failure to aid debugging.
            states = await _backend_states(migrated_db_url)
            print("\n=== ALL BACKENDS (unfiltered diagnostic) ===")
            for s in states:
                age = s["age_s"]
                state_age = s["state_age_s"]
                print(
                    f"pid={s['pid']} state={s['state']} "
                    f"age={age:.1f}s state_age={state_age:.1f}s "
                    f"wait={s['wait_event_type']}/{s['wait_event']} "
                    f"query={(s['query'] or '')[:120]!r}"
                )
            print(f"=== app_name={app_name} baseline={baseline} peak={peak} current={cur} ===\n")
        assert cur <= baseline, (
            f"listener backends did not reap to baseline within 8s: "
            f"app_name={app_name} baseline={baseline} peak={peak} "
            f"current={cur} (expected drop of {n_streams})"
        )
