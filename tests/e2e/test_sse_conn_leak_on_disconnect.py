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
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import httpx
import pytest

from tests.conftest import needs_docker
from tests.e2e.conftest import live_aios_server, wait_for_predicate
from tests.helpers.db import count_active_backends
from tests.integration.conftest import seed_agent_env_session


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

        # Baseline AFTER session is seeded so the seed-helper pool's
        # lifecycle is out of the way.
        await asyncio.sleep(0.2)
        baseline = await count_active_backends(migrated_db_url)

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

            # Wait for the SSE backends to attach.  Each open stream
            # adds one ``listen_for_events`` backend; the pool may add
            # more on backfill ``pool.acquire()`` calls if it grew
            # lazily.  Poll until we see at least n_streams new client
            # backends.
            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                peak = await count_active_backends(migrated_db_url)
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

        # After cleanup, the N dedicated listener backends must be reaped.
        # The api process's pool may keep up to pool_max_size pool conns
        # idle for reuse — those aren't leaks, just capacity.  Assert on
        # the LISTENER reduction (peak - n_streams) rather than absolute
        # baseline so the test is robust to pool growth during peak.
        target = peak - n_streams

        async def _reaped_to_target() -> bool:
            return await count_active_backends(migrated_db_url) <= target

        with contextlib.suppress(AssertionError):
            await wait_for_predicate(_reaped_to_target, max_wait_s=8.0, interval_s=0.1)
        cur = await count_active_backends(migrated_db_url)
        if cur > target:
            states = await _backend_states(migrated_db_url)
            print("\n=== LEAKED BACKENDS ===")
            for s in states:
                age = s["age_s"]
                state_age = s["state_age_s"]
                print(
                    f"pid={s['pid']} state={s['state']} "
                    f"age={age:.1f}s state_age={state_age:.1f}s "
                    f"wait={s['wait_event_type']}/{s['wait_event']} "
                    f"query={(s['query'] or '')[:120]!r}"
                )
            print(f"=== baseline={baseline} peak={peak} current={cur} target={target} ===\n")
        assert cur <= target, (
            f"listener backends did not reap within 8s: peak={peak} "
            f"current={cur} target={target} (expected drop of {n_streams})"
        )
