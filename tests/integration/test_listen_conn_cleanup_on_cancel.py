"""Sanity test that the listener helpers don't leak under raw task cancel.

The 6 listener context managers in ``src/aios/db/listen.py`` cleanup via
``conn.terminate()`` (synchronous transport-abort, never interrupted by
cancellation).  Under raw ``asyncio.Task.cancel()`` the old
``await conn.close()`` cleanup also worked — asyncpg's own ``close()``
catches ``CancelledError`` and falls back to ``terminate()``.

The production leak (#606) only manifests under sse-starlette's *anyio*
task-group scope-cancellation, where every subsequent await raises
``CancelledError`` immediately.  That stack is exercised by
``tests/e2e/test_sse_conn_leak_on_disconnect.py``.  This integration
test stays useful as a low-friction guard that the listener helpers
themselves remain leak-free under plain task cancellation.

Issue #606.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

import asyncpg
import pytest

from aios.db.listen import (
    listen_for_connection_discovery,
    listen_for_connector_calls_by_type,
    listen_for_connector_result,
    listen_for_events,
    listen_for_management_calls,
    listen_for_session_interrupts,
)
from tests.conftest import needs_docker

pytestmark = [pytest.mark.integration, needs_docker]


# One adapter per listener; each takes the db url and returns the
# listener's async context manager.  The yield value (queue type) is
# uniformly ``asyncio.Queue[str]`` so the consumer below can block on
# ``queue.get()`` regardless of which listener is under test.
ListenerFactory = Callable[[str], "contextlib.AbstractAsyncContextManager[asyncio.Queue[str]]"]

LISTENERS: dict[str, ListenerFactory] = {
    "events": lambda db_url: listen_for_events(db_url, "sess_test_leak"),
    "connector_result": lambda db_url: listen_for_connector_result(db_url, "call_test_leak"),
    "connector_calls_by_type": lambda db_url: listen_for_connector_calls_by_type(
        db_url, "test_leak"
    ),
    "management_calls": lambda db_url: listen_for_management_calls(db_url, "test_leak"),
    "connection_discovery": lambda db_url: listen_for_connection_discovery(db_url, "test_leak"),
    "session_interrupts": lambda db_url: listen_for_session_interrupts(db_url),
}


async def _backend_count(db_url: str) -> int:
    """Count this database's client backends, excluding our own measurement conn."""
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.fetchval(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
        return int(result)
    finally:
        await conn.close()


@pytest.mark.parametrize("listener_name", list(LISTENERS.keys()))
async def test_listener_cancellation_does_not_leak_pg_conn(
    migrated_db_url: str,
    _reset_db_state: object,
    listener_name: str,
) -> None:
    factory = LISTENERS[listener_name]

    async def _hold(ready: asyncio.Event) -> None:
        async with factory(migrated_db_url) as queue:
            ready.set()
            await queue.get()  # blocks indefinitely until the task is cancelled

    baseline = await _backend_count(migrated_db_url)

    n_streams = 5
    readies = [asyncio.Event() for _ in range(n_streams)]
    tasks = [asyncio.create_task(_hold(r)) for r in readies]

    # Wait for every listener's LISTEN to be in place.
    await asyncio.wait_for(asyncio.gather(*(r.wait() for r in readies)), timeout=5.0)

    peak = await _backend_count(migrated_db_url)
    assert peak >= baseline + n_streams, (
        f"only {peak - baseline}/{n_streams} backends observed during peak; "
        f"the LISTENs may not be using dedicated conns"
    )

    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t

    # Give Postgres a small grace window for Terminate / TCP close to
    # propagate.  pg_stat_activity is roughly real-time but can lag.
    deadline = asyncio.get_running_loop().time() + 5.0
    while True:
        cur = await _backend_count(migrated_db_url)
        if cur <= baseline:
            return
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(
                f"backend count did not return to baseline within 5s: "
                f"listener={listener_name} baseline={baseline} peak={peak} current={cur}"
            )
        await asyncio.sleep(0.1)
