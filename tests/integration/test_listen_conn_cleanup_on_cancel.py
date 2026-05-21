"""Sanity guard that the listener helpers don't leak Postgres backends
under raw asyncio task cancellation.

The high-fidelity leak repro lives in ``tests/e2e/`` and exercises the
sse-starlette anyio scope-cancellation path that motivated the
``conn.terminate()`` cleanup.  This test covers the plain-asyncio path
in isolation, so a regression that re-introduces an async cleanup step
fails here before reaching e2e.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

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
from tests.helpers.db import count_active_backends

pytestmark = [pytest.mark.integration, needs_docker]


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
            await queue.get()

    baseline = await count_active_backends(migrated_db_url)

    n_streams = 5
    readies = [asyncio.Event() for _ in range(n_streams)]
    tasks = [asyncio.create_task(_hold(r)) for r in readies]

    await asyncio.wait_for(asyncio.gather(*(r.wait() for r in readies)), timeout=5.0)

    peak = await count_active_backends(migrated_db_url)
    assert peak >= baseline + n_streams, (
        f"only {peak - baseline}/{n_streams} backends observed during peak; "
        f"the LISTENs may not be using dedicated conns"
    )

    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t

    deadline = asyncio.get_running_loop().time() + 5.0
    while True:
        cur = await count_active_backends(migrated_db_url)
        if cur <= baseline:
            return
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(
                f"backend count did not return to baseline within 5s: "
                f"listener={listener_name} baseline={baseline} peak={peak} current={cur}"
            )
        await asyncio.sleep(0.1)
