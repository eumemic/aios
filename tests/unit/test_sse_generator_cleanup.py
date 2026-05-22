"""Unit tests for SSE generator cleanup contracts (issue #376).

After the preflight refactor, the four SSE generators in ``aios.api.sse``
accept a pre-established :class:`aios.db.listen.ListenSubscription`
instead of opening their own LISTEN connection.  Each generator now
owns terminating the subscription via a ``try/finally`` around the
body — under client disconnect (asyncio.CancelledError) AND under
mid-body exceptions (e.g. the backfill query raising during a
Postgres outage).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.api.sse import (
    connection_discovery_stream,
    management_calls_stream,
    runtime_connector_calls_stream,
    sse_event_stream,
)
from aios.db.listen import ListenSubscription


def _mk_subscription() -> ListenSubscription:
    conn = MagicMock()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    return ListenSubscription(queue=queue, _conn=conn)


def _mk_pool() -> MagicMock:
    """Mock pool whose ``async with pool.acquire()`` yields a fresh MagicMock conn."""
    conn = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


def _install_query_backfill(
    monkeypatch: pytest.MonkeyPatch,
    attr: str,
    pool: MagicMock,
    exc: Exception | None,
) -> None:
    """Stub a ``aios.api.sse.queries.<attr>`` backfill function."""
    if exc is None:
        monkeypatch.setattr(f"aios.api.sse.queries.{attr}", AsyncMock(return_value=[]))
    else:
        monkeypatch.setattr(f"aios.api.sse.queries.{attr}", AsyncMock(side_effect=exc))


def _install_session_event_backfill(
    monkeypatch: pytest.MonkeyPatch,
    pool: MagicMock,
    exc: Exception | None,
) -> None:
    """``sse_event_stream`` backfills via ``conn.fetch`` directly (no queries
    module helper), so patch the pool-acquired conn's ``fetch`` method.
    """
    conn = MagicMock()
    if exc is None:
        conn.fetch = AsyncMock(return_value=[])
    else:
        conn.fetch = AsyncMock(side_effect=exc)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)


# Each case: (build_generator, install_backfill).
#  - build_generator(subscription, pool) returns the async generator under test.
#  - install_backfill(monkeypatch, pool, exc) wires up the backfill path:
#       exc is None  -> empty list / empty fetch (falls through to live tail)
#       exc is Exception -> backfill raises that exception
_CASES = [
    pytest.param(
        lambda sub, pool: sse_event_stream(sub, pool, "ses_X", after_seq=0),
        lambda mp, pool, exc: _install_session_event_backfill(mp, pool, exc),
        id="sse_event_stream",
    ),
    pytest.param(
        lambda sub, pool: runtime_connector_calls_stream(
            sub, pool, "telegram", account_id="acct_X"
        ),
        lambda mp, pool, exc: _install_query_backfill(
            mp, "list_pending_calls_for_connector", pool, exc
        ),
        id="runtime_connector_calls_stream",
    ),
    pytest.param(
        lambda sub, pool: management_calls_stream(sub, pool, "telegram", account_id="acct_X"),
        lambda mp, pool, exc: _install_query_backfill(
            mp, "list_pending_management_calls_for_connector", pool, exc
        ),
        id="management_calls_stream",
    ),
    pytest.param(
        lambda sub, pool: connection_discovery_stream(sub, pool, "telegram", account_id="acct_X"),
        lambda mp, pool, exc: _install_query_backfill(mp, "list_connections", pool, exc),
        id="connection_discovery_stream",
    ),
]


@pytest.mark.parametrize("build_gen, install_backfill", _CASES)
async def test_generator_terminates_conn_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    build_gen: Any,
    install_backfill: Any,
) -> None:
    """Cancelling the consumer task must trigger subscription.terminate()."""
    subscription = _mk_subscription()
    pool = _mk_pool()
    install_backfill(monkeypatch, pool, None)

    gen = build_gen(subscription, pool)

    async def _consume() -> None:
        async for _ in gen:
            pass

    task = asyncio.create_task(_consume())
    # Yield control so the generator enters the live-tail ``await queue.get()``.
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    subscription._conn.terminate.assert_called_once()


@pytest.mark.parametrize("build_gen, install_backfill", _CASES)
async def test_generator_terminates_conn_when_backfill_raises(
    monkeypatch: pytest.MonkeyPatch,
    build_gen: Any,
    install_backfill: Any,
) -> None:
    """Backfill query failure must still terminate the subscription."""
    subscription = _mk_subscription()
    pool = _mk_pool()
    install_backfill(monkeypatch, pool, RuntimeError("backfill kaboom"))

    gen = build_gen(subscription, pool)

    async def _consume() -> None:
        async for _ in gen:
            pass

    with pytest.raises(RuntimeError, match="backfill kaboom"):
        await _consume()

    subscription._conn.terminate.assert_called_once()
