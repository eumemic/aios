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
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.api.sse import runtime_connector_calls_stream
from aios.db.listen import ListenSubscription


def _mk_subscription() -> ListenSubscription:
    conn = MagicMock()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    return ListenSubscription(queue=queue, _conn=conn)


async def test_runtime_connector_calls_stream_terminates_conn_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the consumer task must trigger subscription.terminate()."""
    subscription = _mk_subscription()

    # Stub the backfill query to return empty so the generator immediately
    # enters the live-tail loop.
    monkeypatch.setattr(
        "aios.api.sse.queries.list_pending_calls_for_connector",
        AsyncMock(return_value=[]),
    )

    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)

    gen = runtime_connector_calls_stream(subscription, pool, "telegram", account_id="acct_X")

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


async def test_runtime_connector_calls_stream_terminates_conn_when_backfill_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backfill query failure must still terminate the subscription."""
    subscription = _mk_subscription()

    monkeypatch.setattr(
        "aios.api.sse.queries.list_pending_calls_for_connector",
        AsyncMock(side_effect=RuntimeError("backfill kaboom")),
    )

    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)

    gen = runtime_connector_calls_stream(subscription, pool, "telegram", account_id="acct_X")

    async def _consume() -> None:
        async for _ in gen:
            pass

    with pytest.raises(RuntimeError, match="backfill kaboom"):
        await _consume()

    subscription._conn.terminate.assert_called_once()
