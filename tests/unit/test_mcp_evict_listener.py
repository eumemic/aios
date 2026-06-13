"""Unit coverage for :func:`aios.harness.worker._run_mcp_evict_listener`.

The MCP-pool eviction listener is one long-lived background task in the
worker. An operator credential mutation fires a NOTIFY on
``aios_mcp_evict_vault`` (from the API process); this listener drains it
and calls :meth:`McpSessionPool.evict_by_vault` so the rotated secret
propagates immediately instead of waiting out the idle TTL (#1030).

It pins the same survivability contract as the interrupt listener: a
per-payload dispatch exception is isolated (try INSIDE while True) and a
LISTEN-connection failure escapes to the outer reconnect loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import runtime
from aios.harness.worker import _run_mcp_evict_listener
from aios.mcp.pool import McpSessionPool

pytestmark = pytest.mark.asyncio


@pytest.fixture
def _restore_runtime_pool() -> Iterator[None]:
    saved = runtime.mcp_session_pool
    yield
    runtime.mcp_session_pool = saved


async def test_listener_evicts_on_notify(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_pool: None
) -> None:
    """A vault_id payload drives ``evict_by_vault(vault_id)`` on the pool."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_mcp_evict_vault", fake_listen)

    pool = MagicMock(spec=McpSessionPool)
    evicted = asyncio.Event()

    async def evict_by_vault(_vault_id: str) -> None:
        evicted.set()

    pool.evict_by_vault.side_effect = evict_by_vault
    runtime.mcp_session_pool = pool

    task = asyncio.create_task(_run_mcp_evict_listener("postgresql://stub"))
    try:
        await queue.put("vlt_rotated")
        await asyncio.wait_for(evicted.wait(), timeout=1.0)
        pool.evict_by_vault.assert_awaited_with("vlt_rotated")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_noop_when_pool_unset(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_pool: None
) -> None:
    """If the pool global isn't set yet, a NOTIFY is a harmless no-op and the
    listener survives to process the next one."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_mcp_evict_vault", fake_listen)
    runtime.mcp_session_pool = None

    pool = MagicMock(spec=McpSessionPool)
    second_evicted = asyncio.Event()

    async def evict_by_vault(_vault_id: str) -> None:
        second_evicted.set()

    pool.evict_by_vault.side_effect = evict_by_vault

    task = asyncio.create_task(_run_mcp_evict_listener("postgresql://stub"))
    try:
        # First NOTIFY arrives while the pool is None — must not crash.
        await queue.put("vlt_early")
        await asyncio.sleep(0)
        # Now the pool is up; the next NOTIFY dispatches.
        runtime.mcp_session_pool = pool
        await queue.put("vlt_later")
        await asyncio.wait_for(second_evicted.wait(), timeout=1.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_survives_dispatch_exception(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_pool: None
) -> None:
    """One exception in dispatch must not disable the listener."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_mcp_evict_vault", fake_listen)

    pool = MagicMock(spec=McpSessionPool)
    first = asyncio.Event()
    second = asyncio.Event()
    calls = 0

    async def evict_by_vault(_vault_id: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            first.set()
            raise RuntimeError("simulated transient evict failure")
        second.set()

    pool.evict_by_vault.side_effect = evict_by_vault
    runtime.mcp_session_pool = pool

    task = asyncio.create_task(_run_mcp_evict_listener("postgresql://stub"))
    try:
        await queue.put("vlt_one")
        await asyncio.wait_for(first.wait(), timeout=1.0)
        await queue.put("vlt_two")
        try:
            await asyncio.wait_for(second.wait(), timeout=1.0)
        except TimeoutError as exc:
            raise AssertionError(
                "listener did not process second eviction — it died on the first"
            ) from exc
        assert calls == 2
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_reconnects_after_termination(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_pool: None
) -> None:
    """An empty-string payload (termination sentinel) tears down the inner loop
    and the outer loop re-enters LISTEN."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    attempts = 0
    second_entered = asyncio.Event()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            second_entered.set()
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_mcp_evict_vault", fake_listen)
    monkeypatch.setattr("aios.harness.worker._LISTEN_RECONNECT_BACKOFF_SECONDS", 0)
    runtime.mcp_session_pool = MagicMock(spec=McpSessionPool)
    runtime.mcp_session_pool.evict_by_vault = AsyncMock()

    task = asyncio.create_task(_run_mcp_evict_listener("postgresql://stub"))
    try:
        await queue.put("")  # termination sentinel
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        assert attempts == 2
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
