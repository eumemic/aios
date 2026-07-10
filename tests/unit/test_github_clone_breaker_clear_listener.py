"""Unit coverage for
:func:`aios.harness.worker._run_github_clone_breaker_clear_listener` (#1720).

A token rotation runs in the API process and NOTIFYs
``aios_github_clone_breaker_clear`` with the ``ghrepo_...`` resource id; the
worker owns ``runtime.github_clone_breaker`` so it LISTENs here and calls
:meth:`GithubCloneBreaker.clear` — a fixed credential re-probes on the next
provision instead of serving out a cooldown opened under the old secret.

Mirrors ``test_mcp_evict_listener.py``: it pins the same survivability
contract (a per-payload dispatch exception is isolated INSIDE ``while True``;
an empty-string termination sentinel escapes to the outer reconnect loop).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from aios.harness import runtime
from aios.harness.worker import _run_github_clone_breaker_clear_listener
from aios.sandbox.github_clone_breaker import GithubCloneBreaker

pytestmark = pytest.mark.asyncio


@pytest.fixture
def _restore_runtime_breaker() -> Iterator[None]:
    saved = runtime.github_clone_breaker
    yield
    runtime.github_clone_breaker = saved


async def test_listener_clears_breaker_on_notify(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_breaker: None
) -> None:
    """A resource_id payload drives ``clear(resource_id)`` on the breaker."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_github_clone_breaker_clear", fake_listen)

    breaker = MagicMock(spec=GithubCloneBreaker)
    cleared = asyncio.Event()

    def clear(_resource_id: str) -> None:
        cleared.set()

    breaker.clear.side_effect = clear
    runtime.github_clone_breaker = breaker

    task = asyncio.create_task(_run_github_clone_breaker_clear_listener("postgresql://stub"))
    try:
        await queue.put("ghrepo_rotated")
        await asyncio.wait_for(cleared.wait(), timeout=1.0)
        breaker.clear.assert_called_with("ghrepo_rotated")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_noop_when_breaker_unset(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_breaker: None
) -> None:
    """If the breaker global isn't set yet (startup race), a NOTIFY is a
    harmless no-op and the listener survives to process the next one."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_github_clone_breaker_clear", fake_listen)
    runtime.github_clone_breaker = None

    breaker = MagicMock(spec=GithubCloneBreaker)
    second_cleared = asyncio.Event()
    breaker.clear.side_effect = lambda _rid: second_cleared.set()

    task = asyncio.create_task(_run_github_clone_breaker_clear_listener("postgresql://stub"))
    try:
        # First NOTIFY arrives while the breaker is None — must not crash.
        await queue.put("ghrepo_early")
        await asyncio.sleep(0)
        # Now the breaker is up; the next NOTIFY dispatches.
        runtime.github_clone_breaker = breaker
        await queue.put("ghrepo_later")
        await asyncio.wait_for(second_cleared.wait(), timeout=1.0)
        breaker.clear.assert_called_with("ghrepo_later")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_survives_dispatch_exception(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_breaker: None
) -> None:
    """One exception in dispatch must not disable the listener."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    @asynccontextmanager
    async def fake_listen(_db_url: str) -> AsyncIterator[asyncio.Queue[str]]:
        yield queue

    monkeypatch.setattr("aios.harness.worker.listen_for_github_clone_breaker_clear", fake_listen)

    breaker = MagicMock(spec=GithubCloneBreaker)
    first = asyncio.Event()
    second = asyncio.Event()
    calls = 0

    def clear(_resource_id: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            first.set()
            raise RuntimeError("simulated transient clear failure")
        second.set()

    breaker.clear.side_effect = clear
    runtime.github_clone_breaker = breaker

    task = asyncio.create_task(_run_github_clone_breaker_clear_listener("postgresql://stub"))
    try:
        await queue.put("ghrepo_one")
        await asyncio.wait_for(first.wait(), timeout=1.0)
        await queue.put("ghrepo_two")
        try:
            await asyncio.wait_for(second.wait(), timeout=1.0)
        except TimeoutError as exc:
            raise AssertionError(
                "listener did not process second clear — it died on the first"
            ) from exc
        assert calls == 2
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_listener_reconnects_after_termination(
    monkeypatch: pytest.MonkeyPatch, _restore_runtime_breaker: None
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

    monkeypatch.setattr("aios.harness.worker.listen_for_github_clone_breaker_clear", fake_listen)
    monkeypatch.setattr("aios.harness.worker._LISTEN_RECONNECT_BACKOFF_SECONDS", 0)
    runtime.github_clone_breaker = MagicMock(spec=GithubCloneBreaker)

    task = asyncio.create_task(_run_github_clone_breaker_clear_listener("postgresql://stub"))
    try:
        await queue.put("")  # termination sentinel
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        assert attempts == 2
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
