"""Unit coverage for the single-sourced LISTEN-open lifecycle (issue #1079).

The five ``open_listen_for_*`` drop-oldest listeners and their four thin
``listen_for_*`` context-manager wrappers were collapsed onto one private
helper (:func:`aios.db.listen._open_drop_oldest_listener`) plus one generic
context manager (:func:`aios.db.listen._listen_subscription`). These tests pin
the single-sourced behavior: the channel template each public open uses, the
drop-oldest backpressure, and the #81 subscriber-lock ``on_connected`` hook.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db import listen


def _mock_conn() -> MagicMock:
    conn = MagicMock()
    conn.add_listener = AsyncMock()
    return conn


def _captured_callback(conn: MagicMock) -> Any:
    """Return the ``_callback`` passed to ``conn.add_listener(channel, cb)``."""
    assert conn.add_listener.await_count == 1
    return conn.add_listener.await_args.args[1]


@pytest.mark.parametrize(
    ("opener", "args", "expected_channel"),
    [
        (listen.open_listen_for_events, ("sess_X",), "events_sess_X"),
        (listen.open_listen_for_run_events, ("run_X",), "wf_run_events_run_X"),
        (
            listen.open_listen_for_connector_calls_by_type,
            ("telegram",),
            "connector_calls_telegram",
        ),
        (
            listen.open_listen_for_management_calls,
            ("telegram",),
            "connector_management_calls_telegram",
        ),
        (
            listen.open_listen_for_connection_discovery,
            ("telegram",),
            "connections_telegram",
        ),
    ],
)
async def test_open_listeners_listen_on_expected_channel(
    opener: Any, args: tuple[str, ...], expected_channel: str
) -> None:
    """Each public ``open_listen_for_*`` LISTENs on its own channel template."""
    conn = _mock_conn()
    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        patch("aios.db.listen.acquire_subscriber_lock", AsyncMock()),
    ):
        sub = await opener("postgresql://stub/aios", *args)
    assert conn.add_listener.await_args.args[0] == expected_channel
    sub.terminate()


async def test_drop_oldest_callback_evicts_oldest_on_overflow() -> None:
    """On QueueFull the callback drops the OLDEST payload and enqueues the new."""
    conn = _mock_conn()
    with patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)):
        sub = await listen.open_listen_for_run_events(
            "postgresql://stub/aios", "run_X", queue_max=2
        )
    cb = _captured_callback(conn)

    cb(conn, 1, "wf_run_events_run_X", "a")
    cb(conn, 1, "wf_run_events_run_X", "b")
    # Queue is now full (maxsize=2): the next put drops the oldest ("a").
    cb(conn, 1, "wf_run_events_run_X", "c")

    drained = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
    assert drained == ["b", "c"]


async def test_open_listen_for_events_acquires_subscriber_lock_by_default() -> None:
    """Default open path takes the #81 subscriber advisory lock on the conn."""
    conn = _mock_conn()
    acquire = AsyncMock()
    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        patch("aios.db.listen.acquire_subscriber_lock", acquire),
    ):
        sub = await listen.open_listen_for_events("postgresql://stub/aios", "sess_X")
    acquire.assert_awaited_once_with(conn, session_id="sess_X")
    sub.terminate()


async def test_open_listen_for_events_on_connected_none_omits_lock() -> None:
    """``on_connected=None`` (the await-poller's choice) skips the lock."""
    conn = _mock_conn()
    acquire = AsyncMock()
    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        patch("aios.db.listen.acquire_subscriber_lock", acquire),
    ):
        sub = await listen.open_listen_for_events(
            "postgresql://stub/aios", "sess_X", on_connected=None
        )
    acquire.assert_not_awaited()
    sub.terminate()


async def test_open_listen_for_run_events_does_not_acquire_lock() -> None:
    """Non-events openers never touch the subscriber lock."""
    conn = _mock_conn()
    acquire = AsyncMock()
    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        patch("aios.db.listen.acquire_subscriber_lock", acquire),
    ):
        sub = await listen.open_listen_for_run_events("postgresql://stub/aios", "run_X")
    acquire.assert_not_awaited()
    sub.terminate()


async def test_listen_subscription_wrapper_terminates_on_exit() -> None:
    """The generic ``listen_for_*`` wrapper terminates the conn on exit."""
    conn = _mock_conn()
    with patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)):
        async with listen.listen_for_connection_discovery(
            "postgresql://stub/aios", "telegram"
        ) as queue:
            assert isinstance(queue, asyncio.Queue)
    conn.terminate.assert_called_once()
