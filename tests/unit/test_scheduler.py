"""Unit tests for ``aios.harness.scheduler`` (#940).

Two properties this pins:

1. ``_compute_sleep_seconds`` clamps the next-wake delay to
   ``[0, _HEARTBEAT_SECONDS]`` — a past-due ``next_fire`` yields ``0`` so the
   startup iteration claims it immediately (the "startup reconcile" the plan
   relies on instead of new code); ``None`` and far-future values cap at the
   heartbeat.
2. ``_scheduler_loop``'s FIRST iteration claims due rows WITHOUT any NOTIFY —
   the heartbeat lowering (#940) is a safety net, not the primary bound, and
   past-due rows present at worker startup fire on the first pass.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness import scheduler


def _fake_pool() -> MagicMock:
    """Pool whose ``acquire()`` returns a context-managed MagicMock conn.

    Mirrors the ``fake_pool`` idiom in ``tests/unit/test_attachment_gc.py``.
    """
    pool = MagicMock()

    class _AsyncCm:
        async def __aenter__(self) -> object:
            return MagicMock()

        async def __aexit__(self, *_args: object) -> None:
            return None

    pool.acquire.return_value = _AsyncCm()
    return pool


async def test_compute_sleep_seconds_clamps_past_due_to_zero() -> None:
    fake_pool = _fake_pool()
    with patch(
        "aios.harness.scheduler.queries.fetch_next_trigger_event",
        AsyncMock(return_value=datetime.now(UTC) - timedelta(minutes=5)),
    ):
        assert await scheduler._compute_sleep_seconds(fake_pool) == 0.0


async def test_compute_sleep_seconds_none_returns_heartbeat() -> None:
    fake_pool = _fake_pool()
    with patch(
        "aios.harness.scheduler.queries.fetch_next_trigger_event",
        AsyncMock(return_value=None),
    ):
        assert await scheduler._compute_sleep_seconds(fake_pool) == scheduler._HEARTBEAT_SECONDS


async def test_compute_sleep_seconds_caps_far_future_at_heartbeat() -> None:
    fake_pool = _fake_pool()
    with patch(
        "aios.harness.scheduler.queries.fetch_next_trigger_event",
        AsyncMock(return_value=datetime.now(UTC) + timedelta(days=1)),
    ):
        assert await scheduler._compute_sleep_seconds(fake_pool) == scheduler._HEARTBEAT_SECONDS


async def test_compute_sleep_seconds_returns_delta_for_near_future() -> None:
    fake_pool = _fake_pool()
    with patch(
        "aios.harness.scheduler.queries.fetch_next_trigger_event",
        AsyncMock(return_value=datetime.now(UTC) + timedelta(seconds=120)),
    ):
        result = await scheduler._compute_sleep_seconds(fake_pool)
    # Allow a small tolerance below 120 for the wall-clock spent in the call.
    assert 118 <= result <= 120


async def test_heartbeat_is_five_minutes() -> None:
    assert scheduler._HEARTBEAT_SECONDS == 300.0


async def test_scheduler_loop_first_iteration_claims_without_notify() -> None:
    """Startup-reconcile pin: the loop's first iteration claims due rows even
    though the NOTIFY event is NEVER set — proving past-due rows present at
    worker startup fire without waiting for a NOTIFY or the heartbeat."""
    fake_pool = _fake_pool()
    notify_event = asyncio.Event()  # never set

    calls: list[object] = []

    async def _record_then_cancel(_pool: object) -> None:
        calls.append(_pool)
        raise asyncio.CancelledError

    with (
        patch.object(scheduler, "_compute_sleep_seconds", AsyncMock(return_value=0.0)),
        patch.object(
            scheduler,
            "_claim_and_enqueue_due_triggers",
            AsyncMock(side_effect=_record_then_cancel),
        ),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await scheduler._scheduler_loop(fake_pool, notify_event)

    assert len(calls) >= 1
    assert not notify_event.is_set()
