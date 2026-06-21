"""Unit tests for the session await done-predicates over ``await_completion``.

Pure: a real ``asyncio.Queue`` + scripted ``read_state``/``is_done``, no DB — the two
monotonic session predicates (request_id correlation, now the unified awaiter's session
arm; reacted>=watermark, the quiescence alias) exercised against the shared
:func:`aios.services.await_completion.await_completion` loop without standing up a session row.

Mirrors ``tests/unit/test_await_completion.py``: covers the three exit shapes
(already-done, done-after-notify, timeout) for each mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from aios.services.await_completion import await_completion

# ─── Mode 1: request_id correlation (is_done = state is not None) ─────────────


async def test_mode1_pending_then_response_done() -> None:
    """A notify drives a re-read; the loop stops once a response (non-None) lands."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    it: Iterator[dict[str, Any] | None] = iter(
        [None, {"result": 42, "is_error": False, "error": None}]
    )

    async def read_state() -> dict[str, Any] | None:
        return next(it)

    queue.put_nowait("evt")  # one notify so the loop advances to the second read
    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s is not None, timeout_seconds=5
    )
    assert state == {"result": 42, "is_error": False, "error": None}


async def test_mode1_already_responded_returns_immediately() -> None:
    """A first read that already has the response returns without touching the queue."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def read_state() -> dict[str, Any] | None:
        return {"result": 7, "is_error": False, "error": None}

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s is not None, timeout_seconds=5
    )
    assert state == {"result": 7, "is_error": False, "error": None}
    assert queue.empty()


async def test_mode1_pending_times_out() -> None:
    """Never-responded + empty queue → return the last (None) state on timeout."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def read_state() -> dict[str, Any] | None:
        return None

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s is not None, timeout_seconds=0.05
    )
    assert state is None


# ─── Mode 2: watermark (is_done = reacted >= watermark) ──────────────────────


async def test_mode2_reacted_reaches_watermark() -> None:
    """A notify drives a re-read; the loop stops once reacted reaches the watermark."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    it: Iterator[int] = iter([3, 5])

    async def read_state() -> int:
        return next(it)

    queue.put_nowait("evt")
    state = await await_completion(
        queue, read_state=read_state, is_done=lambda r: r >= 5, timeout_seconds=5
    )
    assert state == 5


async def test_mode2_already_at_watermark() -> None:
    """A first read already at/above the watermark returns without touching the queue."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def read_state() -> int:
        return 5

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda r: r >= 5, timeout_seconds=5
    )
    assert state == 5
    assert queue.empty()


async def test_mode2_blocks_below_watermark_times_out() -> None:
    """reacted stuck below the watermark + empty queue → return the last value on timeout."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def read_state() -> int:
        return 3

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda r: r >= 5, timeout_seconds=0.05
    )
    assert state == 3
