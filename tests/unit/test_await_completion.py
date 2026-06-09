"""Unit tests for ``await_completion`` — the await-a-completion loop.

Pure: a real ``asyncio.Queue`` + scripted ``read_state``/``is_done``, no DB. Covers the
three exit shapes (already-done, done-after-notify, timeout) and the LISTEN-before-read
ordering the run/session backings rely on.
"""

from __future__ import annotations

import asyncio

from aios.services.await_completion import await_completion


async def test_returns_immediately_when_already_done() -> None:
    """A first read that already satisfies the predicate returns without ever consuming the
    queue — which also proves the read happens BEFORE any ``queue.get`` (LISTEN-before-read):
    an empty queue + a 5s budget could not return this fast had it waited on the queue first."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    reads = 0

    async def read_state() -> str:
        nonlocal reads
        reads += 1
        return "completed"

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s == "completed", timeout_seconds=5
    )
    assert state == "completed"
    assert reads == 1
    assert queue.empty()


async def test_returns_terminal_state_after_a_notify() -> None:
    """A notify drives a re-read; the loop stops the first time the predicate holds."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    states = iter(["running", "completed"])

    async def read_state() -> str:
        return next(states)

    queue.put_nowait("evt_1")  # one notify so the loop advances to the second read
    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s == "completed", timeout_seconds=5
    )
    assert state == "completed"


async def test_returns_current_state_on_timeout() -> None:
    """Never-done + empty queue → return the last (non-terminal) state once the deadline
    passes — the caller's signal to re-poll."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def read_state() -> str:
        return "running"

    state = await await_completion(
        queue, read_state=read_state, is_done=lambda s: s == "completed", timeout_seconds=0.05
    )
    assert state == "running"
