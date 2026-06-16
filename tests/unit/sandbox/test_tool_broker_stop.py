"""Unit tests for :meth:`ToolBroker.stop` drain concurrency.

The broker runs two uvicorn servers (TCP + UDS). Each pays uvicorn's
~150-200 ms fixed shutdown floor once ``should_exit`` is set. Draining
them sequentially (set TCP flag, await TCP, *then* set UDS flag, await
UDS) doubles that floor on every worker shutdown and across the unit
suite. ``stop`` must set both flags first, then await the two serve
tasks concurrently — while still running the finally-block cleanup for
both tasks.
"""

from __future__ import annotations

import asyncio

from aios.sandbox.tool_broker import ToolBroker


class _FakeServer:
    """Minimal uvicorn.Server stand-in.

    Records the wall-clock order in which ``should_exit`` is set on it
    (relative to its sibling) so the test can assert both flags flip
    before either drain wait resolves. The flag setter also fires an
    :class:`asyncio.Event` the serve task waits on, so the drain doesn't
    busy-poll.
    """

    def __init__(self, log: list[str], name: str) -> None:
        self.started = True
        self._should_exit = False
        self._log = log
        self._name = name
        self.exit_requested = asyncio.Event()

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if value:
            self._log.append(f"flag:{self._name}")
            self.exit_requested.set()


def _drain_task(server: _FakeServer, log: list[str], name: str, delay: float) -> asyncio.Task[None]:
    """A serve task that only completes after ``should_exit`` is set.

    Sleeps ``delay`` after the flag flips to model uvicorn's fixed
    shutdown floor, then records its completion.
    """

    async def _serve() -> None:
        await server.exit_requested.wait()
        await asyncio.sleep(delay)
        log.append(f"done:{name}")

    return asyncio.create_task(_serve())


class TestStopDrainsConcurrently:
    async def test_both_flags_set_before_either_drain_completes(self) -> None:
        log: list[str] = []
        broker = ToolBroker()

        tcp = _FakeServer(log, "tcp")
        uds = _FakeServer(log, "uds")
        broker._server = tcp  # type: ignore[assignment]
        broker._uds_server = uds  # type: ignore[assignment]
        broker._serve_task = _drain_task(tcp, log, "tcp", delay=0.05)
        broker._uds_serve_task = _drain_task(uds, log, "uds", delay=0.05)

        await broker.stop()

        # Both should_exit flags must be flipped before either serve task
        # is allowed to finish draining — i.e. the two shutdown floors
        # overlap rather than serialize.
        first_done = next(i for i, e in enumerate(log) if e.startswith("done:"))
        flags_before = {e for e in log[:first_done] if e.startswith("flag:")}
        assert flags_before == {"flag:tcp", "flag:uds"}, log

    async def test_stop_completes_faster_than_sequential_floor(self) -> None:
        """Concurrent drain wall-time ≈ one floor, not two."""
        log: list[str] = []
        broker = ToolBroker()

        tcp = _FakeServer(log, "tcp")
        uds = _FakeServer(log, "uds")
        broker._server = tcp  # type: ignore[assignment]
        broker._uds_server = uds  # type: ignore[assignment]
        floor = 0.15
        broker._serve_task = _drain_task(tcp, log, "tcp", delay=floor)
        broker._uds_serve_task = _drain_task(uds, log, "uds", delay=floor)

        loop = asyncio.get_running_loop()
        start = loop.time()
        await broker.stop()
        elapsed = loop.time() - start

        # Sequential would be ~2*floor; concurrent stays well under it.
        assert elapsed < floor * 1.8, elapsed

    async def test_both_tasks_finalized_when_uds_absent(self) -> None:
        """TCP-only broker (no UDS) still drains cleanly."""
        log: list[str] = []
        broker = ToolBroker()

        tcp = _FakeServer(log, "tcp")
        broker._server = tcp  # type: ignore[assignment]
        broker._serve_task = _drain_task(tcp, log, "tcp", delay=0.02)

        await broker.stop()

        assert "done:tcp" in log
        assert broker._serve_task is not None
        assert broker._serve_task.done()
