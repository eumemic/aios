"""The SIGKILL cleanup paths in ``_subprocess`` must release the
parent-side pipe FDs.

When a child is killed but its transport isn't closed — the
double-timeout give-up path, or an outer ``CancelledError`` that skips
the timeout branch — the cancelled ``communicate()`` strands the
parent-side stdout/stderr pipe FDs. Every container provision, host
git-clone, and snapshot flatten funnels through this module; pre-fix,
repeated wedges/cancellations exhaust the worker's FD ceiling and starve
both connection pools (and, for the ``docker export | docker import``
flatten pipeline, orphan multi-GB subprocesses). The mechanism is
documented at each fix site in ``_subprocess.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.sandbox import _subprocess
from aios.sandbox._subprocess import run_docker_pipeline, run_subprocess_with_timeout
from aios.sandbox.backends.base import SandboxBackendError


class _WedgedProc:
    """A child stuck in D-state: ``communicate()`` never completes and
    the pending SIGKILL from ``kill()`` never takes effect."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        # The sole non-blocking handle to the parent-side pipe FDs.
        # asyncio.subprocess.Process.kill() also delegates to the
        # transport, so route kill through the same mock — the fake
        # can't drift from Process's real kill/_transport coupling.
        self._transport = MagicMock(name="transport")
        self.stdin: Any = None
        self.stdout: Any = None
        self.stderr: Any = None

    def kill(self) -> None:
        self._transport.kill()

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)  # outlives both (tiny) timeouts
        raise AssertionError("unreachable: communicate must be cancelled")


async def test_give_up_path_closes_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outer drain times out → SIGKILL → inner drain *also* times out.

    The give-up branch must close ``proc._transport`` so the parent-side
    stdout/stderr pipe FDs are released; otherwise they leak for as long
    as the wedged child lives (effectively forever under a flapping
    daemon)."""
    proc = _WedgedProc()

    async def _fake_spawn(*_argv: Any, **_kwargs: Any) -> _WedgedProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)
    # Keep the secondary drain tiny so the test is fast and deterministic.
    monkeypatch.setattr(_subprocess, "_DRAIN_AFTER_KILL_TIMEOUT_S", 0.01)

    rc, out, err, timed_out = await run_subprocess_with_timeout(["docker", "ps"], timeout_s=0.01)

    # The (-1, b"", b"", True) tuple is only reachable after proc.kill()
    # on the SIGKILL path, so this also pins that we took the give-up
    # branch rather than the happy or drain-succeeded paths.
    assert (rc, out, err, timed_out) == (-1, b"", b"", True)

    proc._transport.close.assert_called_once()


async def test_outer_cancellation_kills_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer cancellation must SIGKILL the child and close the transport
    before propagating. ``CancelledError`` is a ``BaseException``, not a
    ``TimeoutError``, so it skips the give-up branch from #457."""
    proc = _WedgedProc()

    async def _fake_spawn(*_argv: Any, **_kwargs: Any) -> _WedgedProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    # Generous timeout so the wait_for itself never fires — outer
    # cancellation is what we're testing.
    task = asyncio.create_task(run_subprocess_with_timeout(["docker", "ps"], timeout_s=60.0))
    # Let the task reach `await asyncio.wait_for(proc.communicate(), ...)`.
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert proc._transport.kill.called
    proc._transport.close.assert_called_once()


async def test_pipeline_outer_cancel_kills_and_closes_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer cancellation of the flatten pipeline must SIGKILL *both*
    docker children and close *both* pipe transports before propagating.

    ``CancelledError`` is a ``BaseException``, not a ``TimeoutError``, so
    it skips ``run_docker_pipeline``'s timeout branch — the same gap the
    sibling ``run_subprocess_with_timeout`` closes. Without it, a worker
    SIGTERM or job-deadline cancel mid ``docker export | docker import``
    leaves both children running and strands their parent-side pipe
    FDs."""

    async def _hang(*_args: Any) -> bytes:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    producer = _WedgedProc()
    producer.stdout = MagicMock()
    producer.stdout.read = AsyncMock(side_effect=_hang)
    producer.stderr = MagicMock()
    producer.stderr.read = AsyncMock(side_effect=_hang)
    consumer = _WedgedProc()
    consumer.stdin = MagicMock()
    consumer.stdin.drain = AsyncMock()
    consumer.stdout = MagicMock()
    consumer.stdout.read = AsyncMock(side_effect=_hang)
    consumer.stderr = MagicMock()
    consumer.stderr.read = AsyncMock(side_effect=_hang)
    spawned = iter((producer, consumer))

    async def _fake_spawn(*_argv: Any, **_kwargs: Any) -> _WedgedProc:
        return next(spawned)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    # Generous deadlines so neither timeout fires — outer cancellation is
    # what we're testing.
    task = asyncio.create_task(
        run_docker_pipeline(
            ["docker", "export", "x"],
            ["docker", "import", "-", "tag"],
            stall_timeout_s=60.0,
            max_timeout_s=60.0,
        )
    )
    # Let the task reach `await asyncio.wait_for(_drain(), ...)`.
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert producer._transport.kill.called
    assert consumer._transport.kill.called
    producer._transport.close.assert_called_once()
    consumer._transport.close.assert_called_once()


# ── progress-based deadlines (the flatten-brick fix) ─────────────────────────
#
# The flatten pipeline (``docker export | docker import``) replaced a single
# size-scaled timeout — which a genuinely large writable layer could never
# beat, permanently bricking salvage — with two progress-based deadlines: a
# per-op *stall* bound (no bytes moving) and an *absolute* ceiling on the whole
# relay. BOTH the read side (producer withholding bytes) AND the write side
# (consumer refusing to drain them) must be under those deadlines, or a wedged
# ``docker import`` fills the pipe and the relay hangs forever with neither
# deadline able to fire.


def _pipe_proc() -> _WedgedProc:
    """A pipeline child stub: real ``_transport``/``kill`` coupling (reused
    from ``_WedgedProc``) plus per-stream mocks the caller fills in."""
    proc = _WedgedProc()
    proc.returncode = 0
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.wait_closed = AsyncMock()
    proc.stdout = MagicMock()
    proc.stdout.read = AsyncMock(return_value=b"")
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)  # type: ignore[attr-defined]
    return proc


def _spawn_pair(
    monkeypatch: pytest.MonkeyPatch, producer: _WedgedProc, consumer: _WedgedProc
) -> None:
    spawned = iter((producer, consumer))

    async def _fake_spawn(*_argv: Any, **_kwargs: Any) -> _WedgedProc:
        return next(spawned)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)


async def test_pipeline_consumer_backpressure_trips_write_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer that stops draining must trip the stall deadline on the
    WRITE side, not hang.

    This is the regression that exposes the half-fixed brick: the producer
    keeps yielding bytes (reads succeed), but ``docker import`` stops
    consuming stdin, so ``drain()`` blocks on backpressure. If only the read
    side is bounded, the whole pipeline wedges forever. The outer
    ``wait_for`` guard turns a re-introduced unbounded ``drain()`` into a
    loud failure instead of a suite hang."""
    producer = _pipe_proc()
    producer.stdout.read = AsyncMock(return_value=b"x" * (1024 * 1024))  # never EOF
    consumer = _pipe_proc()

    async def _never_drains() -> None:
        await asyncio.sleep(3600)

    consumer.stdin.drain = AsyncMock(side_effect=_never_drains)
    _spawn_pair(monkeypatch, producer, consumer)

    with pytest.raises(SandboxBackendError) as excinfo:
        await asyncio.wait_for(
            run_docker_pipeline(
                ["docker", "export", "x"],
                ["docker", "import", "-", "tag"],
                stall_timeout_s=0.05,
                max_timeout_s=5.0,
            ),
            timeout=5.0,
        )

    assert "write stalled" in str(excinfo.value)
    # Both children SIGKILLed on the timeout unwind.
    assert producer._transport.kill.called
    assert consumer._transport.kill.called


async def test_pipeline_slow_but_moving_data_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data that moves in small increments — each read/drain under the stall
    bound but their sum well over it — must NOT trip the stall deadline:
    every completed op resets the stall clock. Total relay time (~0.15s) far
    exceeds the 0.05s stall bound, yet the pipeline completes."""
    chunks = iter([b"a" * 1000, b"b" * 1000, b"c" * 1000, b"d" * 1000, b""])

    async def _slow_read(_n: int) -> bytes:
        await asyncio.sleep(0.03)  # < stall bound; genuine progress each time
        return next(chunks)

    producer = _pipe_proc()
    producer.stdout.read = _slow_read
    consumer = _pipe_proc()
    consumer.stdout.read = AsyncMock(return_value=b"sha256:imported\n")
    _spawn_pair(monkeypatch, producer, consumer)

    rc, out, _err = await run_docker_pipeline(
        ["docker", "export", "x"],
        ["docker", "import", "-", "tag"],
        stall_timeout_s=0.05,
        max_timeout_s=5.0,
    )

    assert rc == 0
    assert out == b"sha256:imported\n"
    assert consumer.stdin.write.call_count == 4  # four non-empty chunks relayed
    consumer.stdin.close.assert_called_once()  # write side closed on EOF


async def test_pipeline_post_eof_finalization_exempt_from_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After EOF the consumer may legitimately spend a long time finalizing
    the import; only the ABSOLUTE ceiling applies then, never the (short)
    stall bound. A finalize that outlasts the stall bound but beats the
    ceiling must still succeed."""
    producer = _pipe_proc()  # immediate EOF (default read → b"")
    consumer = _pipe_proc()

    async def _slow_finalize() -> int:
        await asyncio.sleep(0.15)  # > stall bound (0.05), < ceiling (5.0)
        return 0

    producer.wait = _slow_finalize  # type: ignore[attr-defined]
    consumer.wait = _slow_finalize  # type: ignore[attr-defined]
    consumer.stdout.read = AsyncMock(return_value=b"sha256:imported\n")
    _spawn_pair(monkeypatch, producer, consumer)

    rc, out, _err = await run_docker_pipeline(
        ["docker", "export", "x"],
        ["docker", "import", "-", "tag"],
        stall_timeout_s=0.05,
        max_timeout_s=5.0,
    )

    assert rc == 0
    assert out == b"sha256:imported\n"


async def test_pipeline_finalization_bounded_by_absolute_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-EOF finalize is stall-exempt but NOT unbounded: a consumer
    that never exits must still trip the absolute ceiling."""
    producer = _pipe_proc()  # immediate EOF

    async def _never_exits() -> int:
        await asyncio.sleep(3600)  # outlives the (tiny) absolute ceiling
        return 0  # unreachable; satisfies the return type

    consumer = _pipe_proc()
    consumer.wait = _never_exits  # type: ignore[attr-defined]
    producer.wait = _never_exits  # type: ignore[attr-defined]
    _spawn_pair(monkeypatch, producer, consumer)

    with pytest.raises(SandboxBackendError) as excinfo:
        await asyncio.wait_for(
            run_docker_pipeline(
                ["docker", "export", "x"],
                ["docker", "import", "-", "tag"],
                stall_timeout_s=5.0,
                max_timeout_s=0.1,
            ),
            timeout=5.0,
        )

    assert "timed out" in str(excinfo.value)
    assert producer._transport.kill.called
    assert consumer._transport.kill.called


async def test_pipeline_relays_large_progressing_stream_without_size_derived_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large corpse that keeps making progress must salvage to completion.

    The retired design put a SINGLE size-derived wall-clock timeout on the whole
    ``docker export | docker import`` — a corpse whose flatten legitimately ran
    longer than that estimate was killed, retained, and retried forever (the
    brick). The progress-based deadlines replace it: as long as bytes keep
    MOVING within the stall window, an arbitrarily large stream completes,
    bounded only by the (generous) absolute ceiling.

    This drives the REAL ``run_docker_pipeline`` and relays a substantial stream
    (256 MiB across 256 chunks) whose aggregate wall-time far exceeds the stall
    bound — a single stall/size-sized wall-clock cap would have killed it — yet
    it finishes with every byte relayed."""
    chunk = b"x" * (1024 * 1024)  # one reused 1 MiB buffer → low resident memory
    n_chunks = 256
    remaining = {"n": n_chunks}

    async def _read(_n: int) -> bytes:
        if remaining["n"] == 0:
            return b""  # EOF
        remaining["n"] -= 1
        await asyncio.sleep(0.002)  # each chunk arrives within the stall window
        return chunk

    producer = _pipe_proc()
    producer.stdout.read = _read
    consumer = _pipe_proc()
    consumer.stdout.read = AsyncMock(return_value=b"sha256:flattened\n")
    _spawn_pair(monkeypatch, producer, consumer)

    loop = asyncio.get_running_loop()
    start = loop.time()
    rc, out, _err = await run_docker_pipeline(
        ["docker", "export", "big-corpse"],
        ["docker", "import", "-", "tag"],
        stall_timeout_s=0.05,  # << aggregate relay time (~0.5s)
        max_timeout_s=30.0,
    )
    elapsed = loop.time() - start

    assert rc == 0
    assert out == b"sha256:flattened\n"
    # The full stream was relayed chunk-by-chunk through the real pump.
    assert consumer.stdin.write.call_count == n_chunks
    moved = sum(len(call.args[0]) for call in consumer.stdin.write.call_args_list)
    assert moved == n_chunks * len(chunk)  # 256 MiB relayed end-to-end
    consumer.stdin.close.assert_called_once()  # write side closed on EOF
    # Aggregate wall-time outran the per-op stall bound ~10x: a single
    # size/stall-sized wall-clock cap would have killed this large corpse;
    # progress-based deadlines let it finish.
    assert elapsed > 0.05
