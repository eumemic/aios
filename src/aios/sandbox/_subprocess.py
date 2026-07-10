"""Subprocess primitive used by sandbox host-side helpers.

Both the Docker backend (``backends/docker.py``, for ``docker`` CLI
calls) and the github-clone helpers (``github_clone.py``, for ``git``
on the host) need a non-blocking subprocess runner with timeout +
SIGKILL semantics. Lifting it out of either consumer keeps the
backend free to be a clean Protocol implementation and keeps github
clone code from depending on a backend module.

This module is sandbox-internal; nothing outside ``aios.sandbox`` should
import from it.
"""

from __future__ import annotations

import asyncio
import contextlib

from aios.sandbox.backends.base import SandboxBackendError

# Bound on the post-kill drain so a CLI stuck in an uninterruptible
# state can't hold the worker indefinitely past the wait_for timeout.
_DRAIN_AFTER_KILL_TIMEOUT_S = 2.0

# Bound every ``docker`` management call so a stalled daemon can't wedge
# the worker step path (per issue #179 / commit e675ed2).
DOCKER_CLI_TIMEOUT_S = 30.0


async def run_subprocess_with_timeout(
    argv: list[str], *, timeout_s: float
) -> tuple[int, bytes, bytes, bool]:
    """Launch ``argv``, return ``(returncode, stdout, stderr, timed_out)``.

    On timeout, sends SIGKILL and drains the pipes with a secondary 2s
    bound. Returns ``-1`` for ``returncode`` if the process was never
    reaped. Raises :class:`SandboxBackendError` only on launch failure
    (missing binary, OS resource exhaustion, etc.).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as host_err:
        raise SandboxBackendError(f"failed to launch subprocess: {host_err}") from host_err
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return (
            proc.returncode if proc.returncode is not None else -1,
            stdout_bytes,
            stderr_bytes,
            False,
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_DRAIN_AFTER_KILL_TIMEOUT_S
            )
        except (TimeoutError, OSError):
            # Child wedged in uninterruptible D-state: the SIGKILL above
            # stays pending until it leaves the syscall, so communicate()
            # never reaches its pipe-close/reap. The cancelled
            # communicate() left the parent-side stdout/stderr pipe FDs
            # open, and the loop only reclaims them on a child-exit EOF
            # that won't come. Close the transport to free them now —
            # await proc.wait() would re-introduce the very hang this
            # branch defends against.
            proc._transport.close()  # type: ignore[attr-defined]  # typeshed omits Process._transport
            stdout_bytes, stderr_bytes = b"", b""
        return -1, stdout_bytes, stderr_bytes, True
    except BaseException:
        # CancelledError is a BaseException, not a TimeoutError, so the
        # outer-cancel path (caller's job timeout, worker shutdown)
        # skips the give-up cleanup above — child keeps running and
        # pipe FDs leak. SIGKILL + close the transport before propagating.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        proc._transport.close()  # type: ignore[attr-defined]
        raise


async def run_docker_cli(
    argv: list[str], *, timeout_s: float = DOCKER_CLI_TIMEOUT_S
) -> tuple[int, bytes, bytes]:
    """Run a ``docker`` CLI call. Returns ``(exit_code, stdout, stderr)``.

    Raises :class:`SandboxBackendError` on launch failure or timeout. A
    nonzero ``docker`` exit is returned as a regular tuple — callers
    decide whether it's fatal.
    """
    rc, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
        argv, timeout_s=timeout_s
    )
    if timed_out:
        raise SandboxBackendError(f"docker cli timed out after {timeout_s}s: {' '.join(argv)}")
    return rc, stdout_bytes, stderr_bytes


async def run_docker_pipeline(
    producer_argv: list[str],
    consumer_argv: list[str],
    *,
    stall_timeout_s: float,
    max_timeout_s: float,
) -> tuple[int, bytes, bytes]:
    """Run a Docker producer/consumer pipeline with progress-based deadlines.

    The relay makes stream progress observable. A lack of bytes before EOF is
    a stall; after EOF the consumer may legitimately spend substantial time
    finalizing, so only the absolute ceiling applies then.
    """
    producer: asyncio.subprocess.Process | None = None
    consumer: asyncio.subprocess.Process | None = None
    started = asyncio.get_running_loop().time()

    async def _kill_all(*, close_transports: bool = False) -> None:
        for proc in (producer, consumer):
            if proc is not None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                if close_transports:
                    proc._transport.close()  # type: ignore[attr-defined]

    try:
        try:
            producer = await asyncio.create_subprocess_exec(
                *producer_argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            consumer = await asyncio.create_subprocess_exec(
                *consumer_argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError) as host_err:
            raise SandboxBackendError(f"failed to launch docker pipeline: {host_err}") from host_err
        assert producer.stdout is not None and consumer.stdin is not None

        async def _pump(
            producer_stdout: asyncio.StreamReader, consumer_stdin: asyncio.StreamWriter
        ) -> int:
            moved = 0
            while True:
                remaining = max_timeout_s - (asyncio.get_running_loop().time() - started)
                if remaining <= 0:
                    raise TimeoutError("absolute ceiling")
                try:
                    chunk = await asyncio.wait_for(
                        producer_stdout.read(1024 * 1024), timeout=min(stall_timeout_s, remaining)
                    )
                except TimeoutError as err:
                    raise TimeoutError("stream stalled") from err
                if not chunk:
                    consumer_stdin.close()
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        await consumer_stdin.wait_closed()
                    return moved
                consumer_stdin.write(chunk)
                await consumer_stdin.drain()
                moved += len(chunk)

        prod_err_task = asyncio.create_task(producer.stderr.read())  # type: ignore[union-attr]
        cons_out_task = asyncio.create_task(consumer.stdout.read())  # type: ignore[union-attr]
        cons_err_task = asyncio.create_task(consumer.stderr.read())  # type: ignore[union-attr]
        try:
            moved = await _pump(producer.stdout, consumer.stdin)
            remaining = max_timeout_s - (asyncio.get_running_loop().time() - started)
            if remaining <= 0:
                raise TimeoutError("absolute ceiling")
            await asyncio.wait_for(asyncio.gather(producer.wait(), consumer.wait()), remaining)
            prod_err, cons_out, cons_err = await asyncio.gather(
                prod_err_task, cons_out_task, cons_err_task
            )
        except TimeoutError as timeout_err:
            await _kill_all()
            raise SandboxBackendError(
                f"docker pipeline timed out ({timeout_err}; {moved if 'moved' in locals() else 0} bytes moved): "
                f"{' '.join(producer_argv)} | {' '.join(consumer_argv)}"
            ) from timeout_err
        if producer.returncode != 0:
            raise SandboxBackendError(
                f"{producer_argv[0]} export failed (exit {producer.returncode}): "
                f"{prod_err.decode('utf-8', errors='replace').strip()}"
            )
        return consumer.returncode if consumer.returncode is not None else -1, cons_out, cons_err
    except BaseException:
        # Outer cancellation returns immediately, so close both transports
        # after killing to release their parent-side pipe descriptors.
        await _kill_all(close_transports=True)
        raise
