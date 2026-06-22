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
import os

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
    timeout_s: float,
) -> tuple[int, bytes, bytes]:
    """Run ``producer_argv | consumer_argv`` joined by an ``os.pipe()`` fd pair.

    Used for the flatten path ``docker export <id> | docker import - <tag>``
    without a host shell: both argvs go straight to ``execve`` and are
    connected at the OS level, so no shell metacharacter interpretation
    happens on the host (same security posture as
    :func:`run_subprocess_with_timeout`). Returns the **consumer's**
    ``(returncode, stdout, stderr)`` — ``docker import`` prints the new
    image id to stdout — so the caller can read the flattened image id.

    Raises :class:`SandboxBackendError` on launch failure, on timeout
    (both children are SIGKILLed), or when the **producer** exits nonzero
    (a failed ``export`` would otherwise feed the consumer a truncated
    stream that imports as a silently-corrupt image). The consumer's own
    nonzero exit is returned for the caller to interpret, mirroring
    :func:`run_docker_cli`.
    """
    rd, wr = os.pipe()
    producer: asyncio.subprocess.Process | None = None
    consumer: asyncio.subprocess.Process | None = None
    try:
        try:
            producer = await asyncio.create_subprocess_exec(
                *producer_argv, stdout=wr, stderr=asyncio.subprocess.PIPE
            )
        except (OSError, FileNotFoundError) as host_err:
            raise SandboxBackendError(
                f"failed to launch {producer_argv[0]}: {host_err}"
            ) from host_err
        # The producer now holds the only writer; the parent must drop its
        # copy so the consumer sees EOF when the producer exits.
        os.close(wr)
        wr = -1
        try:
            consumer = await asyncio.create_subprocess_exec(
                *consumer_argv,
                stdin=rd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError) as host_err:
            raise SandboxBackendError(
                f"failed to launch {consumer_argv[0]}: {host_err}"
            ) from host_err
        os.close(rd)
        rd = -1

        async def _drain() -> tuple[tuple[bytes, bytes], tuple[bytes, bytes]]:
            # Drive both concurrently so neither stderr/stdout pipe can
            # fill and deadlock the other. Both are non-None here (assigned
            # above before _drain is ever awaited).
            return await asyncio.gather(consumer.communicate(), producer.communicate())

        try:
            (cons_out, cons_err), (_prod_out, prod_err) = await asyncio.wait_for(
                _drain(), timeout=timeout_s
            )
        except TimeoutError as timeout_err:
            for proc in (producer, consumer):
                if proc is not None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
            raise SandboxBackendError(
                f"docker pipeline timed out after {timeout_s}s: "
                f"{' '.join(producer_argv)} | {' '.join(consumer_argv)}"
            ) from timeout_err
        except BaseException:
            # CancelledError is a BaseException, not a TimeoutError, so an
            # outer cancel (caller's job deadline, worker SIGTERM mid
            # multi-GB export) skips the timeout branch above — both
            # children keep running and their parent-side pipe FDs leak.
            # Mirror run_subprocess_with_timeout: SIGKILL both and close
            # their transports before propagating. (The timeout branch
            # needs no close — the loop reclaims each transport on the
            # killed child's pipe EOF — but a propagating cancel returns
            # immediately, so it must release them here.)
            for proc in (producer, consumer):
                if proc is not None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    proc._transport.close()  # type: ignore[attr-defined]  # typeshed omits Process._transport
            raise

        if producer.returncode != 0:
            raise SandboxBackendError(
                f"{producer_argv[0]} export failed (exit {producer.returncode}): "
                f"{prod_err.decode('utf-8', errors='replace').strip()}"
            )
        return (
            consumer.returncode if consumer.returncode is not None else -1,
            cons_out,
            cons_err,
        )
    finally:
        # Close any fd we still own (launch-failure / early-raise paths).
        for fd in (rd, wr):
            if fd != -1:
                with contextlib.suppress(OSError):
                    os.close(fd)
