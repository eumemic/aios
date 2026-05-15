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
            stdout_bytes, stderr_bytes = b"", b""
        return -1, stdout_bytes, stderr_bytes, True


async def run_docker_cli(
    argv: list[str], *, timeout_s: float = DOCKER_CLI_TIMEOUT_S
) -> tuple[int, bytes, bytes]:
    """Run a ``docker`` CLI invocation. Returns ``(exit_code, stdout, stderr)``.

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
