"""A handle to a running sandbox container.

One :class:`ContainerHandle` corresponds to one running Docker container
that a session is using. The handle tracks the container id, the session
it belongs to, and the host-side workspace path. Its main method,
:meth:`run_command`, runs a shell command inside the container via the
``docker exec`` CLI and returns the exit code, stdout, and stderr.

The handle does NOT own its own lifecycle. Creation and teardown live in
:mod:`aios.sandbox.provisioner`. The handle is a dumb I/O object — caching
and orphan reaping live in :mod:`aios.sandbox.registry`.

Why the ``docker exec`` CLI via :func:`asyncio.create_subprocess_exec`
instead of aiodocker's exec API? aiodocker wraps Docker's exec HTTP
endpoint and requires manual demultiplexing of the stdout/stderr
multiplexed stream. Shelling out to the ``docker`` CLI via
``asyncio.create_subprocess_exec`` (which passes argv directly to the OS,
with NO shell interpretation) gives us clean stdout/stderr separation
and the timeout semantics we need.

Security note: the command string the agent supplies is passed as a
single argv element to ``bash -c`` inside the container. The ``docker
exec`` argv we build on the host contains ONLY trusted values
(container_id we created, the literal string ``bash``, ``-c``, and the
agent-supplied command). No host shell is invoked on the outside;
injection on the host side is impossible because
``create_subprocess_exec`` does not interpret shell metacharacters.
Injection on the inside is by design — the agent IS expected to run
arbitrary shell inside the container. That's what a sandbox is for.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aios.models.github_repositories import GithubRepositoryResourceEcho
    from aios.models.memory_stores import MemoryStoreResourceEcho
    from aios.sandbox.git_proxy import GitProxy


def mount_snapshot_from_echoes(
    memory_echoes: Iterable[MemoryStoreResourceEcho],
    github_echoes: Iterable[GithubRepositoryResourceEcho],
) -> frozenset[tuple[str, ...]]:
    """The set of inputs that determines the docker ``--volume`` argv.

    Order-independent so rank reorders don't trigger spurious recycles.
    Each tuple is type-prefixed so memory and github namespaces can't
    collide. The github tuple includes ``updated_at`` so token rotation
    (which bumps ``updated_at``) propagates to a container recycle.
    """
    from aios.ids import GITHUB_REPOSITORY, MEMORY_STORE

    items: set[tuple[str, ...]] = set()
    for m in memory_echoes:
        items.add((MEMORY_STORE, m.memory_store_id, m.name, m.access))
    for g in github_echoes:
        items.add((GITHUB_REPOSITORY, g.id, g.mount_path, g.updated_at.isoformat()))
    return frozenset(items)


@dataclass(slots=True, frozen=True)
class CommandResult:
    """Result of a single command running inside a container."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class ContainerError(Exception):
    """Raised when the docker invocation itself fails (e.g. container gone,
    daemon hiccup). Distinct from a command that runs and returns nonzero
    — that's a successful :class:`CommandResult` with a nonzero ``exit_code``.
    """


# Bound on the post-kill drain so a docker CLI stuck in an uninterruptible
# state can't hold the worker indefinitely past the wait_for timeout.
_DRAIN_AFTER_KILL_TIMEOUT_S = 2.0


async def run_subprocess_with_timeout(
    argv: list[str], *, timeout_s: float
) -> tuple[int, bytes, bytes, bool]:
    """Launch ``argv``, return ``(returncode, stdout, stderr, timed_out)``.

    On timeout, sends SIGKILL and drains the pipes with a secondary 2s
    bound. Returns ``-1`` for ``returncode`` if the process was never
    reaped. Raises :class:`ContainerError` only on launch failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as host_err:
        raise ContainerError(f"failed to launch subprocess: {host_err}") from host_err
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


class ContainerHandle:
    """A handle to one running sandbox container.

    Construct via :func:`aios.sandbox.provisioner.provision_for_session`.
    Tear down via :func:`aios.sandbox.provisioner.release`. This class is
    not a context manager because its lifetime is owned by the worker's
    sandbox registry, not by individual tool calls.
    """

    def __init__(
        self,
        *,
        session_id: str,
        container_id: str,
        workspace_path: Path,
        mount_snapshot: frozenset[tuple[str, ...]] = frozenset(),
        git_proxy: GitProxy | None = None,
    ) -> None:
        self.session_id = session_id
        self.container_id = container_id
        self.workspace_path = workspace_path
        self.mount_snapshot = mount_snapshot
        # Per-session credential broker for github_repository attachments.
        # Held here so the handle's release path can stop it.
        self.git_proxy = git_proxy

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
        cwd: str = "/workspace",
    ) -> CommandResult:
        """Run a shell command inside the container.

        Runs ``bash -c <command>`` via the ``docker exec`` CLI. Output
        beyond ``max_output_bytes`` is truncated. The command is killed
        if it runs longer than ``timeout_seconds`` (SIGKILL on the host
        process; the container itself stays up).

        Raises :class:`ContainerError` if the docker invocation fails to
        start or the docker daemon is unreachable. A command that runs
        and returns a nonzero exit code is NOT an error — it's a
        successful :class:`CommandResult` with the nonzero ``exit_code``.
        """
        argv = [
            "docker",
            "exec",
            "--workdir",
            cwd,
            self.container_id,
            "bash",
            "-c",
            command,
        ]

        exit_code, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
            argv, timeout_s=float(timeout_seconds)
        )

        stdout_str, out_truncated = _decode_and_truncate(stdout_bytes, max_output_bytes)
        stderr_str, err_truncated = _decode_and_truncate(stderr_bytes, max_output_bytes)

        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            timed_out=timed_out,
            truncated=out_truncated or err_truncated,
        )


def _decode_and_truncate(raw: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode ``raw`` as UTF-8 (with replacement) and truncate to ``max_bytes``.

    Returns the decoded string and a flag indicating whether truncation
    happened. Truncation is byte-based (not char-based) to avoid surprises
    on very large outputs.
    """
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace"), False
    truncated = raw[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n\n[output truncated]", True
