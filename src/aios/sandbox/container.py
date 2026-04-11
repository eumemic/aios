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
from dataclasses import dataclass
from pathlib import Path


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
    ) -> None:
        self.session_id = session_id
        self.container_id = container_id
        self.workspace_path = workspace_path

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

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError) as host_err:
            raise ContainerError(f"failed to launch docker cli: {host_err}") from host_err

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            # Kill the host-side docker process. The container keeps running.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            try:
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""

        exit_code = proc.returncode if proc.returncode is not None else -1

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
