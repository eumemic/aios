"""Adapters between aios sandbox primitives and vendored hermes tools.

The vendored ``ShellFileOperations`` code (from
``aios.vendor.hermes_files.file_operations``) expects a terminal-like
object with an async ``run_command(command, cwd, timeout, stdin_data)``
method that returns an object exposing ``.stdout``, ``.exit_code``, and
``.stderr`` attributes. aios's ``ContainerHandle.run_command`` has a
different signature (``timeout_seconds`` not ``timeout``, with
``max_output_bytes`` as well) and returns a different dataclass
(``CommandResult``). This module is the single bridge between the two.

:class:`SandboxTerminalEnv` also handles ``stdin_data``, which aios's
``ContainerHandle`` does not accept natively: the bytes are base64-encoded
on the host and decoded inside the container via
``base64 -d <<< <quoted> | <original_command>``. This wrapping pays a
~4/3 size overhead but is robust against every possible payload byte
(newlines, NULs, shell metacharacters, single quotes, UTF-8) in a way
that heredocs are not. The ARG_MAX ceiling after base64 inflation is
~1.5 MB, comfortably above ``file_write_max_chars``.

The separate ``file_tool_max_output_bytes`` config knob lets file tools
return larger outputs than the bash tool default — file reads naturally
want a higher cap than interactive shell commands.
"""

from __future__ import annotations

import base64
import shlex
from dataclasses import dataclass

from aios.config import get_settings
from aios.sandbox.container import ContainerHandle


@dataclass(slots=True, frozen=True)
class ExecuteResult:
    """Result of one shell command run via :class:`SandboxTerminalEnv`.

    The field names match what the vendored ``ShellFileOperations`` code
    reads off its execute-result object: ``.stdout`` and ``.exit_code``
    are required, ``.stderr`` is surfaced separately because ripgrep's
    error path needs to distinguish "no matches" (exit 1, empty stdout)
    from "rg failed" (exit 2, error on stderr).
    """

    stdout: str
    exit_code: int
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False


class SandboxTerminalEnv:
    """Adapt aios's :class:`ContainerHandle` to the interface the vendored
    ``ShellFileOperations`` code expects.

    The vendored code calls::

        result = await self.env.run_command(
            cmd,
            cwd=cwd,
            timeout=N,
            stdin_data=...,
        )

    and reads ``result.stdout`` / ``result.exit_code`` / ``result.stderr``.
    This adapter presents that surface and delegates to
    ``ContainerHandle.run_command`` underneath.

    The ``cwd`` attribute is also read by some vendored helpers that do
    ``getattr(env, 'cwd', None)``, so we expose it as a plain attribute.
    """

    def __init__(self, handle: ContainerHandle, *, default_cwd: str = "/workspace") -> None:
        self.handle = handle
        self.default_cwd = default_cwd
        # Exposed so vendored code's ``getattr(env, 'cwd', None)`` works.
        self.cwd = default_cwd

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,  # noqa: ASYNC109  # hermes API contract, not asyncio cancellation
        stdin_data: str | None = None,
    ) -> ExecuteResult:
        """Run one shell command via the underlying container handle.

        ``stdin_data`` is supported by base64-encoding the payload on the
        host and decoding it inside the container with ``base64 -d``. The
        underlying ``ContainerHandle`` abstraction does not natively pipe
        stdin, and adding that capability would mean changing the docker
        invocation used by the bash tool too. Base64 wrapping is a pure
        userspace trick that keeps the sandbox primitive unchanged.

        The base64 alphabet contains only ``[A-Za-z0-9+/=]``, none of
        which are shell metacharacters, so ``shlex.quote`` is belt-and-
        braces but kept for explicitness.

        Raises :class:`aios.sandbox.container.ContainerError` if the
        underlying docker invocation fails at the host level (container
        gone, daemon unreachable). The tool dispatcher converts that into
        an ``is_error=true`` tool message and evicts the session's cached
        state.
        """
        settings = get_settings()
        effective_timeout = timeout or settings.bash_default_timeout_seconds
        effective_cwd = cwd or self.default_cwd

        if stdin_data is not None:
            encoded = base64.b64encode(stdin_data.encode("utf-8")).decode("ascii")
            # `base64 -d <<< "<b64>" | <original>` is bash-specific (the
            # `<<<` here-string), which is fine: every command we run
            # inside the sandbox is already wrapped in `bash -c` by the
            # ContainerHandle.run_command layer.
            wrapped = f"base64 -d <<< {shlex.quote(encoded)} | {command}"
        else:
            wrapped = command

        result = await self.handle.run_command(
            wrapped,
            timeout_seconds=effective_timeout,
            max_output_bytes=settings.file_tool_max_output_bytes,
            cwd=effective_cwd,
        )

        return ExecuteResult(
            stdout=result.stdout,
            exit_code=result.exit_code,
            stderr=result.stderr,
            timed_out=result.timed_out,
            truncated=result.truncated,
        )
