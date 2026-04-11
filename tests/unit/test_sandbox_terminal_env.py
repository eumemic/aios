"""Unit tests for the SandboxTerminalEnv adapter.

Tests the boundary between aios's ContainerHandle and hermes's
ShellFileOperations. Stubs the ContainerHandle with AsyncMock so we
don't touch Docker.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.sandbox.container import CommandResult, ContainerError, ContainerHandle
from aios.tools.adapters import ExecuteResult, SandboxTerminalEnv


def _handle_with_canned_result(result: CommandResult) -> ContainerHandle:
    handle = ContainerHandle(
        session_id="sess_01TEST",
        container_id="container_abc123",
        workspace_path=Path("/tmp/aios-test-workspace"),
    )
    handle.run_command = AsyncMock(return_value=result)  # type: ignore[method-assign]
    return handle


def _default_result(stdout: str = "", exit_code: int = 0, stderr: str = "") -> CommandResult:
    return CommandResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        truncated=False,
    )


class TestPassthrough:
    async def test_plain_command_passed_through(self) -> None:
        handle = _handle_with_canned_result(_default_result(stdout="hello\n"))
        env = SandboxTerminalEnv(handle)

        result = await env.run_command("echo hello")

        assert isinstance(result, ExecuteResult)
        assert result.stdout == "hello\n"
        assert result.exit_code == 0
        # Verify the wrapped command was unchanged since stdin_data is None.
        handle.run_command.assert_awaited_once()  # type: ignore[attr-defined]
        args: tuple[Any, ...] = handle.run_command.await_args.args  # type: ignore[attr-defined]
        assert args[0] == "echo hello"

    async def test_cwd_default_is_workspace(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        await env.run_command("pwd")

        kwargs: dict[str, Any] = handle.run_command.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["cwd"] == "/workspace"

    async def test_cwd_override_passed_through(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        await env.run_command("pwd", cwd="/tmp")

        kwargs: dict[str, Any] = handle.run_command.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["cwd"] == "/tmp"

    async def test_cwd_attribute_exposed(self) -> None:
        """Some vendored code reads env.cwd via getattr; make sure it's set."""
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        assert env.cwd == "/workspace"

    async def test_custom_default_cwd(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle, default_cwd="/srv/app")

        assert env.cwd == "/srv/app"
        await env.run_command("pwd")
        kwargs: dict[str, Any] = handle.run_command.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["cwd"] == "/srv/app"


class TestTimeout:
    async def test_timeout_default(self) -> None:
        from aios.config import get_settings

        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        await env.run_command("true")

        kwargs: dict[str, Any] = handle.run_command.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["timeout_seconds"] == get_settings().bash_default_timeout_seconds

    async def test_timeout_explicit(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        await env.run_command("sleep 10", timeout=5)

        kwargs: dict[str, Any] = handle.run_command.await_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["timeout_seconds"] == 5


class TestStdinData:
    async def test_stdin_data_is_base64_wrapped(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        payload = "hello world\n"
        await env.run_command("tee /workspace/out.txt", stdin_data=payload)

        args: tuple[Any, ...] = handle.run_command.await_args.args  # type: ignore[attr-defined]
        wrapped = args[0]
        assert wrapped.startswith("base64 -d <<< ")
        assert "tee /workspace/out.txt" in wrapped

    async def test_stdin_data_decodes_to_original(self) -> None:
        """The base64 token the wrapper emits must decode back to the input."""
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        payload = "line1\nline2\n"
        await env.run_command("cat > /workspace/x", stdin_data=payload)

        args: tuple[Any, ...] = handle.run_command.await_args.args  # type: ignore[attr-defined]
        wrapped = args[0]
        # Extract the quoted base64 token between `<<< ` and ` |`.
        prefix = "base64 -d <<< "
        pipe_idx = wrapped.index(" | ")
        quoted = wrapped[len(prefix) : pipe_idx]
        # shlex.quote wraps in single quotes when the payload has no
        # special chars; for the base64 alphabet the quotes may be
        # omitted, so handle both.
        token = quoted[1:-1] if quoted.startswith("'") and quoted.endswith("'") else quoted
        decoded = base64.b64decode(token).decode("utf-8")
        assert decoded == payload

    async def test_adversarial_payload_roundtrips(self) -> None:
        """Newlines, NULs, single quotes, shell metacharacters, UTF-8."""
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        payload = "a\x00b'c\"d$e`f|g\n\thello \u00e9\u00f1\u00fc\n"
        await env.run_command("cat > /workspace/y", stdin_data=payload)

        args: tuple[Any, ...] = handle.run_command.await_args.args  # type: ignore[attr-defined]
        wrapped = args[0]
        prefix = "base64 -d <<< "
        pipe_idx = wrapped.index(" | ")
        quoted = wrapped[len(prefix) : pipe_idx]
        token = quoted[1:-1] if quoted.startswith("'") and quoted.endswith("'") else quoted
        decoded = base64.b64decode(token).decode("utf-8")
        assert decoded == payload

    async def test_no_stdin_is_not_wrapped(self) -> None:
        handle = _handle_with_canned_result(_default_result())
        env = SandboxTerminalEnv(handle)

        await env.run_command("ls /workspace")

        args: tuple[Any, ...] = handle.run_command.await_args.args  # type: ignore[attr-defined]
        assert args[0] == "ls /workspace"


class TestResultMapping:
    async def test_result_fields_propagate(self) -> None:
        handle = _handle_with_canned_result(
            CommandResult(
                exit_code=42,
                stdout="out\n",
                stderr="err\n",
                timed_out=True,
                truncated=True,
            )
        )
        env = SandboxTerminalEnv(handle)

        result = await env.run_command("true")

        assert result.exit_code == 42
        assert result.stdout == "out\n"
        assert result.stderr == "err\n"
        assert result.timed_out is True
        assert result.truncated is True


class TestErrorPropagation:
    async def test_container_error_bubbles(self) -> None:
        handle = ContainerHandle(
            session_id="sess_01TEST",
            container_id="container_abc123",
            workspace_path=Path("/tmp/aios-test-workspace"),
        )
        handle.run_command = AsyncMock(side_effect=ContainerError("daemon hiccup"))  # type: ignore[method-assign]
        env = SandboxTerminalEnv(handle)

        with pytest.raises(ContainerError):
            await env.run_command("true")
