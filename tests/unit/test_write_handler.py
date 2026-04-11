"""Unit tests for the write tool handler.

Mocks ContainerHandle.run_command and inspects the shell command the
handler constructs to verify base64 encoding, parent-dir creation, and
path quoting.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.tools.write import WriteArgumentError, write_handler


class _StubRegistry:
    def __init__(self, handle: ContainerHandle) -> None:
        self._handle = handle

    async def get_or_provision(self, session_id: str) -> ContainerHandle:
        return self._handle


@pytest.fixture
def stub_handle() -> ContainerHandle:
    handle = ContainerHandle(
        session_id="sess_01TEST",
        container_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )
    handle.run_command = AsyncMock(  # type: ignore[method-assign]
        return_value=CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            truncated=False,
        )
    )
    return handle


@pytest.fixture
def stub_registry(stub_handle: ContainerHandle) -> Any:
    previous = runtime.sandbox_registry
    runtime.sandbox_registry = _StubRegistry(stub_handle)  # type: ignore[assignment]
    try:
        yield
    finally:
        runtime.sandbox_registry = previous


class TestArguments:
    async def test_rejects_missing_path(self, stub_registry: Any) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"content": "hi"})

    async def test_rejects_missing_content(self, stub_registry: Any) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"path": "/workspace/a.txt"})

    async def test_rejects_non_string_content(self, stub_registry: Any) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": 42})


class TestHappyPath:
    async def test_returns_bytes_written(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/a.txt", "content": "hello"}
        )
        assert result == {"path": "/workspace/a.txt", "bytes_written": 5}

    async def test_bytes_written_counts_utf8_bytes_not_chars(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        # "héllo" is 6 bytes in UTF-8 (é is 2 bytes) but 5 chars.
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/a.txt", "content": "héllo"}
        )
        assert result["bytes_written"] == 6

    async def test_command_base64_encodes_content(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        content = "hello world\n"
        await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": content})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        expected_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        assert f"base64 -d <<< '{expected_b64}'" in cmd

    async def test_command_creates_parent_dirs(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        await write_handler("sess_01TEST", {"path": "/workspace/a/b/c.txt", "content": "hi"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        # shlex.quote leaves simple paths unquoted; assert mkdir + dirname + path.
        assert "mkdir -p --" in cmd
        assert "dirname --" in cmd
        assert "/workspace/a/b/c.txt" in cmd

    async def test_command_redirects_to_quoted_path(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        await write_handler("sess_01TEST", {"path": "/workspace/a file.txt", "content": "hi"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "> '/workspace/a file.txt'" in cmd

    async def test_handles_special_characters_in_content(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        # Content containing quotes, newlines, shell metacharacters.
        tricky = "line with 'quotes' and \"doubles\" and $vars\nand newlines"
        await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": tricky})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        # Base64 is quote-safe so no escaping gymnastics.
        expected_b64 = base64.b64encode(tricky.encode("utf-8")).decode("ascii")
        assert expected_b64 in cmd
        # No literal quotes or dollar signs leaked into the command.
        assert "'quotes'" not in cmd
        assert "$vars" not in cmd


class TestErrorPath:
    async def test_nonzero_exit_returns_error_dict(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="bash: /readonly/a.txt: Permission denied\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await write_handler("sess_01TEST", {"path": "/readonly/a.txt", "content": "hi"})
        assert "error" in result
        assert "Permission denied" in result["error"]
        assert result["path"] == "/readonly/a.txt"
