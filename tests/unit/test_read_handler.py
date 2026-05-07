"""Unit tests for the read tool handler.

Stubs the sandbox registry and mocks SandboxHandle.run_command so the
tests don't touch Docker. The same pattern as test_bash_handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.read import ReadArgumentError, read_handler


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        return self._handle


@pytest.fixture
def stub_handle() -> SandboxHandle:
    handle = SandboxHandle(
        session_id="sess_01TEST",
        sandbox_id="container_abc",
        workspace_path=Path("/tmp/aios-test"),
    )
    return handle


@pytest.fixture
def canned_result() -> CommandResult:
    return CommandResult(
        exit_code=0,
        stdout="     1\thello\n     2\tworld\n",
        stderr="",
        timed_out=False,
        truncated=False,
    )


@pytest.fixture
def stub_registry(stub_handle: SandboxHandle, canned_result: CommandResult) -> Any:
    from unittest.mock import MagicMock

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    stub = _StubRegistry(stub_handle, canned_result)
    runtime.sandbox_registry = stub  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield stub
    finally:
        runtime.sandbox_registry = prev_registry
        runtime.pool = prev_pool


class TestArguments:
    async def test_rejects_missing_path(self, stub_registry: Any) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {})

    async def test_rejects_empty_path(self, stub_registry: Any) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "   "})

    async def test_rejects_non_string_path(self, stub_registry: Any) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": 42})

    async def test_rejects_zero_offset(self, stub_registry: Any) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "/workspace/a.txt", "offset": 0})

    async def test_rejects_negative_limit(self, stub_registry: Any) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "/workspace/a.txt", "limit": -1})


class TestHappyPath:
    async def test_returns_content(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        result = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert result == {
            "path": "/workspace/a.txt",
            "content": "     1\thello\n     2\tworld\n",
        }

    async def test_passes_path_and_range_to_cat_sed_pipeline(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await read_handler(
            "sess_01TEST",
            {"path": "/workspace/a.txt", "offset": 5, "limit": 10},
        )
        stub_registry.exec.assert_awaited_once()  # type: ignore[attr-defined]
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        # shlex.quote leaves simple paths unquoted; just assert substring presence.
        assert "cat -n --" in cmd
        assert "/workspace/a.txt" in cmd
        assert "sed -n" in cmd
        # Range is offset..offset+limit-1, so 5..14 for offset=5 limit=10.
        assert "5,14p" in cmd

    async def test_default_range_starts_at_line_one(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "1,2000p" in cmd

    async def test_quotes_paths_with_spaces(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await read_handler("sess_01TEST", {"path": "/workspace/a file.txt"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "'/workspace/a file.txt'" in cmd


class TestErrorPath:
    async def test_cat_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="cat: /nope: No such file or directory\\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await read_handler("sess_01TEST", {"path": "/nope"})
        assert "error" in result
        assert "No such file" in result["error"]
        assert result["path"] == "/nope"
