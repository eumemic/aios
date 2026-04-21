"""Unit tests for the grep tool handler.

Stubs the sandbox registry and mocks ContainerHandle.run_command so the
tests don't touch Docker. The same pattern as test_read_handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.tools.grep import GrepArgumentError, grep_handler


class _StubRegistry:
    def __init__(self, handle: ContainerHandle) -> None:
        self._handle = handle

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> ContainerHandle:
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
            stdout="/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
            stderr="",
            timed_out=False,
            truncated=False,
        )
    )
    return handle


@pytest.fixture
def stub_registry(stub_handle: ContainerHandle) -> Any:
    from unittest.mock import MagicMock

    prev_registry = runtime.sandbox_registry
    prev_pool = runtime.pool
    runtime.sandbox_registry = _StubRegistry(stub_handle)  # type: ignore[assignment]
    runtime.pool = MagicMock()
    try:
        yield
    finally:
        runtime.sandbox_registry = prev_registry
        runtime.pool = prev_pool


class TestArguments:
    async def test_rejects_missing_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {})

    async def test_rejects_empty_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "   "})

    async def test_rejects_non_string_pattern(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": 42})

    async def test_rejects_empty_path(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "hello", "path": "   "})

    async def test_rejects_empty_include(self, stub_registry: Any) -> None:
        with pytest.raises(GrepArgumentError):
            await grep_handler("sess_01TEST", {"pattern": "hello", "include": "   "})


class TestHappyPath:
    async def test_returns_matches(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        result = await grep_handler("sess_01TEST", {"pattern": "hello"})
        assert result == {
            "matches": "/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
        }

    async def test_default_path(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert cmd.startswith("rg")
        assert "/workspace" in cmd

    async def test_include_flag(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "include": "*.py"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "--glob" in cmd
        assert "*.py" in cmd

    async def test_no_matches_returns_empty_string(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )
        result = await grep_handler("sess_01TEST", {"pattern": "nonexistent"})
        assert result == {"matches": ""}


class TestOutputModes:
    async def test_files_with_matches_mode(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "files_with_matches"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert " -l " in cmd

    async def test_count_mode(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "count"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert " -c " in cmd

    async def test_content_mode_has_line_numbers(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "content"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert " -n " in cmd


class TestAdvancedFlags:
    async def test_context_lines(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "context": 3})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "-C 3" in cmd

    async def test_case_insensitive(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "case_insensitive": True})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert " -i " in cmd

    async def test_multiline(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "multiline": True})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "-U" in cmd
        assert "--multiline-dotall" in cmd

    async def test_file_type_filter(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "file_type": "py"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "--type" in cmd
        assert "py" in cmd

    async def test_custom_limit(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "limit": 100})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "head -100" in cmd

    async def test_default_limit_250(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "head -250" in cmd

    async def test_max_columns_flag(self, stub_registry: Any, stub_handle: ContainerHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_handle.run_command.await_args.args[0]  # type: ignore[attr-defined]
        assert "--max-columns=500" in cmd


class TestErrorPath:
    async def test_grep_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: ContainerHandle
    ) -> None:
        stub_handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=2,
                stdout="",
                stderr="grep: invalid regex\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await grep_handler("sess_01TEST", {"pattern": "[invalid"})
        assert "error" in result
        assert "invalid regex" in result["error"]
