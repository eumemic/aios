"""Unit tests for the grep tool handler.

Stubs the sandbox registry and mocks SandboxHandle.run_command so the
tests don't touch Docker. The same pattern as test_read_handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.grep import GrepArgumentError, grep_handler


class _StubRegistry:
    """Minimal stand-in for SandboxRegistry used by handler tests."""

    def __init__(self, handle: SandboxHandle, result: CommandResult) -> None:
        self._handle = handle
        self.exec = AsyncMock(return_value=result)

    async def get_or_provision(self, session_id: str, **_kwargs: Any) -> SandboxHandle:
        return self._handle


@pytest.fixture
def stub_handle(**kwargs: Any) -> SandboxHandle:
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
        stdout="/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
        stderr="",
        timed_out=False,
        truncated=False,
    )


@pytest.fixture
def stub_registry(stub_handle: SandboxHandle, canned_result: CommandResult, **kwargs: Any) -> Any:
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
    async def test_returns_matches(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        result = await grep_handler("sess_01TEST", {"pattern": "hello"})
        assert result == {
            "matches": "/workspace/foo.py:10:def hello():\n/workspace/bar.py:3:import os\n",
        }

    async def test_default_path(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "rg " in cmd
        assert "/workspace" in cmd

    async def test_include_flag(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "include": "*.py"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "--glob" in cmd
        assert "*.py" in cmd

    async def test_no_matches_returns_empty_string(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = AsyncMock(  # type: ignore[method-assign]
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
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "files_with_matches"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert " -l " in cmd

    async def test_count_mode(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "count"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert " -c " in cmd

    async def test_content_mode_has_line_numbers(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "output_mode": "content"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert " -n " in cmd


class TestAdvancedFlags:
    async def test_context_lines(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "context": 3})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "-C 3" in cmd

    async def test_case_insensitive(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "case_insensitive": True})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert " -i " in cmd

    async def test_multiline(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "multiline": True})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "-U" in cmd
        assert "--multiline-dotall" in cmd

    async def test_file_type_filter(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "file_type": "py"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "--type" in cmd
        assert "py" in cmd

    async def test_custom_limit(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello", "limit": 100})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "head -100" in cmd

    async def test_default_limit_250(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "head -250" in cmd

    async def test_max_columns_flag(self, stub_registry: Any, stub_handle: SandboxHandle) -> None:
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "--max-columns=500" in cmd


class TestErrorPath:
    async def test_grep_failure_returns_error_dict(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=2,
                stdout="",
                stderr="grep: invalid regex\\n",
                timed_out=False,
                truncated=False,
            )
        )
        result = await grep_handler("sess_01TEST", {"pattern": "[invalid"})
        assert "error" in result
        assert "invalid regex" in result["error"]

    async def test_cmd_uses_pipefail_so_rg_failure_propagates(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        """``rg <pattern> <path> 2>/dev/null | head -N`` is a pipe whose final
        exit code is ``head``'s — which is 0 even when ``rg`` failed (bad regex
        ``rg`` exit 2, path traversal error, missing path) because ``head``
        happily consumes empty input and exits 0. Without ``set -o pipefail``
        the result returned to the model is ``{"matches": ""}`` — empty,
        looking like "no matches" rather than the actual rg error. The model
        chases red herrings: tries different paths, different patterns, never
        realizing rg itself rejected its input.

        Stderr is also swallowed via ``2>/dev/null`` so there's nothing to
        surface even if the exit code were correct — but the existing
        ``result.exit_code not in (0, 1)`` branch would convert it to an
        error dict if the exit code propagated.

        Mirrors the read-tool fix in PR #513: prepend ``set -o pipefail`` so
        any non-zero exit anywhere in the pipe is surfaced as the overall exit
        code, which the existing error-detection branch then turns into the
        error dict the model can act on.
        """
        await grep_handler("sess_01TEST", {"pattern": "hello"})
        cmd: str = stub_registry.exec.await_args.args[1]  # type: ignore[attr-defined]
        assert "pipefail" in cmd, (
            f"cmd must enable ``pipefail`` so a failing ``rg`` (e.g., invalid "
            f"regex, missing path) propagates through the ``| head -N`` to "
            f"the overall pipe exit code; without it, rg errors silently "
            f"return empty matches. Got: {cmd!r}"
        )
