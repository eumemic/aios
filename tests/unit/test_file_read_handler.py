"""Unit tests for the read tool handler.

Stubs the sandbox registry and injects a test ShellFileOperations so
the tests don't touch Docker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.tools import file_session
from aios.tools.read import ReadArgumentError, read_handler
from aios.vendor.hermes_files.file_operations import ReadResult


class _StubFileOps:
    """Stand-in for ShellFileOperations with recorded calls and canned results."""

    def __init__(self) -> None:
        self.read_calls: list[tuple[str, int, int]] = []
        self._read_result = ReadResult(
            content="     1|hello\n     2|world",
            total_lines=2,
            file_size=12,
            truncated=False,
        )
        self._mtime: float | None = 1234567890.0

    def set_read_result(self, result: ReadResult) -> None:
        self._read_result = result

    def set_mtime(self, mtime: float | None) -> None:
        self._mtime = mtime

    async def read_file(self, path: str, offset: int, limit: int) -> ReadResult:
        self.read_calls.append((path, offset, limit))
        return self._read_result

    async def _exec(self, command: str, **kwargs: Any) -> Any:
        class _R:
            def __init__(self, stdout: str, exit_code: int) -> None:
                self.stdout = stdout
                self.exit_code = exit_code

        if "stat -c %Y" in command:
            if self._mtime is None:
                return _R(stdout="", exit_code=1)
            return _R(stdout=f"{self._mtime}\n", exit_code=0)
        return _R(stdout="", exit_code=0)

    def _escape_shell_arg(self, arg: str) -> str:
        return "'" + arg.replace("'", "'\"'\"'") + "'"


@pytest.fixture(autouse=True)
def _reset_file_sessions() -> None:
    """Clear the module-level file-session registry around each test."""
    file_session._sessions.clear()
    yield
    file_session._sessions.clear()


@pytest.fixture
def stub_fileops() -> _StubFileOps:
    return _StubFileOps()


@pytest.fixture
def stub_session(stub_fileops: _StubFileOps) -> file_session.FileToolSession:
    """Install a pre-built FileToolSession so get_or_create returns it."""
    sess = file_session.FileToolSession(file_ops=stub_fileops)  # type: ignore[arg-type]
    file_session._sessions["sess_01TEST"] = sess
    return sess


@pytest.fixture
def _session(stub_session: file_session.FileToolSession) -> None:
    """Marker fixture -- tests that need the session installed depend on this."""
    _ = stub_session


class TestArguments:
    async def test_missing_path_raises(self, _session: None) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {})

    async def test_empty_path_raises(self, _session: None) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "   "})

    async def test_non_string_path_raises(self, _session: None) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": 123})

    async def test_zero_offset_raises(self, _session: None) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "/workspace/x.txt", "offset": 0})

    async def test_negative_limit_raises(self, _session: None) -> None:
        with pytest.raises(ReadArgumentError):
            await read_handler("sess_01TEST", {"path": "/workspace/x.txt", "limit": -1})


class TestDeviceBlocklist:
    async def test_dev_zero_blocked(self, _session: None) -> None:
        result = await read_handler("sess_01TEST", {"path": "/dev/zero"})
        assert "error" in result
        assert "device file" in result["error"].lower()

    async def test_dev_stdin_blocked(self, _session: None) -> None:
        result = await read_handler("sess_01TEST", {"path": "/dev/stdin"})
        assert "error" in result

    async def test_proc_fd_blocked(self, _session: None) -> None:
        result = await read_handler("sess_01TEST", {"path": "/proc/self/fd/0"})
        assert "error" in result

    async def test_regular_path_not_blocked(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await read_handler("sess_01TEST", {"path": "/workspace/ok.txt"})
        assert "error" not in result
        assert stub_fileops.read_calls == [("/workspace/ok.txt", 1, 500)]


class TestBinaryExtension:
    async def test_png_rejected(self, _session: None) -> None:
        result = await read_handler("sess_01TEST", {"path": "/workspace/logo.png"})
        assert "error" in result
        assert "binary" in result["error"].lower()

    async def test_exe_rejected(self, _session: None) -> None:
        result = await read_handler("sess_01TEST", {"path": "/workspace/a.exe"})
        assert "error" in result

    async def test_txt_allowed(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert "error" not in result


class TestCharLimit:
    async def test_oversize_rejected(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_read_result(
            ReadResult(
                content="x" * 200_000,
                total_lines=1,
                file_size=200_000,
                truncated=False,
            )
        )
        result = await read_handler("sess_01TEST", {"path": "/workspace/big.txt"})
        assert "error" in result
        assert "exceeds" in result["error"].lower()
        assert "200,000" in result["error"]
        assert "100,000" in result["error"]


class TestDedup:
    async def test_second_identical_read_returns_dedup_stub(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        r1 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert "dedup" not in r1

        r2 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert r2.get("dedup") is True


class TestConsecutiveLoop:
    async def test_three_reads_warn(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        # Moving mtime so dedup doesn't short-circuit.
        mtimes = iter([1.0, 2.0, 3.0, 4.0, 5.0])

        async def mtime_op(command: str, **kwargs: Any) -> Any:
            class _R:
                def __init__(self, stdout: str, exit_code: int) -> None:
                    self.stdout = stdout
                    self.exit_code = exit_code

            if "stat -c %Y" in command:
                return _R(stdout=f"{next(mtimes)}\n", exit_code=0)
            return _R(stdout="", exit_code=0)

        stub_fileops._exec = mtime_op  # type: ignore[assignment,method-assign]

        r1 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        r2 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        r3 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert "_warning" not in r1
        assert "_warning" not in r2
        assert "_warning" in r3
        assert "consecutively" in r3["_warning"]

    async def test_four_reads_block(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        counter = [0.0]

        async def mtime_op(command: str, **kwargs: Any) -> Any:
            class _R:
                def __init__(self, stdout: str, exit_code: int) -> None:
                    self.stdout = stdout
                    self.exit_code = exit_code

            if "stat -c %Y" in command:
                counter[0] += 1.0
                return _R(stdout=f"{counter[0]}\n", exit_code=0)
            return _R(stdout="", exit_code=0)

        stub_fileops._exec = mtime_op  # type: ignore[assignment,method-assign]

        for _ in range(3):
            await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        r4 = await read_handler("sess_01TEST", {"path": "/workspace/a.txt"})
        assert "error" in r4
        assert "BLOCKED" in r4["error"]


class TestLazyFileSessionCreation:
    async def test_no_session_installed_walks_sandbox_registry(self) -> None:
        """If there is no pre-installed session, get_or_create walks the
        sandbox registry. Stub the registry here to exercise that path."""

        class _StubSandboxRegistry:
            def __init__(self, handle: ContainerHandle) -> None:
                self._handle = handle

            async def get_or_provision(self, session_id: str) -> ContainerHandle:
                return self._handle

        handle = ContainerHandle(
            session_id="sess_01TEST2",
            container_id="container_x",
            workspace_path=Path("/tmp/aios-test"),
        )
        handle.run_command = AsyncMock(  # type: ignore[method-assign]
            return_value=CommandResult(
                exit_code=0,
                stdout="hello",
                stderr="",
                timed_out=False,
                truncated=False,
            )
        )
        previous = runtime.sandbox_registry
        runtime.sandbox_registry = _StubSandboxRegistry(handle)  # type: ignore[assignment]
        try:
            # Binary-extension guard rejects the read before it touches
            # the real ShellFileOperations -- but get_or_create has to
            # walk through the sandbox registry to build the session in
            # the first place. The fact that we get the expected error
            # dict means that path succeeded.
            result = await read_handler("sess_01TEST2", {"path": "/workspace/a.png"})
            assert "error" in result
            assert "binary" in result["error"].lower()
        finally:
            runtime.sandbox_registry = previous
