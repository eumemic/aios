"""Unit tests for the write tool handler.

Stubs the sandbox registry and injects a test ShellFileOperations so
the tests don't touch Docker.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.tools import file_session
from aios.tools.write import WriteArgumentError, write_handler
from aios.vendor.hermes_files.file_operations import WriteResult


class _StubFileOps:
    def __init__(self) -> None:
        self.write_calls: list[tuple[str, str]] = []
        self._write_result = WriteResult(bytes_written=5, dirs_created=False)
        self._mtime: float | None = 1000.0

    def set_write_result(self, result: WriteResult) -> None:
        self._write_result = result

    def set_mtime(self, mtime: float | None) -> None:
        self._mtime = mtime

    async def write_file(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        return self._write_result

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
    file_session._sessions.clear()
    yield
    file_session._sessions.clear()


@pytest.fixture
def stub_fileops() -> _StubFileOps:
    return _StubFileOps()


@pytest.fixture
def stub_session(stub_fileops: _StubFileOps) -> file_session.FileToolSession:
    sess = file_session.FileToolSession(file_ops=stub_fileops)  # type: ignore[arg-type]
    file_session._sessions["sess_01TEST"] = sess
    return sess


class TestArguments:
    async def test_missing_path_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"content": "hi"})

    async def test_missing_content_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"path": "/workspace/a.txt"})

    async def test_non_string_content_raises(
        self, stub_session: file_session.FileToolSession
    ) -> None:
        with pytest.raises(WriteArgumentError):
            await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": 42})


class TestSensitivePath:
    async def test_etc_passwd_rejected(self, stub_session: file_session.FileToolSession) -> None:
        result = await write_handler("sess_01TEST", {"path": "/etc/passwd", "content": "evil"})
        assert "error" in result
        assert "sensitive" in result["error"].lower()

    async def test_boot_rejected(self, stub_session: file_session.FileToolSession) -> None:
        result = await write_handler("sess_01TEST", {"path": "/boot/grub.cfg", "content": "evil"})
        assert "error" in result

    async def test_docker_sock_rejected(self, stub_session: file_session.FileToolSession) -> None:
        result = await write_handler(
            "sess_01TEST",
            {"path": "/var/run/docker.sock", "content": "evil"},
        )
        assert "error" in result

    async def test_workspace_path_allowed(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/a.txt", "content": "hello"}
        )
        assert "error" not in result
        assert stub_fileops.write_calls == [("/workspace/a.txt", "hello")]


class TestCharLimit:
    async def test_oversize_rejected(self, stub_session: file_session.FileToolSession) -> None:
        content = "x" * 2_000_000
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/big.txt", "content": content}
        )
        assert "error" in result
        assert "exceeds" in result["error"].lower()


class TestResultPassthrough:
    async def test_bytes_written_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_write_result(WriteResult(bytes_written=42, dirs_created=True))
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/subdir/a.txt", "content": "hi"}
        )
        assert result["bytes_written"] == 42
        assert result["dirs_created"] is True

    async def test_write_error_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_write_result(WriteResult(error="disk full"))
        result = await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": "hi"})
        assert "error" in result


class TestStalenessWarning:
    async def test_stale_file_warns(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        # Simulate the read tool having recorded a prior mtime.
        import os

        normpath = os.path.normpath("/workspace/a.txt")
        stub_session.read_timestamps[normpath] = 1000.0
        # Now the file has a DIFFERENT mtime -> staleness warning.
        stub_fileops.set_mtime(2000.0)

        result = await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": "hi"})
        assert result.get("_warning") is not None
        assert "modified externally" in result["_warning"]

    async def test_fresh_file_no_warning(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        import os

        normpath = os.path.normpath("/workspace/a.txt")
        stub_session.read_timestamps[normpath] = 1000.0
        stub_fileops.set_mtime(1000.0)  # unchanged

        result = await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": "hi"})
        assert "_warning" not in result
