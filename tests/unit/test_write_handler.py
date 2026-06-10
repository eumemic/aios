"""Unit tests for the write tool handler.

Mocks SandboxHandle.run_command and inspects the shell command the
handler constructs to verify base64 encoding, parent-dir creation, and
path quoting.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness import runtime
from aios.sandbox.backends.base import CommandResult, SandboxHandle
from aios.tools.write import WriteArgumentError, write_handler


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
        stdout="",
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
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/a.txt", "content": "hello"}
        )
        assert result == {"path": "/workspace/a.txt", "bytes_written": 5}

    async def test_bytes_written_counts_utf8_bytes_not_chars(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        # "héllo" is 6 bytes in UTF-8 (é is 2 bytes) but 5 chars.
        result = await write_handler(
            "sess_01TEST", {"path": "/workspace/a.txt", "content": "héllo"}
        )
        assert result["bytes_written"] == 6

    async def test_command_base64_encodes_content(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        content = "hello world\n"
        await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": content})
        cmd: str = stub_registry.exec.await_args.args[1]
        expected_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        assert f"base64 -d <<< '{expected_b64}'" in cmd

    async def test_command_creates_parent_dirs(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await write_handler("sess_01TEST", {"path": "/workspace/a/b/c.txt", "content": "hi"})
        cmd: str = stub_registry.exec.await_args.args[1]
        # shlex.quote leaves simple paths unquoted; assert mkdir + dirname + path.
        assert "mkdir -p --" in cmd
        assert "dirname --" in cmd
        assert "/workspace/a/b/c.txt" in cmd

    async def test_command_redirects_to_quoted_path(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        await write_handler("sess_01TEST", {"path": "/workspace/a file.txt", "content": "hi"})
        cmd: str = stub_registry.exec.await_args.args[1]
        assert "> '/workspace/a file.txt'" in cmd

    async def test_handles_special_characters_in_content(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        # Content containing quotes, newlines, shell metacharacters.
        tricky = "line with 'quotes' and \"doubles\" and $vars\nand newlines"
        await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": tricky})
        cmd: str = stub_registry.exec.await_args.args[1]
        # Base64 is quote-safe so no escaping gymnastics.
        expected_b64 = base64.b64encode(tricky.encode("utf-8")).decode("ascii")
        assert expected_b64 in cmd
        # No literal quotes or dollar signs leaked into the command.
        assert "'quotes'" not in cmd
        assert "$vars" not in cmd


class TestPerEnvTimeoutCeiling:
    """write routes its sandbox exec through the per-environment bash-timeout
    ceiling (#725), not the hardcoded global default."""

    async def test_exec_uses_resolved_ceiling(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        with patch(
            "aios.tools.write.resolve_bash_timeout_ceiling",
            new_callable=AsyncMock,
            return_value=600,
        ):
            await write_handler("sess_01TEST", {"path": "/workspace/a.txt", "content": "hello"})
        kwargs: dict[str, Any] = stub_registry.exec.await_args.kwargs
        assert kwargs["timeout_seconds"] == 600


class TestErrorPath:
    async def test_nonzero_exit_returns_error_dict(
        self, stub_registry: Any, stub_handle: SandboxHandle
    ) -> None:
        stub_registry.exec = AsyncMock(
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


class TestMemoryReadShaCacheRefresh:
    """A successful write to a memory-mount target must refresh the per-
    session ``read_sha`` cache so a subsequent write doesn't fail its
    precondition check with the now-stale pre-write sha.

    Edit already does this (`tools/edit.py` after `update_memory` /
    `create_memory`); write skipped it. Without the refresh, a model that
    writes the same memory file twice in a row sees its second write
    fail with "the file at X changed since your last read; re-read it
    and retry the write" — but nothing changed externally, the model
    wrote both times itself. The error message is also misleading,
    blaming an external change when the cache is the actual source of
    the staleness.
    """

    @pytest.fixture
    def memory_session(self, stub_registry: Any) -> Any:
        """Attach a memory mount and seed the read_sha cache as if the
        session had read the file once before."""
        import hashlib
        from unittest.mock import AsyncMock, MagicMock, patch

        from aios.models.memory_stores import MemoryStoreResourceEcho
        from aios.services import memory_stores as memory_service

        SESSION = "sess_01TEST"
        STORE = "memstore_01STORE0000000000000000001"
        PATH = "/notes/x.md"
        MOUNT = "/mnt/memory/notes"

        runtime.set_session_memory_mounts(
            SESSION,
            [
                MemoryStoreResourceEcho(
                    memory_store_id=STORE,
                    access="read_write",
                    instructions="",
                    name="notes",
                    description="",
                    mount_path=MOUNT,
                )
            ],
        )

        seed_content = "first body\n"
        seed_sha = hashlib.sha256(seed_content.encode("utf-8")).hexdigest()
        runtime.set_read_sha(SESSION, STORE, PATH, seed_sha)

        existing = MagicMock()
        existing.id = "mem_01EXISTING000000000000000001"
        existing.content_sha256 = seed_sha
        existing.content = seed_content

        with (
            patch.object(
                memory_service,
                "get_memory_by_path",
                AsyncMock(return_value=existing),
            ),
            patch.object(
                memory_service,
                "update_memory",
                AsyncMock(return_value=None),
            ),
        ):
            yield SESSION, STORE, PATH, MOUNT

        runtime.clear_session_memory_mounts(SESSION)
        runtime.clear_session_read_shas(SESSION)

    async def test_write_refreshes_read_sha_to_new_content(
        self, memory_session: tuple[str, str, str, str]
    ) -> None:
        import hashlib

        session, store, store_path, mount_path = memory_session
        new_content = "second body — completely different\n"

        result = await write_handler(
            session,
            {"path": f"{mount_path}{store_path}", "content": new_content},
        )
        assert "error" not in result, result

        expected_sha = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        actual_sha = runtime.get_read_sha(session, store, store_path)
        assert actual_sha == expected_sha, (
            f"write must refresh read_sha to the new content's sha so a "
            f"subsequent write doesn't false-fail its precondition; got "
            f"cached sha {actual_sha!r}, expected {expected_sha!r}. Pre-fix "
            f"the cache still holds the pre-write sha, so the next "
            f"`update_memory(precondition_sha256=stale_sha)` would raise "
            f"MemoryPreconditionFailedError and the handler would return "
            f"the misleading 'file changed since your last read' error."
        )
