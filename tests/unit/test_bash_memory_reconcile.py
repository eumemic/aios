"""Unit tests for bash_memory_reconcile — snapshot + reconcile helpers.

All tests are pure in-memory: no Docker, no Postgres.  The file system
operations use pytest's ``tmp_path`` fixture; DB calls are mocked via
``unittest.mock``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.models.memory_stores import MemoryStoreResourceEcho

SESSION_ID = "sesn_01RECONCILETEST00000000001"
STORE_A = "memstore_01STOREA000000000000000001"
STORE_B = "memstore_01STOREB000000000000000001"


def _echo(
    store_id: str = STORE_A,
    name: str = "notes",
    access: str = "read_write",
) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=store_id,
        access=access,  # type: ignore[arg-type]
        instructions="",
        name=name,
        description="",
        mount_path=f"/mnt/memory/{name}",
    )


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _clear_runtime() -> Any:
    runtime.clear_session_memory_mounts(SESSION_ID)
    runtime.clear_session_read_shas(SESSION_ID)
    yield
    runtime.clear_session_memory_mounts(SESSION_ID)
    runtime.clear_session_read_shas(SESSION_ID)


# ── TestSnapshot ─────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_empty_when_no_mounts(self, tmp_path: Path) -> None:
        """Session has no mounts attached; snapshot returns {}."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=tmp_path):
            result = snapshot_memory_mounts(SESSION_ID)
        assert result == {}

    def test_empty_when_host_dir_missing(self, tmp_path: Path) -> None:
        """Mount in cache but host dir doesn't exist; snapshot returns {}."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        missing = tmp_path / "nonexistent_store"
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=missing):
            result = snapshot_memory_mounts(SESSION_ID)
        assert result == {}

    def test_empty_when_not_materialized(self, tmp_path: Path) -> None:
        """Host dir exists but .materialized marker absent; returns {}."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        # Write a file but no .materialized marker
        (host_dir / "foo.md").write_text("hello")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result = snapshot_memory_mounts(SESSION_ID)
        assert result == {}

    def test_includes_file_sha(self, tmp_path: Path) -> None:
        """Host dir has foo.md with known content; snapshot returns correct sha."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        content = "hello world\n"
        (host_dir / "foo.md").write_text(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result = snapshot_memory_mounts(SESSION_ID)

        expected_sha = _sha256(content)
        assert (STORE_A, "/foo.md") in result
        assert result[(STORE_A, "/foo.md")] == expected_sha

    def test_nested_path(self, tmp_path: Path) -> None:
        """File at a/b/c.md resolves to store path /a/b/c.md."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        nested = host_dir / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "c.md").write_text("nested")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result = snapshot_memory_mounts(SESSION_ID)

        assert (STORE_A, "/a/b/c.md") in result

    def test_skips_read_only_mounts(self, tmp_path: Path) -> None:
        """read_only mount is present; its files are not included."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        (host_dir / "secret.md").write_text("protected")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo(access="read_only")])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result = snapshot_memory_mounts(SESSION_ID)

        assert result == {}

    def test_materialized_marker_itself_excluded(self, tmp_path: Path) -> None:
        """.materialized file not treated as a memory file."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result = snapshot_memory_mounts(SESSION_ID)

        assert result == {}


# ── TestReconcile ─────────────────────────────────────────────────────────────


class TestReconcile:
    """Reconcile tests mock out memory_service functions and pool."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        pool = MagicMock()
        return pool

    @pytest.fixture(autouse=True)
    def _install_pool(self, mock_pool: MagicMock) -> Any:
        prev = runtime.pool
        runtime.pool = mock_pool
        yield
        runtime.pool = prev

    def _make_host_dir(self, tmp_path: Path, store_id: str = STORE_A) -> Path:
        host_dir = tmp_path / store_id
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / ".materialized").touch()
        return host_dir

    async def test_new_file_calls_create_memory(self, tmp_path: Path) -> None:
        """before={}, after has one file; create_memory called with correct args."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "new content\n"
        (host_dir / "new.md").write_text(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            mock_create.return_value = MagicMock(content_sha256=_sha256(content))
            warnings = await reconcile_memory_mounts(SESSION_ID, before={})

        assert warnings == []
        mock_create.assert_awaited_once()
        call_kwargs = mock_create.await_args.kwargs
        assert call_kwargs["store_id"] == STORE_A
        assert call_kwargs["path"] == "/new.md"
        assert call_kwargs["content"] == content

    async def test_modified_file_calls_update_memory(self, tmp_path: Path) -> None:
        """before has sha A, after sha B; update_memory called with precondition_sha256=existing.content_sha256."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        old_content = "old\n"
        new_content = "new\n"
        (host_dir / "mod.md").write_text(new_content)

        old_sha = _sha256(old_content)
        db_sha = _sha256("db-current\n")  # DB's sha may differ from before_sha
        before = {(STORE_A, "/mod.md"): old_sha}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000001"
        fake_memory.content_sha256 = db_sha

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=fake_memory,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
                return_value=fake_memory,
            ) as mock_update,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before=before)

        assert warnings == []
        mock_update.assert_awaited_once()
        call_kwargs = mock_update.await_args.kwargs
        assert call_kwargs["precondition_sha256"] == db_sha
        assert call_kwargs["new_content"] == new_content

    async def test_deleted_file_calls_delete_memory(self, tmp_path: Path) -> None:
        """path in before but not after; delete_memory called."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        # No file written — before has it, after won't
        before = {(STORE_A, "/gone.md"): _sha256("old content\n")}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000002"

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=fake_memory,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before=before)

        assert warnings == []
        mock_delete.assert_awaited_once()
        call_kwargs = mock_delete.await_args.kwargs
        assert call_kwargs["store_id"] == STORE_A
        assert call_kwargs["memory_id"] == fake_memory.id

    async def test_unchanged_file_no_db_call(self, tmp_path: Path) -> None:
        """Same sha in before and after; no create/update/delete called."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "unchanged\n"
        (host_dir / "same.md").write_text(content)
        sha = _sha256(content)
        before = {(STORE_A, "/same.md"): sha}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before=before)

        assert warnings == []
        mock_create.assert_not_awaited()
        mock_update.assert_not_awaited()
        mock_delete.assert_not_awaited()

    async def test_binary_file_skipped_with_warning(self, tmp_path: Path) -> None:
        """Non-UTF-8 bytes; create not called; warning returned."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        # Write binary data that can't be decoded as UTF-8
        (host_dir / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before={})

        assert len(warnings) == 1
        assert "binary.bin" in warnings[0] or "binary" in warnings[0].lower()
        mock_create.assert_not_awaited()

    async def test_oversized_file_skipped_with_warning(self, tmp_path: Path) -> None:
        """File > MAX_CONTENT_BYTES; warning returned, no create."""
        from aios.models.memory_stores import MAX_CONTENT_BYTES
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        big_content = "x" * (MAX_CONTENT_BYTES + 1)
        (host_dir / "big.md").write_text(big_content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before={})

        assert len(warnings) == 1
        assert (
            "big.md" in warnings[0]
            or "size" in warnings[0].lower()
            or "exceeds" in warnings[0].lower()
        )
        mock_create.assert_not_awaited()

    async def test_skips_read_only_mounts_in_reconcile(self, tmp_path: Path) -> None:
        """read_only mount; no DB call even if file changed."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        (host_dir / "file.md").write_text("content\n")

        # read_only mount: even if we have a before sha that differs, no DB call
        before = {(STORE_A, "/file.md"): "different_sha"}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo(access="read_only")])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before=before)

        assert warnings == []
        mock_create.assert_not_awaited()
        mock_update.assert_not_awaited()
        mock_delete.assert_not_awaited()

    async def test_read_sha_updated_after_create(self, tmp_path: Path) -> None:
        """runtime.set_read_sha called after create_memory."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "fresh\n"
        (host_dir / "fresh.md").write_text(content)
        expected_sha = _sha256(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
                return_value=MagicMock(content_sha256=expected_sha),
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            await reconcile_memory_mounts(SESSION_ID, before={})

        cached_sha = runtime.get_read_sha(SESSION_ID, STORE_A, "/fresh.md")
        assert cached_sha == expected_sha

    async def test_modified_file_no_db_record_skips(self, tmp_path: Path) -> None:
        """before has sha A, after sha B, but get_memory_by_path returns None (race); update_memory NOT called."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        new_content = "new content after race\n"
        (host_dir / "race.md").write_text(new_content)

        old_sha = _sha256("old content\n")
        before = {(STORE_A, "/race.md"): old_sha}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_created = MagicMock()
        fake_created.content_sha256 = _sha256(new_content)

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
                return_value=fake_created,
            ) as mock_create,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            warnings = await reconcile_memory_mounts(SESSION_ID, before=before)

        # No crash; update_memory is not called (create_memory is used instead)
        mock_update.assert_not_awaited()
        mock_create.assert_awaited_once()
        # No warning for this path — it is handled gracefully as a create
        assert warnings == []

    async def test_read_sha_updated_after_update(self, tmp_path: Path) -> None:
        """runtime.set_read_sha called after update_memory."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        new_content = "updated\n"
        (host_dir / "upd.md").write_text(new_content)
        old_sha = _sha256("old\n")
        new_sha = _sha256(new_content)
        before = {(STORE_A, "/upd.md"): old_sha}

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKEUPDATETEST0000000001"
        fake_memory.content_sha256 = new_sha

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                new_callable=AsyncMock,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                new_callable=AsyncMock,
                return_value=fake_memory,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.update_memory",
                new_callable=AsyncMock,
                return_value=fake_memory,
            ),
            patch(
                "aios.tools.bash_memory_reconcile.memory_service.delete_memory",
                new_callable=AsyncMock,
            ),
        ):
            await reconcile_memory_mounts(SESSION_ID, before=before)

        cached_sha = runtime.get_read_sha(SESSION_ID, STORE_A, "/upd.md")
        assert cached_sha == new_sha
