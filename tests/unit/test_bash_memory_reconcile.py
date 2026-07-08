"""Unit tests for bash_memory_reconcile — stat-signature snapshot + reconcile (#1748).

All tests are pure in-memory: no Docker, no Postgres. The file system
operations use pytest's ``tmp_path`` fixture; DB calls are mocked via
``unittest.mock``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.tools import bash_memory_telemetry as telemetry

SESSION_ID = "sesn_01RECONCILETEST00000000001"
STORE_A = "memstore_01STOREA000000000000000001"


def _echo(
    store_id: str = STORE_A,
    name: str = "notes",
    access: str = "read_write",
) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=store_id,
        access=access,
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
    telemetry.reset()
    yield
    runtime.clear_session_memory_mounts(SESSION_ID)
    runtime.clear_session_read_shas(SESSION_ID)
    telemetry.reset()


@pytest.fixture(autouse=True)
def _prefilter_enabled() -> Any:
    """Default every test to the "prefilter enabled" state unless overridden."""
    from aios.tools.bash_memory_reconcile import PrefilterState, set_prefilter_state

    set_prefilter_state(PrefilterState(enabled=True, observed_granule_ns=1_000_000))
    yield
    set_prefilter_state(PrefilterState(enabled=True, observed_granule_ns=1_000_000))


def _far_past_ns() -> int:
    """A snapshot_ns far enough in the future that any real file's ctime is 'cold'."""
    return time.time_ns() + 10_000_000_000  # +10s


@contextlib.contextmanager
def _patch_all(mocks: dict[str, Any]) -> Iterator[None]:
    """Apply every ``target -> mock`` pair in ``mocks`` for the block's duration."""
    with contextlib.ExitStack() as stack:
        for target, mock in mocks.items():
            stack.enter_context(patch(target, mock))
        yield


# ── TestSnapshot ─────────────────────────────────────────────────────────────


class TestSnapshot:
    async def test_empty_when_no_mounts(self, tmp_path: Path) -> None:
        """Session has no mounts attached; snapshot returns ({}, ns)."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=tmp_path):
            result, snapshot_ns = snapshot_memory_mounts(SESSION_ID)
        assert result == {}
        assert isinstance(snapshot_ns, int)
        assert snapshot_ns > 0

    async def test_empty_when_host_dir_missing(self, tmp_path: Path) -> None:
        """Mount in cache but host dir doesn't exist; snapshot returns {}."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        missing = tmp_path / "nonexistent_store"
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=missing):
            result, _ = snapshot_memory_mounts(SESSION_ID)
        assert result == {}

    async def test_empty_when_not_materialized(self, tmp_path: Path) -> None:
        """Host dir exists but .materialized marker absent; returns {}."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        # Write a file but no .materialized marker
        (host_dir / "foo.md").write_text("hello")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)
        assert result == {}

    async def test_includes_file_sig(self, tmp_path: Path) -> None:
        """Host dir has foo.md; snapshot returns a stat _Sig (no bytes read)."""
        from aios.tools.bash_memory_reconcile import _Sig, snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        content = "hello world\n"
        fpath = host_dir / "foo.md"
        fpath.write_text(content)
        st = os.stat(fpath, follow_symlinks=False)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert (STORE_A, "/foo.md") in result
        sig = result[(STORE_A, "/foo.md")]
        assert isinstance(sig, _Sig)
        assert sig.size == st.st_size
        assert sig.ino == st.st_ino

    async def test_snapshot_does_not_read_bytes(self, tmp_path: Path) -> None:
        """The stat-only snapshot must never call Path.read_bytes."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        (host_dir / "foo.md").write_text("hello world\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        real_read_bytes = Path.read_bytes

        def _spy_read_bytes(self: Path) -> bytes:
            raise AssertionError(f"snapshot_memory_mounts must not read bytes of {self}")

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch.object(Path, "read_bytes", _spy_read_bytes),
        ):
            snapshot_memory_mounts(SESSION_ID)
        assert real_read_bytes  # sanity: original still bound elsewhere

    async def test_nested_path(self, tmp_path: Path) -> None:
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
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert (STORE_A, "/a/b/c.md") in result

    async def test_skips_read_only_mounts(self, tmp_path: Path) -> None:
        """read_only mount is present; its files are not included."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        (host_dir / "secret.md").write_text("protected")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo(access="read_only")])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert result == {}

    async def test_materialized_marker_itself_excluded(self, tmp_path: Path) -> None:
        """.materialized file not treated as a memory file."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert result == {}


class TestSymlinkRejection:
    """Bash inside the sandbox can ``ln -s <worker-side-path> /mnt/memory/<store>/leak``.
    The host_dir read happens on the WORKER side, so a stat/read on the
    symlinked file resolves the link against the worker's filesystem — which
    exposes any file the worker process can read (vault key material on disk,
    other tenants' state, /etc, etc.) to a memory store that subsequently
    persists the bytes to the DB and renders them to the model. Reject
    symlinks at the walk site, matching #497's policy for
    ``walk_skill_dir`` (same confused-deputy threat class)."""

    def test_snapshot_skips_symlinks_to_host_files(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        sensitive = tmp_path / "host_secret.txt"
        sensitive.write_text("HOST-ONLY SECRET")

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        # Bash inside the sandbox would create this symlink; the target is
        # resolved against the worker's filesystem at read time.
        (host_dir / "leak.txt").symlink_to(sensitive)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert (STORE_A, "/leak.txt") not in result, (
            f"snapshot must skip symlinks; got entries for symlinked paths: "
            f"{[k for k in result if k[1] == '/leak.txt']!r}"
        )

    def test_snapshot_skips_symlinks_to_in_tree_files(self, tmp_path: Path) -> None:
        """Strict rejection — even an in-tree symlink is skipped. Matches the
        #497 policy for skill files: copy the file, don't link it."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        (host_dir / "real.md").write_text("real content")
        (host_dir / "alias.md").symlink_to(host_dir / "real.md")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            result, _ = snapshot_memory_mounts(SESSION_ID)

        assert (STORE_A, "/real.md") in result
        assert (STORE_A, "/alias.md") not in result


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

    def _mocks(self, **overrides: Any) -> dict[str, Any]:
        defaults = {
            "aios.tools.bash_memory_reconcile.memory_service.create_memory": AsyncMock(),
            "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                return_value=None
            ),
            "aios.tools.bash_memory_reconcile.memory_service.update_memory": AsyncMock(),
            "aios.tools.bash_memory_reconcile.memory_service.delete_memory": AsyncMock(),
            "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id": AsyncMock(
                return_value="acct_01FAKE"
            ),
        }
        defaults.update(overrides)
        return defaults

    async def test_new_file_calls_create_memory(self, tmp_path: Path) -> None:
        """before={}, after has one file; create_memory called with correct args."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "new content\n"
        (host_dir / "new.md").write_text(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.create_memory": AsyncMock(
                    return_value=MagicMock(content_sha256=_sha256(content))
                )
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before={}, snapshot_ns=_far_past_ns()
            )

        mock_create = mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"]
        assert warnings == []
        mock_create.assert_awaited_once()
        call_kwargs = mock_create.await_args.kwargs
        assert call_kwargs["store_id"] == STORE_A
        assert call_kwargs["path"] == "/new.md"
        assert call_kwargs["content"] == content

    async def test_modified_file_calls_update_memory(self, tmp_path: Path) -> None:
        """before has a sentinel sig guaranteed to differ, after content differs
        from the DB sha; update_memory called with precondition_sha256=existing.content_sha256."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        new_content = "new\n"
        (host_dir / "mod.md").write_text(new_content)

        db_sha = _sha256("db-current\n")  # DB's sha differs from new content
        # Build a before dict with a fabricated old sig for the SAME path so
        # it counts as "existing" (in ``before``), but stat-differs from
        # whatever the after-scan observes — guaranteeing the modify branch
        # (not the "unchanged" skip) fires.
        from aios.tools.bash_memory_reconcile import _Sig

        before = {(STORE_A, "/mod.md"): _Sig(size=0, mtime_ns=0, ctime_ns=1, ino=0)}

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000001"
        fake_memory.content_sha256 = db_sha

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_memory
                ),
                "aios.tools.bash_memory_reconcile.memory_service.update_memory": AsyncMock(
                    return_value=fake_memory
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mock_update = mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"]
        assert warnings == []
        mock_update.assert_awaited_once()
        call_kwargs = mock_update.await_args.kwargs
        assert call_kwargs["precondition_sha256"] == db_sha
        assert call_kwargs["new_content"] == new_content

    async def test_deleted_file_calls_delete_memory(self, tmp_path: Path) -> None:
        """path in before but not after; delete_memory called."""
        from aios.tools.bash_memory_reconcile import _Sig, reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        # No file written — before has it, after won't
        before = {(STORE_A, "/gone.md"): _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000002"

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_memory
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mock_delete = mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"]
        assert warnings == []
        mock_delete.assert_awaited_once()
        call_kwargs = mock_delete.await_args.kwargs
        assert call_kwargs["store_id"] == STORE_A
        assert call_kwargs["memory_id"] == fake_memory.id

    async def test_unchanged_file_no_db_call(self, tmp_path: Path) -> None:
        """Same sig in before and after, not hot; no create/update/delete called,
        zero candidate reads, zero account_id load."""
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "unchanged\n"
        (host_dir / "same.md").write_text(content)
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, _snapshot_ns = snapshot_memory_mounts(SESSION_ID)

        # snapshot_ns far in the past relative to "now" so the file is not hot.
        cold_snapshot_ns = _far_past_ns()  # far enough ahead that real ctimes are "cold"

        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        mocks = self._mocks()
        real_read_bytes = Path.read_bytes

        def _spy_read_bytes(self: Path) -> bytes:
            raise AssertionError(f"unchanged file must not be read: {self}")

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch.object(Path, "read_bytes", _spy_read_bytes),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=cold_snapshot_ns
            )

        assert real_read_bytes
        assert warnings == []
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"].assert_not_awaited()
        mocks[
            "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id"
        ].assert_not_awaited()
        assert telemetry.last_candidate_read_count() == 0

    async def test_binary_file_skipped_with_warning(self, tmp_path: Path) -> None:
        """Non-UTF-8 bytes; create not called; warning returned."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        # Write binary data that can't be decoded as UTF-8
        (host_dir / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        mocks = self._mocks()

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before={}, snapshot_ns=_far_past_ns()
            )

        assert len(warnings) == 1
        assert "binary.bin" in warnings[0] or "binary" in warnings[0].lower()
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()

    async def test_oversized_file_skipped_with_warning(self, tmp_path: Path) -> None:
        """File > MAX_CONTENT_BYTES; warning returned, no create."""
        from aios.models.memory_stores import MAX_CONTENT_BYTES
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        big_content = "x" * (MAX_CONTENT_BYTES + 1)
        (host_dir / "big.md").write_text(big_content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        mocks = self._mocks()

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before={}, snapshot_ns=_far_past_ns()
            )

        assert len(warnings) == 1
        assert (
            "big.md" in warnings[0]
            or "size" in warnings[0].lower()
            or "exceeds" in warnings[0].lower()
        )
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()

    async def test_unreadable_after_file_not_deleted(self, tmp_path: Path) -> None:
        """File readable at before-snapshot-time but unreadable at after: warned, not deleted."""
        from aios.tools.bash_memory_reconcile import (
            reconcile_memory_mounts,
            snapshot_memory_mounts,
        )

        host_dir = self._make_host_dir(tmp_path)
        notes = host_dir / "notes.md"
        notes.write_text("notes content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, _ = snapshot_memory_mounts(SESSION_ID)
        assert (STORE_A, "/notes.md") in before

        # Make the file unreadable at after-time by patching Path.read_bytes to
        # raise OSError for this specific path (deterministic across platforms).
        # Also mutate its mtime/content so it becomes a candidate even though
        # read fails.
        os.utime(notes, ns=(time.time_ns(), time.time_ns()))
        real_read_bytes = Path.read_bytes

        def _fake_read_bytes(self: Path) -> bytes:
            if self == notes:
                raise OSError("permission denied")
            return real_read_bytes(self)

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000003"

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_memory
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch.object(Path, "read_bytes", _fake_read_bytes),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        assert any("/notes.md" in w for w in warnings)

    async def test_store_unscannable_after_not_deleted(self, tmp_path: Path) -> None:
        """Whole store drops out of the after-scan (echo-cache cleared mid-bash):
        every delete is suppressed and the store is warned — the direct #1705
        regression guard. No account_id load, no DB call at all (pure fast path)."""
        from aios.tools.bash_memory_reconcile import (
            reconcile_memory_mounts,
            snapshot_memory_mounts,
        )

        host_dir = self._make_host_dir(tmp_path)
        (host_dir / "notes.md").write_text("notes content\n")
        (host_dir / "log.md").write_text("log content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, _ = snapshot_memory_mounts(SESSION_ID)
        assert (STORE_A, "/notes.md") in before
        assert (STORE_A, "/log.md") in before

        # Simulate the mount echo-cache clearing between before and reconcile.
        runtime.clear_session_memory_mounts(SESSION_ID)

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKE0000000000000000004"

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_memory
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks[
            "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path"
        ].assert_not_awaited()
        mocks[
            "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id"
        ].assert_not_awaited()
        assert len(warnings) == 1
        assert STORE_A in warnings[0]
        assert "2" in warnings[0]

    async def test_unreadable_in_both_snapshots_is_unchanged(self, tmp_path: Path) -> None:
        """A path unreadable in the before snapshot has the sentinel sig — it is
        ALWAYS a candidate (sentinel never compares equal), so it's re-read at
        after time. If it's STILL unreadable, it's warned once (not silently
        dropped)."""
        from aios.tools.bash_memory_reconcile import (
            reconcile_memory_mounts,
            snapshot_memory_mounts,
        )

        host_dir = self._make_host_dir(tmp_path)
        notes = host_dir / "notes.md"
        notes.write_text("notes content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        real_read_bytes = Path.read_bytes

        def _fake_read_bytes(self: Path) -> bytes:
            if self == notes:
                raise OSError("permission denied")
            return real_read_bytes(self)

        # Sentinel sig doesn't come from read failure anymore (stat-only), so
        # simulate a stat failure at before-time via a patched os.stat.
        real_stat = os.lstat

        def _fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            if Path(path) == notes:
                raise OSError("permission denied (stat)")
            return real_stat(path, *args, **kwargs)

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch("aios.tools.bash_memory_reconcile.os.lstat", _fake_stat),
        ):
            before, _ = snapshot_memory_mounts(SESSION_ID)
        assert (STORE_A, "/notes.md") in before

        mocks = self._mocks()

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch("aios.tools.bash_memory_reconcile.os.lstat", _fake_stat),
            patch.object(Path, "read_bytes", _fake_read_bytes),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"].assert_not_awaited()
        assert any("/notes.md" in w for w in warnings)

    async def test_skips_read_only_mounts_in_reconcile(self, tmp_path: Path) -> None:
        """read_only mount; no DB call — files from read_only mounts never appear in before or after."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        (host_dir / "file.md").write_text("content\n")

        before: dict[tuple[str, str], Any] = {}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo(access="read_only")])
        mocks = self._mocks()

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        assert warnings == []
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.delete_memory"].assert_not_awaited()
        mocks[
            "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id"
        ].assert_not_awaited()

    async def test_read_sha_updated_after_create(self, tmp_path: Path) -> None:
        """runtime.set_read_sha called after create_memory."""
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "fresh\n"
        (host_dir / "fresh.md").write_text(content)
        expected_sha = _sha256(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.create_memory": AsyncMock(
                    return_value=MagicMock(content_sha256=expected_sha)
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            await reconcile_memory_mounts(SESSION_ID, before={}, snapshot_ns=_far_past_ns())

        cached_sha = runtime.get_read_sha(SESSION_ID, STORE_A, "/fresh.md")
        assert cached_sha == expected_sha

    async def test_modified_file_no_db_record_falls_back_to_create(self, tmp_path: Path) -> None:
        """before has the path, after content differs, get_memory_by_path returns
        None (race); falls back to create_memory."""
        from aios.tools.bash_memory_reconcile import _Sig, reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        new_content = "new content after race\n"
        (host_dir / "race.md").write_text(new_content)

        before = {(STORE_A, "/race.md"): _Sig(size=0, mtime_ns=0, ctime_ns=1, ino=0)}
        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_created = MagicMock()
        fake_created.content_sha256 = _sha256(new_content)

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.create_memory": AsyncMock(
                    return_value=fake_created
                ),
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=None
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_awaited_once()
        assert warnings == []

    async def test_read_sha_updated_after_update(self, tmp_path: Path) -> None:
        """runtime.set_read_sha called after update_memory."""
        from aios.tools.bash_memory_reconcile import _Sig, reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        new_content = "updated\n"
        (host_dir / "upd.md").write_text(new_content)
        new_sha = _sha256(new_content)
        db_sha = _sha256("old\n")
        before = {(STORE_A, "/upd.md"): _Sig(size=0, mtime_ns=0, ctime_ns=1, ino=0)}

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        fake_memory = MagicMock()
        fake_memory.id = "mem_01FAKEUPDATETEST0000000001"
        fake_memory.content_sha256 = new_sha
        fake_existing = MagicMock()
        fake_existing.id = "mem_01FAKEUPDATETEST0000000001"
        fake_existing.content_sha256 = db_sha

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_existing
                ),
                "aios.tools.bash_memory_reconcile.memory_service.update_memory": AsyncMock(
                    return_value=fake_memory
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            await reconcile_memory_mounts(SESSION_ID, before=before, snapshot_ns=_far_past_ns())

        cached_sha = runtime.get_read_sha(SESSION_ID, STORE_A, "/upd.md")
        assert cached_sha == new_sha

    async def test_touch_unchanged_content_no_db_write(self, tmp_path: Path) -> None:
        """`touch` (mtime bump, content unchanged): candidate (sig differs) but
        NO DB write, because the candidate's content sha matches the DB sha."""
        from aios.tools.bash_memory_reconcile import _Sig, reconcile_memory_mounts

        host_dir = self._make_host_dir(tmp_path)
        content = "same content\n"
        fpath = host_dir / "touched.md"
        fpath.write_text(content)
        # Force a stat-differing before-sig (simulate `touch` having changed
        # mtime/ctime from some prior state) while content is identical.
        before = {(STORE_A, "/touched.md"): _Sig(size=999, mtime_ns=1, ctime_ns=1, ino=999)}
        db_sha = _sha256(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        fake_existing = MagicMock()
        fake_existing.content_sha256 = db_sha

        mocks = self._mocks(
            **{
                "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path": AsyncMock(
                    return_value=fake_existing
                ),
            }
        )

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            _patch_all(mocks),
        ):
            warnings = await reconcile_memory_mounts(
                SESSION_ID, before=before, snapshot_ns=_far_past_ns()
            )

        mocks["aios.tools.bash_memory_reconcile.memory_service.update_memory"].assert_not_awaited()
        mocks["aios.tools.bash_memory_reconcile.memory_service.create_memory"].assert_not_awaited()
        assert warnings == []
        # Candidate read DID happen (it's a candidate), but no DB write.
        assert telemetry.last_candidate_read_count() == 1
        # read-sha cache still stamped so a later session write isn't
        # spuriously treated as unread.
        assert runtime.get_read_sha(SESSION_ID, STORE_A, "/touched.md") == db_sha


class TestSoundPrefilter:
    """Direct unit tests of the change-detection predicate + hot window,
    independent of the full async reconcile plumbing."""

    def test_new_path_is_always_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        after = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        assert _is_candidate(None, after, snapshot_ns=0, force_hash=False) is True

    def test_sig_equal_not_hot_is_not_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        sig = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        # snapshot_ns far bigger than ctime_ns + HOT_WINDOW_NS -> not hot
        assert _is_candidate(sig, sig, snapshot_ns=10_000_000_000, force_hash=False) is False

    def test_sig_equal_but_hot_is_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import HOT_WINDOW_NS, _is_candidate, _Sig

        ctime_ns = 5_000_000_000
        sig = _Sig(size=1, mtime_ns=1, ctime_ns=ctime_ns, ino=1)
        # snapshot_ns within HOT_WINDOW_NS of ctime_ns -> hot
        snapshot_ns = ctime_ns + HOT_WINDOW_NS - 1
        assert _is_candidate(sig, sig, snapshot_ns=snapshot_ns, force_hash=False) is True

    def test_sentinel_before_is_always_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _SENTINEL_SIG, _is_candidate, _Sig

        after = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        assert _is_candidate(_SENTINEL_SIG, after, snapshot_ns=0, force_hash=False) is True

    def test_sentinel_after_is_always_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _SENTINEL_SIG, _is_candidate, _Sig

        before = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        assert _is_candidate(before, _SENTINEL_SIG, snapshot_ns=0, force_hash=False) is True

    def test_sig_differs_is_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        before = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        after = _Sig(
            size=2, mtime_ns=1, ctime_ns=1, ino=1
        )  # touch -r style: mtime same, ctime same, size differs
        assert _is_candidate(before, after, snapshot_ns=10_000_000_000, force_hash=False) is True

    def test_ctime_bump_same_size_mtime_is_candidate(self) -> None:
        """same-size in-place edit (dd conv=notrunc): ctime bumps -> candidate."""
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        before = _Sig(size=10, mtime_ns=100, ctime_ns=100, ino=1)
        after = _Sig(size=10, mtime_ns=100, ctime_ns=200, ino=1)
        assert _is_candidate(before, after, snapshot_ns=10_000_000_000, force_hash=False) is True

    def test_atomic_replace_new_inode_is_candidate(self) -> None:
        """os.replace / atomic_mirror: new inode + ctime -> candidate."""
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        before = _Sig(size=10, mtime_ns=100, ctime_ns=100, ino=1)
        after = _Sig(size=10, mtime_ns=300, ctime_ns=300, ino=2)
        assert _is_candidate(before, after, snapshot_ns=10_000_000_000, force_hash=False) is True

    def test_force_hash_makes_everything_a_candidate(self) -> None:
        from aios.tools.bash_memory_reconcile import _is_candidate, _Sig

        sig = _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1)
        assert _is_candidate(sig, sig, snapshot_ns=10_000_000_000, force_hash=True) is True


class TestSnapshotNsContract:
    """Lens 0 #2: snapshot_ns must be captured pre-exec, not re-sampled in the
    after-scan — a post-exec sample would silently disable the hot-window net
    for any exec longer than HOT_WINDOW_NS."""

    async def test_snapshot_ns_captured_before_scan_after_call(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import snapshot_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])
        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()

        before_call_ns = time.time_ns()
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            _, snapshot_ns = snapshot_memory_mounts(SESSION_ID)
        after_call_ns = time.time_ns()

        assert before_call_ns <= snapshot_ns <= after_call_ns

    async def test_long_running_exec_still_marks_recent_write_hot(self, tmp_path: Path) -> None:
        """Model a bash call that runs longer than HOT_WINDOW_NS: a file
        modified just before snapshot_ns (pre-exec) must still be 'hot' at
        after-scan time, i.e. always a candidate regardless of the after
        stat — proving snapshot_ns is NOT re-sampled post-exec."""
        from aios.tools.bash_memory_reconcile import (
            HOT_WINDOW_NS,
            _is_candidate,
            _Sig,
        )

        # File modified 0.5s before the (pre-exec) snapshot.
        ctime_ns = 10_000_000_000
        snapshot_ns = ctime_ns + 500_000_000  # snapshot taken 0.5s after write
        sig = _Sig(size=1, mtime_ns=ctime_ns, ctime_ns=ctime_ns, ino=1)

        # Even though "now" (post a long exec) is way later, the candidate
        # decision uses the ORIGINAL pre-exec snapshot_ns, so this file
        # (modified 0.5s before it) is still within HOT_WINDOW_NS and hot.
        assert snapshot_ns - ctime_ns < HOT_WINDOW_NS
        assert _is_candidate(sig, sig, snapshot_ns=snapshot_ns, force_hash=False) is True


class TestAfterSigsCompleteness:
    """Lens 0 #3: after_sigs must contain EVERY walked path, including stat
    failures, so the delete-diff never keys off the (smaller) candidate set."""

    async def test_unchanged_sibling_survives_transient_stat_failure(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import _scan_after

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        changed = host_dir / "changed.md"
        changed.write_text("new\n")
        sibling = host_dir / "sibling.md"
        sibling.write_text("sibling content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        real_stat = os.lstat

        def _flaky_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            if Path(path) == sibling:
                raise OSError("transient stat failure")
            return real_stat(path, *args, **kwargs)

        before: dict[tuple[str, str], Any] = {}

        with (
            patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir),
            patch("aios.tools.bash_memory_reconcile.os.lstat", _flaky_stat),
        ):
            scan = _scan_after(SESSION_ID, before, snapshot_ns=0, force_hash=False)

        # sibling.md is present in after_sigs (under the sentinel), NOT absent.
        assert (STORE_A, "/sibling.md") in scan.after_sigs
        assert (STORE_A, "/changed.md") in scan.after_sigs

    async def test_sibling_not_deleted_end_to_end(self, tmp_path: Path) -> None:
        """Full reconcile: a store with one genuinely-changed file and one
        unchanged sibling that transiently fails os.stat in the after-pass —
        the sibling must NOT be deleted."""
        from aios.tools.bash_memory_reconcile import _Sig, reconcile_memory_mounts

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        changed = host_dir / "changed.md"
        changed.write_text("new content\n")
        sibling = host_dir / "sibling.md"
        sibling.write_text("sibling content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        # before has both paths with sentinel-ish sigs guaranteed to differ,
        # AND the sibling is genuinely unchanged on disk (we just want to
        # simulate the transient stat failure independent of change status).
        before = {
            (STORE_A, "/changed.md"): _Sig(size=0, mtime_ns=0, ctime_ns=1, ino=0),
            (STORE_A, "/sibling.md"): _Sig(size=0, mtime_ns=0, ctime_ns=1, ino=0),
        }

        real_stat = os.lstat

        def _flaky_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            if Path(path) == sibling:
                raise OSError("transient stat failure")
            return real_stat(path, *args, **kwargs)

        pool = MagicMock()
        prev_pool = runtime.pool
        runtime.pool = pool

        fake_created = MagicMock()
        fake_created.content_sha256 = _sha256("new content\n")

        try:
            with (
                patch(
                    "aios.tools.bash_memory_reconcile.memory_store_host_dir",
                    return_value=host_dir,
                ),
                patch("aios.tools.bash_memory_reconcile.os.lstat", _flaky_stat),
                patch(
                    "aios.tools.bash_memory_reconcile.memory_service.create_memory",
                    new_callable=AsyncMock,
                    return_value=fake_created,
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
                ) as mock_delete,
                patch(
                    "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                    new_callable=AsyncMock,
                    return_value="acct_01FAKE",
                ),
            ):
                await reconcile_memory_mounts(SESSION_ID, before=before, snapshot_ns=_far_past_ns())
        finally:
            runtime.pool = prev_pool

        # sibling.md must NOT be deleted — it's present in after_sigs (under
        # the stat-failure sentinel), so the delete-diff never fires for it.
        mock_delete.assert_not_awaited()


class TestUnscannableStoreOnNoOpCall:
    """Lens 2 #5: the #1705 warn must survive the no-op fast-path."""

    async def test_unscannable_store_warns_and_makes_no_db_call_on_no_op(
        self, tmp_path: Path
    ) -> None:
        from aios.tools.bash_memory_reconcile import (
            _Sig,
            reconcile_memory_mounts,
        )

        # before has entries for a store that will NOT be scannable in the
        # after-pass (mount cache cleared) — the whole call is otherwise a
        # no-op (no other stores, no writes).
        before = {
            (STORE_A, "/a.md"): _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1),
            (STORE_A, "/b.md"): _Sig(size=1, mtime_ns=1, ctime_ns=1, ino=1),
        }
        # No mounts attached at reconcile time -> store not in scanned set.
        runtime.set_session_memory_mounts(SESSION_ID, [])

        pool = MagicMock()
        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            with patch(
                "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                new_callable=AsyncMock,
            ) as mock_load_account:
                warnings = await reconcile_memory_mounts(
                    SESSION_ID, before=before, snapshot_ns=_far_past_ns()
                )
        finally:
            runtime.pool = prev_pool

        mock_load_account.assert_not_awaited()
        assert len(warnings) == 1
        assert STORE_A in warnings[0]
        assert "2" in warnings[0]


class TestCoarseAbsentCtimeGuard:
    """#1748 §6: fail-closed force-hash on absent/coarse ctime."""

    def test_zero_ctime_sig_is_sentinel(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import _SENTINEL_SIG, _stat_sig

        fpath = tmp_path / "f.md"
        fpath.write_text("x")

        class _FakeStatResult:
            st_size = 1
            st_mtime_ns = 123
            st_ctime_ns = 0
            st_ino = 1

        with patch("aios.tools.bash_memory_reconcile.os.lstat", return_value=_FakeStatResult()):
            sig = _stat_sig(fpath)
        assert sig is _SENTINEL_SIG

    def test_stat_oserror_is_sentinel(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import _SENTINEL_SIG, _stat_sig

        fpath = tmp_path / "missing.md"

        with patch("aios.tools.bash_memory_reconcile.os.lstat", side_effect=OSError("boom")):
            sig = _stat_sig(fpath)
        assert sig is _SENTINEL_SIG

    def test_probe_disables_prefilter_on_absent_ctime(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import probe_mount_ctime_granularity

        real_stat = os.lstat

        def _fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            st = real_stat(path, *args, **kwargs)

            class _Wrapped:
                st_size = st.st_size
                st_mtime_ns = st.st_mtime_ns
                st_ctime_ns = 0
                st_ino = st.st_ino

            return _Wrapped()

        with patch("aios.tools.bash_memory_reconcile.os.lstat", _fake_stat):
            state = probe_mount_ctime_granularity(tmp_path)

        assert state.enabled is False

    def test_probe_disables_prefilter_on_coarse_granule(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import (
            HOT_WINDOW_NS,
            probe_mount_ctime_granularity,
        )

        calls = {"n": 0}
        real_stat = os.lstat

        def _fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            st = real_stat(path, *args, **kwargs)
            calls["n"] += 1

            class _Wrapped:
                st_size = st.st_size
                st_mtime_ns = st.st_mtime_ns
                # First call (post-first-write) returns a fixed value; second
                # call (post-second-write) returns a value >= HOT_WINDOW_NS
                # later, simulating a coarse-granule FS.
                st_ctime_ns = 1_000_000_000 if calls["n"] == 1 else 1_000_000_000 + HOT_WINDOW_NS
                st_ino = st.st_ino

            return _Wrapped()

        with patch("aios.tools.bash_memory_reconcile.os.lstat", _fake_stat):
            state = probe_mount_ctime_granularity(tmp_path)

        assert state.enabled is False
        assert state.observed_granule_ns is not None
        assert state.observed_granule_ns >= HOT_WINDOW_NS

    def test_probe_enables_prefilter_on_fine_granule(self, tmp_path: Path) -> None:
        from aios.tools.bash_memory_reconcile import probe_mount_ctime_granularity

        calls = {"n": 0}
        real_stat = os.lstat

        def _fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            st = real_stat(path, *args, **kwargs)
            calls["n"] += 1

            class _Wrapped:
                st_size = st.st_size
                st_mtime_ns = st.st_mtime_ns
                st_ctime_ns = 1_000_000_000 if calls["n"] == 1 else 1_000_001_000
                st_ino = st.st_ino

            return _Wrapped()

        with patch("aios.tools.bash_memory_reconcile.os.lstat", _fake_stat):
            state = probe_mount_ctime_granularity(tmp_path)

        assert state.enabled is True

    async def test_force_hash_path_taken_when_prefilter_disabled(self, tmp_path: Path) -> None:
        """When the cached prefilter state is disabled, an unchanged file
        (sig-equal, not hot) is STILL read/hashed (force_hash escape)."""
        from aios.tools.bash_memory_reconcile import (
            PrefilterState,
            reconcile_memory_mounts,
            set_prefilter_state,
            snapshot_memory_mounts,
        )

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        content = "same\n"
        (host_dir / "same.md").write_text(content)

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, _ = snapshot_memory_mounts(SESSION_ID)

        set_prefilter_state(PrefilterState(enabled=False, observed_granule_ns=5_000_000_000))
        try:
            pool = MagicMock()
            prev_pool = runtime.pool
            runtime.pool = pool
            db_sha = _sha256(content)
            fake_existing = MagicMock()
            fake_existing.content_sha256 = db_sha
            try:
                with (
                    patch(
                        "aios.tools.bash_memory_reconcile.memory_store_host_dir",
                        return_value=host_dir,
                    ),
                    patch(
                        "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                        new_callable=AsyncMock,
                        return_value=fake_existing,
                    ),
                    patch(
                        "aios.tools.bash_memory_reconcile.memory_service.create_memory",
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
                    patch(
                        "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                        new_callable=AsyncMock,
                        return_value="acct_01FAKE",
                    ),
                ):
                    await reconcile_memory_mounts(
                        SESSION_ID, before=before, snapshot_ns=_far_past_ns()
                    )
            finally:
                runtime.pool = prev_pool
        finally:
            set_prefilter_state(PrefilterState(enabled=True, observed_granule_ns=1_000_000))

        # force_hash means it WAS read even though sig-equal/not-hot.
        assert telemetry.last_candidate_read_count() == 1


class TestUnreadableUnchanged:
    """Lens 0 #6: a file that's read-failing but STAT-succeeding has a stable
    valid sig -> skipped entirely, no read attempt, no per-call warning (an
    improvement over master, which re-read it every call)."""

    async def test_stat_succeeding_read_failing_file_is_skipped_no_warning(
        self, tmp_path: Path
    ) -> None:
        from aios.tools.bash_memory_reconcile import (
            reconcile_memory_mounts,
            snapshot_memory_mounts,
        )

        host_dir = tmp_path / STORE_A
        host_dir.mkdir()
        (host_dir / ".materialized").touch()
        notes = host_dir / "notes.md"
        notes.write_text("notes content\n")

        runtime.set_session_memory_mounts(SESSION_ID, [_echo()])

        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, _ = snapshot_memory_mounts(SESSION_ID)

        # stat still succeeds (sig stable across before/after); read fails,
        # but since sig doesn't change and isn't hot, read is never attempted.
        cold_snapshot_ns = _far_past_ns()

        def _spy_read_bytes(self: Path) -> bytes:
            raise AssertionError(f"unchanged-sig file must not be read: {self}")

        pool = MagicMock()
        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            with (
                patch(
                    "aios.tools.bash_memory_reconcile.memory_store_host_dir",
                    return_value=host_dir,
                ),
                patch.object(Path, "read_bytes", _spy_read_bytes),
                patch(
                    "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                    new_callable=AsyncMock,
                ) as mock_load_account,
            ):
                warnings = await reconcile_memory_mounts(
                    SESSION_ID, before=before, snapshot_ns=cold_snapshot_ns
                )
        finally:
            runtime.pool = prev_pool

        assert warnings == []
        mock_load_account.assert_not_awaited()


class TestFastPath:
    """No writable mounts at all -> reconcile is a pure no-op, no to_thread work
    observable via telemetry, no account load."""

    async def test_no_mounts_skips_account_load(self) -> None:
        from aios.tools.bash_memory_reconcile import reconcile_memory_mounts

        runtime.set_session_memory_mounts(SESSION_ID, [])
        pool = MagicMock()
        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            with patch(
                "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                new_callable=AsyncMock,
            ) as mock_load_account:
                warnings = await reconcile_memory_mounts(SESSION_ID, before={}, snapshot_ns=0)
        finally:
            runtime.pool = prev_pool

        assert warnings == []
        mock_load_account.assert_not_awaited()
        assert telemetry.last_candidate_read_count() == 0
