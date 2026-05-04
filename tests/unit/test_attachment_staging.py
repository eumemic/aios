"""Unit coverage for :mod:`aios.harness.attachment_staging`.

Exercises the rename/copy/cleanup state machine with a temp
``workspace_root`` so we never touch the production attachments
directory. EXDEV is simulated by monkeypatching ``os.rename`` rather
than spinning up a second filesystem.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.harness.attachment_staging import (
    AttachmentStagingError,
    _safe_filename,
    stage_inbound_attachments,
)


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the cached ``Settings`` at a tmpdir for the test.

    Mirrors the pattern in :mod:`tests.unit.test_memory_store_host_dir`:
    mutate the lru_cached ``Settings`` instance directly so every
    ``get_settings()`` call site sees the test root.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


def _make_temp_attachment(tmp_path: Path, name: str, payload: bytes) -> dict[str, Any]:
    """Build a wire-shaped attachment record pointing at a real file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / name
    src.write_bytes(payload)
    return {
        "host_path": str(src),
        "filename": name,
        "content_type": "image/jpeg",
        "size": len(payload),
    }


class TestSafeFilename:
    def test_strips_path_separators(self) -> None:
        # Defeats ``../../etc/passwd`` style traversal attempts.
        assert _safe_filename("../../etc/passwd") == "passwd"

    def test_replaces_unsafe_chars(self) -> None:
        assert _safe_filename("hello world!.jpg") == "hello_world_.jpg"

    def test_preserves_dots_and_dashes(self) -> None:
        assert _safe_filename("photo-2026.05.04.jpg") == "photo-2026.05.04.jpg"

    def test_empty_falls_back_to_unnamed(self) -> None:
        assert _safe_filename("") == "unnamed"

    def test_all_dots_falls_back_to_unnamed(self) -> None:
        assert _safe_filename("...") == "unnamed"

    def test_caps_length(self) -> None:
        result = _safe_filename("a" * 500)
        assert len(result) <= 200


class TestStaging:
    def test_empty_returns_empty(self, temp_workspace_root: Path) -> None:
        assert stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=None,
        ) == ([], [])
        assert stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[],
        ) == ([], [])

    def test_happy_path_renames_file_and_returns_record(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        connector_temp = tmp_path / "connector-temp"
        connector_temp.mkdir()
        att = _make_temp_attachment(connector_temp, "photo.jpg", b"jpegbytes")

        records, staged_paths = stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[att],
        )

        # Wire record carries the in-sandbox path the model will see.
        assert records == [
            {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size": len(b"jpegbytes"),
                "in_sandbox_path": "/mnt/attachments/echo/evt-1-photo.jpg",
            }
        ]
        # File moved to staged path; original temp gone.
        staged = temp_workspace_root / "_attachments" / "sess-1" / "echo" / "evt-1-photo.jpg"
        assert staged.exists()
        assert staged.read_bytes() == b"jpegbytes"
        assert not Path(att["host_path"]).exists()
        # The newly-staged path list lets the supervisor unlink on
        # post-staging dedup failure.
        assert staged_paths == [staged]

    def test_multiple_attachments_all_staged(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        a = _make_temp_attachment(tmp_path, "a.jpg", b"AAA")
        b = _make_temp_attachment(tmp_path, "b.png", b"BBBB")
        b["content_type"] = "image/png"

        records, staged_paths = stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[a, b],
        )

        assert len(records) == 2
        assert records[0]["filename"] == "a.jpg"
        assert records[1]["filename"] == "b.png"
        assert len(staged_paths) == 2

    def test_unsafe_filename_sanitized_in_staged_path(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        # Connector reports a filename with unsafe chars; original temp
        # is at a sane path — the unsafe chars only affect the staged name.
        src = tmp_path / "evil.jpg"
        src.write_bytes(b"x")
        att = {
            "host_path": str(src),
            "filename": "../escape/bad name.jpg",
            "content_type": "image/jpeg",
            "size": 1,
        }
        records, _ = stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[att],
        )
        # Filename in the *record* is the original (model-facing display),
        # the sanitized form only shows up in the staged path.
        assert records[0]["filename"] == "../escape/bad name.jpg"
        assert records[0]["in_sandbox_path"] == "/mnt/attachments/echo/evt-1-bad_name.jpg"

    def test_replay_with_existing_target_skips_rename(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        """Idempotent replay: same event_id delivered twice doesn't double-stage."""
        a = _make_temp_attachment(tmp_path, "photo.jpg", b"first")
        first_records, first_paths = stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[a],
        )
        # First call consumed the temp file; for the replay the connector
        # would re-supply the spool entry but its temp file is gone. The
        # function must still succeed (target already at the staged path).
        a2 = dict(a)  # same params dict the SDK would re-emit from spool
        second_records, second_paths = stage_inbound_attachments(
            session_id="sess-1",
            connector_name="echo",
            event_id="evt-1",
            raw_attachments=[a2],
        )
        # Records match (same event_id → same in_sandbox_path).
        assert first_records == second_records
        # First call materialized one path; replay materialized none —
        # critical so the supervisor's NotFoundError compensating
        # unlink doesn't blow away bytes already referenced by the
        # previously committed event.
        assert len(first_paths) == 1
        assert second_paths == []

    def test_missing_temp_path_raises_and_cleans_up(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        good = _make_temp_attachment(tmp_path, "good.jpg", b"good")
        bogus = {
            "host_path": str(tmp_path / "does-not-exist.jpg"),
            "filename": "missing.jpg",
            "content_type": "image/jpeg",
            "size": 5,
        }

        with pytest.raises(AttachmentStagingError, match="temp path not found"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[good, bogus],
            )

        # Compensating action: the first attachment was newly staged,
        # then the second's failure must roll it back.
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        if sess_dir.exists():
            assert list(sess_dir.iterdir()) == []

    def test_malformed_dict_raises(self, tmp_path: Path, temp_workspace_root: Path) -> None:
        with pytest.raises(AttachmentStagingError, match="missing required fields"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[{"filename": "x.jpg"}],
            )

    def test_non_dict_raises(self, temp_workspace_root: Path) -> None:
        with pytest.raises(AttachmentStagingError, match="not a dict"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=["not a dict"],
            )

    def test_exdev_falls_back_to_copy_unlink(
        self,
        tmp_path: Path,
        temp_workspace_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cross-FS rename returns EXDEV; we fall back to copy+unlink.

        Simulated by patching os.rename rather than spinning up a real
        cross-FS scenario.
        """
        a = _make_temp_attachment(tmp_path, "photo.jpg", b"crossfs")
        original_rename = os.rename

        def fake_rename(src: Any, dst: Any) -> None:
            err = OSError("simulated cross-fs")
            err.errno = errno.EXDEV
            raise err

        monkeypatch.setattr(os, "rename", fake_rename)
        try:
            records, _ = stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[a],
            )
        finally:
            monkeypatch.setattr(os, "rename", original_rename)

        staged = temp_workspace_root / "_attachments" / "sess-1" / "echo" / "evt-1-photo.jpg"
        assert staged.read_bytes() == b"crossfs"
        # Source temp deleted by the unlink-after-copy step.
        assert not Path(a["host_path"]).exists()
        assert records[0]["in_sandbox_path"] == "/mnt/attachments/echo/evt-1-photo.jpg"

    def test_exdev_partial_copy_is_cleaned_up(
        self,
        tmp_path: Path,
        temp_workspace_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If shutil.copy2 fails mid-write under the EXDEV branch, the
        partial target file must still be unlinked by the compensating
        action.  Regression test for the orphan path that survived the
        original implementation (target appended to the cleanup list
        only after a successful copy).
        """
        a = _make_temp_attachment(tmp_path, "photo.jpg", b"crossfs-partial")

        def fake_rename(src: Any, dst: Any) -> None:
            err = OSError("simulated cross-fs")
            err.errno = errno.EXDEV
            raise err

        def fake_copy2(src: Any, dst: Any) -> None:
            # Simulate a partial write: drop some bytes at the target
            # and then fail.  Without the early append-to-cleanup-list,
            # this file would be orphaned.
            Path(dst).write_bytes(b"PART")
            raise OSError("simulated mid-copy failure")

        import shutil as _shutil

        monkeypatch.setattr(os, "rename", fake_rename)
        monkeypatch.setattr(_shutil, "copy2", fake_copy2)

        with pytest.raises(AttachmentStagingError, match="across filesystems"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[a],
            )

        partial = temp_workspace_root / "_attachments" / "sess-1" / "echo" / "evt-1-photo.jpg"
        assert not partial.exists(), "EXDEV partial-copy file leaked past cleanup"

    def test_same_inbound_filename_collision_fails_hard(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        """Two attachments in the same inbound that sanitize to the
        same target name must fail loudly — silently appending a
        second record pointing at the first attachment's bytes would
        corrupt ``metadata.attachments``.

        Realistic trigger: a Telegram album where two photos arrive
        as ``image.jpg`` from different devices.
        """
        a = _make_temp_attachment(tmp_path / "from-device-1", "image.jpg", b"AAA")
        b = _make_temp_attachment(tmp_path / "from-device-2", "image.jpg", b"BBB")

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[a, b],
            )

        # Compensating action: the first iteration's staged file is
        # rolled back so the orphan GC has nothing to clean up.
        sess_dir = temp_workspace_root / "_attachments" / "sess-1" / "echo"
        if sess_dir.exists():
            assert list(sess_dir.iterdir()) == []

    def test_collision_via_sanitization_fails_hard(
        self, tmp_path: Path, temp_workspace_root: Path
    ) -> None:
        """Two distinct filenames that sanitize to identical names
        (``image.jpg`` vs ``image .jpg`` → both end up
        ``evt-1-image_.jpg`` after the space → ``_`` mapping)
        also collide.
        """
        a = _make_temp_attachment(tmp_path / "a", "image_.jpg", b"AAA")
        b = _make_temp_attachment(tmp_path / "b", "image .jpg", b"BBB")

        with pytest.raises(AttachmentStagingError, match="sanitize to the same"):
            stage_inbound_attachments(
                session_id="sess-1",
                connector_name="echo",
                event_id="evt-1",
                raw_attachments=[a, b],
            )
