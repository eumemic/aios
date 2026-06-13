"""Unit coverage for the multipart upload service.

Filename sanitisation lives in ``volumes.safe_filename`` and is covered by
``test_attachment_staging.TestSafeFilename`` — not duplicated here.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from aios.config import get_settings
from aios.errors import NotFoundError, PayloadTooLargeError
from aios.models.files import File
from aios.services.files import stage_upload
from tests.unit.conftest import fake_pool_yielding_conn


class _FakeUpload:
    """Minimal ``UploadStream`` Protocol implementation backed by BytesIO."""

    def __init__(
        self, data: bytes, *, filename: str | None = "test.bin", content_type: str | None = None
    ) -> None:
        self._buf = BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._buf.read(size if size > 0 else -1)


def _patch_session_get(session_id: str = "sess_x") -> Any:
    """Patch ``queries.get_session_bare`` to return a stub session row."""
    stub = MagicMock()
    stub.id = session_id
    return patch(
        "aios.services.files.queries.get_session_bare",
        AsyncMock(return_value=stub),
    )


def _patch_session_not_found() -> Any:
    return patch(
        "aios.services.files.queries.get_session_bare",
        AsyncMock(side_effect=NotFoundError("session sess_x not found", detail={"id": "sess_x"})),
    )


def _patch_insert_file(captured: dict[str, Any]) -> Any:
    """Patch ``queries.insert_file`` so the test can inspect the args."""

    async def _record(_conn: object, **kwargs: Any) -> File:
        captured.update(kwargs)
        return File(
            id=kwargs["file_id"],
            session_id=kwargs["session_id"],
            filename=kwargs["filename"],
            host_path=kwargs["host_path"],
            in_sandbox_path=kwargs["in_sandbox_path"],
            size=kwargs["size"],
            content_type=kwargs["content_type"],
            sha256=kwargs["sha256"],
            created_at=__import__("datetime").datetime.now(),
        )

    return patch("aios.services.files.queries.insert_file", new=_record)


@pytest.fixture
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


class TestStageUploadHappyPath:
    async def test_writes_bytes_and_computes_sha256(self, _workspace: Path) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        data = b"hello world\nthis is a test upload"
        upload = _FakeUpload(data, filename="hello.txt", content_type="text/plain")
        captured: dict[str, Any] = {}
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))

        with _patch_session_get(), _patch_insert_file(captured):
            result = await stage_upload(
                pool, session_id="sess_x", upload=upload, account_id=account_id
            )

        assert result.size == len(data)
        assert result.sha256 == hashlib.sha256(data).hexdigest()
        assert result.filename == "hello.txt"
        assert result.content_type == "text/plain"
        assert result.in_sandbox_path == f"/mnt/uploads/{result.id}/hello.txt"
        # Bytes durable on disk at the host path.
        assert Path(result.host_path).read_bytes() == data  # noqa: ASYNC240
        # File-then-DB: insert called after the bytes are present.
        assert captured["file_id"] == result.id
        assert captured["host_path"] == result.host_path

    async def test_empty_file_accepted(self, _workspace: Path) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        upload = _FakeUpload(b"", filename="empty.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_get(), _patch_insert_file({}):
            result = await stage_upload(
                pool, session_id="sess_x", upload=upload, account_id=account_id
            )
        assert result.size == 0
        assert result.sha256 == hashlib.sha256(b"").hexdigest()
        assert Path(result.host_path).read_bytes() == b""  # noqa: ASYNC240

    async def test_content_type_defaults_to_octet_stream(self, _workspace: Path) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        upload = _FakeUpload(b"x", filename="anon", content_type=None)
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_get(), _patch_insert_file({}):
            result = await stage_upload(
                pool, session_id="sess_x", upload=upload, account_id=account_id
            )
        assert result.content_type == "application/octet-stream"

    async def test_unicode_filename_preserved_on_disk(self, _workspace: Path) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        upload = _FakeUpload(b"hi", filename="图片.png", content_type="image/png")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_get(), _patch_insert_file({}):
            result = await stage_upload(
                pool, session_id="sess_x", upload=upload, account_id=account_id
            )
        assert result.filename == "图片.png"
        assert Path(result.host_path).name == "图片.png"


class TestStageUploadOversize:
    async def test_oversize_raises_413(
        self, _workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_max_size_bytes", 64)
        upload = _FakeUpload(b"a" * 256, filename="big.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with (
            _patch_session_get(),
            _patch_insert_file({}),
            pytest.raises(PayloadTooLargeError) as excinfo,
        ):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)
        assert excinfo.value.status_code == 413
        assert excinfo.value.detail["max_size_bytes"] == 64

    async def test_oversize_drains_remainder(
        self, _workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drain is what lets the client see a clean 413 instead of a
        transport reset. Verifying the upstream is empty after the call
        is the easiest end-state check."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_max_size_bytes", 64)
        upload = _FakeUpload(b"a" * 4096, filename="big.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with (
            _patch_session_get(),
            _patch_insert_file({}),
            pytest.raises(PayloadTooLargeError),
        ):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)
        assert await upload.read() == b""

    async def test_oversize_leaves_no_artifacts(
        self, _workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed upload must not leave a half-written .part or an
        empty file_id directory under _uploads/<session>/. Otherwise a
        sweeper has to reason about state instead of just deleting."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_max_size_bytes", 64)
        upload = _FakeUpload(b"a" * 256, filename="big.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with (
            _patch_session_get(),
            _patch_insert_file({}),
            pytest.raises(PayloadTooLargeError),
        ):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)
        uploads_dir = _workspace / "_uploads" / "sess_x"
        # The session-level dir exists (ensure_session_uploads_dir) but it
        # holds no per-file subdirectories.
        assert uploads_dir.exists()
        assert list(uploads_dir.iterdir()) == []


class TestStageUploadInsertFailure:
    async def test_insert_failure_leaves_no_orphan_bytes(self, _workspace: Path) -> None:
        """If the row insert fails after the bytes are renamed into place,
        the staged file must not leak on disk.

        ``stage_upload`` deliberately writes-then-inserts so a row never
        points at missing bytes. But the renamed ``final_path`` lives
        *outside* the ``except BaseException`` cleanup that guards the
        write, so an ``insert_file`` failure (an FK violation when the
        session is hard-deleted in the TOCTOU window after
        ``get_session_bare``, or a pool/timeout error) used to strand the
        fully-written file plus its ``<file_id>/`` dir forever — there is
        no upload reconciler to sweep them. A failed upload must leave the
        per-session dir as empty as a never-attempted one."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        upload = _FakeUpload(b"durable bytes", filename="ok.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        insert_boom = patch(
            "aios.services.files.queries.insert_file",
            AsyncMock(side_effect=RuntimeError("insert failed")),
        )
        with _patch_session_get(), insert_boom, pytest.raises(RuntimeError):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)
        uploads_dir = _workspace / "_uploads" / "sess_x"
        assert uploads_dir.exists()
        assert list(uploads_dir.iterdir()) == []


class TestStageUploadSessionNotFound:
    async def test_nonexistent_session_propagates_404(self, _workspace: Path) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        upload = _FakeUpload(b"x", filename="ok.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_not_found(), pytest.raises(NotFoundError):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)
        # Short-circuited before any disk activity.
        uploads_dir = _workspace / "_uploads" / "sess_x"
        assert not uploads_dir.exists()


class TestStageUploadOwnership:
    """#959: under root the worker chowns newly-created _uploads tree
    components to the configured owner so the api (uid 1000) can write."""

    async def test_upload_chowns_uploads_root_and_session_dir_as_root(
        self, _workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspaces_owner_uid", 1000)
        monkeypatch.setattr(settings, "workspaces_owner_gid", 1000)
        chowns: list[tuple[str, int, int]] = []
        # ensure_owned_dir uses os.lchown (symlink-swap-race-safe; #959 FIX C).
        monkeypatch.setattr(os, "lchown", lambda p, u, g: chowns.append((str(p), u, g)))
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        account_id = "acc_test_stub"
        upload = _FakeUpload(b"data", filename="a.bin")
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_get(), _patch_insert_file({}):
            await stage_upload(pool, session_id="sess_x", upload=upload, account_id=account_id)

        chowned = {p for p, _u, _g in chowns}
        assert str(_workspace / "_uploads") in chowned
        assert str(_workspace / "_uploads" / "sess_x") in chowned
        assert all((u, g) == (1000, 1000) for _p, u, g in chowns)

    async def test_upload_second_call_no_op_chown(
        self, _workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "workspaces_owner_uid", 1000)
        monkeypatch.setattr(settings, "workspaces_owner_gid", 1000)
        chowns: list[tuple[str, int, int]] = []
        # ensure_owned_dir uses os.lchown (symlink-swap-race-safe; #959 FIX C).
        monkeypatch.setattr(os, "lchown", lambda p, u, g: chowns.append((str(p), u, g)))
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        account_id = "acc_test_stub"
        pool = cast("asyncpg.Pool[Any]", fake_pool_yielding_conn(MagicMock()))
        with _patch_session_get(), _patch_insert_file({}):
            await stage_upload(
                pool,
                session_id="sess_x",
                upload=_FakeUpload(b"one", filename="one.bin"),
                account_id=account_id,
            )
            chowns.clear()
            # Second file, same session: _uploads and _uploads/sess_x already
            # exist, so neither gets re-chowned. (The per-file_id leaf is an
            # api-side bare mkdir that does not chown.)
            await stage_upload(
                pool,
                session_id="sess_x",
                upload=_FakeUpload(b"two", filename="two.bin"),
                account_id=account_id,
            )

        chowned = {p for p, _u, _g in chowns}
        assert str(_workspace / "_uploads") not in chowned
        assert str(_workspace / "_uploads" / "sess_x") not in chowned
