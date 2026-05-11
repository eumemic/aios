"""Multipart upload staging for ``POST /v1/sessions/<id>/files`` (#324).

Two non-obvious choices worth recording here:

* **Drain-before-413.** On overflow the rest of the multipart body is
  read and discarded before raising ``PayloadTooLargeError``. Without
  the drain the parser bails out and the client sees a transport reset,
  which is much harder to diagnose than a clean 413.
* **File-then-DB ordering.** Bytes hit disk and rename atomically into
  place before the row insert. A half-written ``.part`` is a harmless
  orphan; a DB row pointing at missing bytes is observable corruption.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from typing import Any, Protocol

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.errors import PayloadTooLargeError
from aios.ids import FILE, make_id
from aios.models.files import File
from aios.sandbox.volumes import ensure_session_uploads_dir, safe_filename

_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


class UploadStream(Protocol):
    """Subset of ``fastapi.UploadFile`` we depend on.

    Declared as a :class:`Protocol` so unit tests can pass a lightweight
    in-memory shim without dragging starlette into the test path.
    """

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...


async def stage_upload(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    upload: UploadStream,
) -> File:
    """Stream the upload to disk and persist a row in ``files``.

    Raises :class:`NotFoundError` if ``session_id`` doesn't exist and
    :class:`PayloadTooLargeError` (413) if the body exceeds the
    configured size limit.  Returns the inserted :class:`File`.
    """
    settings = get_settings()

    async with pool.acquire() as conn:
        await queries.get_session(conn, session_id)  # 404 if missing

    file_id = make_id(FILE)
    filename = safe_filename(upload.filename)
    content_type = upload.content_type or _DEFAULT_CONTENT_TYPE

    file_dir = ensure_session_uploads_dir(session_id) / file_id
    file_dir.mkdir(parents=True, exist_ok=False)
    final_path = file_dir / filename
    temp_path = file_dir / f"{filename}.part"
    in_sandbox_path = f"/mnt/uploads/{file_id}/{filename}"

    hasher = hashlib.sha256()
    size = 0
    overflow = False
    try:
        # ASYNC230: local-disk write, executor wrap isn't worth the per-chunk cost.
        with open(temp_path, "wb") as f:  # noqa: ASYNC230
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.upload_max_size_bytes:
                    while await upload.read(_CHUNK_SIZE):
                        pass
                    overflow = True
                    break
                hasher.update(chunk)
                f.write(chunk)
            if not overflow:
                f.flush()
                os.fsync(f.fileno())
        if overflow:
            raise PayloadTooLargeError(
                f"upload exceeds {settings.upload_max_size_bytes:,} bytes",
                detail={"max_size_bytes": settings.upload_max_size_bytes},
            )
        os.rename(temp_path, final_path)
    except BaseException:
        # BaseException (not Exception) so partial state still gets cleaned up
        # under task cancellation — CancelledError doesn't inherit from
        # Exception in 3.11+.
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            file_dir.rmdir()
        raise

    async with pool.acquire() as conn:
        return await queries.insert_file(
            conn,
            file_id=file_id,
            session_id=session_id,
            filename=filename,
            host_path=str(final_path),
            in_sandbox_path=in_sandbox_path,
            size=size,
            content_type=content_type,
            sha256=hasher.hexdigest(),
        )
