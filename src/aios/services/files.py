"""Multipart upload staging for ``POST /v1/sessions/<id>/files`` (#324).

Streams the request body to disk in chunks, computing sha256 as it goes
and enforcing :attr:`Settings.upload_max_size_bytes`.  On overflow the
remainder of the body is drained so the client sees a clean 413 rather
than a transport reset (a parser-level reset surfaces as a generic 500
on the client and is hostile to debug).

Ordering is file-then-DB on purpose: a half-written ``.part`` file is a
harmless orphan that a future sweeper can reap, whereas a DB row that
points at missing bytes is observable corruption.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Protocol

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.errors import PayloadTooLargeError
from aios.ids import FILE, make_id
from aios.models.files import File
from aios.sandbox.volumes import ensure_session_uploads_dir

_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_DEFAULT_FILENAME = "upload.bin"


class UploadStream(Protocol):
    """Subset of ``fastapi.UploadFile`` we depend on.

    Declared as a :class:`Protocol` so unit tests can pass a lightweight
    in-memory shim without dragging starlette into the test path.
    """

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...


def sanitize_filename(name: str | None) -> str:
    """Strip path components and replace unsafe characters.

    Python's ``\\w`` matches unicode word characters by default, so
    Chinese / Cyrillic / accented filenames survive intact; only true
    path-unsafe characters (separators, null bytes, control chars,
    spaces, shell metacharacters) get replaced.  Empty result falls
    back to :data:`_DEFAULT_FILENAME`.
    """
    if not name:
        return _DEFAULT_FILENAME
    base = Path(name).name
    # ``Path('..').name == '..'`` (not empty) — pathlib doesn't treat
    # parent-refs specially in the .name accessor. Reject them
    # explicitly so we don't end up writing to ``<file_dir>/..``.
    if base in (".", ".."):
        return _DEFAULT_FILENAME
    cleaned = re.sub(r"[^\w.-]", "_", base)
    return cleaned or _DEFAULT_FILENAME


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
    filename = sanitize_filename(upload.filename)
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
        # ASYNC230: open() on a local-disk path is fast (one syscall) and
        # the file stays open across awaits only as long as the upstream
        # multipart body keeps feeding chunks — exactly the lifetime we
        # want. Wrapping every chunk write in asyncio.to_thread() would
        # add executor overhead per MiB without measurable benefit.
        with open(temp_path, "wb") as f:  # noqa: ASYNC230
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.upload_max_size_bytes:
                    # Drain the rest so the client sees a clean 413 instead of
                    # a transport reset from the multipart parser bailing out.
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
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if final_path.exists():
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
