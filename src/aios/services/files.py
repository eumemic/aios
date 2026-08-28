"""Multipart upload staging for ``POST /v1/sessions/<id>/files`` (#324).

Two non-obvious choices worth recording here:

* **Drain-before-413.** On overflow the rest of the multipart body is
  read and discarded before raising ``PayloadTooLargeError``. Without
  the drain the parser bails out and the client sees a transport reset,
  which is much harder to diagnose than a clean 413.
* **File-then-DB under one cleanup scope.** Bytes hit disk and rename
  atomically into place, then the row inserts — all inside the ``try``
  whose ``except`` unlinks partial state. A failure anywhere up to and
  including the insert leaves no orphan (there is no upload reconciler,
  so a stranded file would be permanent); only the success path, which
  returns before the ``except`` fires, leaves durable bytes. The insert
  being last is what keeps a DB row from ever pointing at missing bytes.
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
DEFAULT_CONTENT_TYPE = "application/octet-stream"

#: Types :func:`aios.api.routers.sessions.download_file` will serve inline
#: with their stored content-type.  Deliberately a positive allowlist of
#: raster image types — the set #179 (composer image thumbnails) actually
#: needs — rather than a blocklist or an ``image/*`` prefix test.
#:
#: ``stage_upload`` records ``upload.content_type`` verbatim from the
#: client, so the stored type is attacker-chosen for any uploader.  A
#: prefix test on ``image/`` admits ``image/svg+xml``, which browsers
#: execute as script in the serving origin — stored XSS against anyone
#: who opens the file.  ``text/html`` and ``application/xhtml+xml`` are
#: the same class.  A blocklist would have to enumerate every such type
#: correctly forever; this allowlist fails closed on anything new.
INLINE_RENDERABLE_CONTENT_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)


def normalized_content_type(content_type: str | None) -> str:
    """Bare lowercase media type, parameters and whitespace stripped.

    ``image/png; charset=utf-8`` and ``IMAGE/PNG `` both normalize to
    ``image/png`` so a parameter or casing trick can't slip a renderable
    type past :data:`INLINE_RENDERABLE_CONTENT_TYPES` — or, worse, sneak
    one *onto* it.
    """
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


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
    account_id: str,
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
        await queries.get_session_bare(conn, session_id, account_id=account_id)  # 404 if missing

    file_id = make_id(FILE)
    filename = safe_filename(upload.filename)
    content_type = upload.content_type or DEFAULT_CONTENT_TYPE

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
        # Insert inside the cleanup scope (see the module docstring): success
        # returns before the except can fire, and any failure here — an FK
        # violation if the session was hard-deleted since get_session_bare, or
        # a pool/timeout error — unlinks the renamed file and re-raises.
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
                account_id=account_id,
            )
    except BaseException:
        # BaseException (not Exception) so partial state still gets cleaned up
        # under task cancellation — CancelledError doesn't inherit from
        # Exception in 3.11+.
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            file_dir.rmdir()
        raise
