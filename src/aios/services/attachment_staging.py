"""Inbound attachment staging.

Connector containers POST inbound messages to ``POST /v1/connectors/inbound``
as ``multipart/form-data``; each attached file ships its bytes inline
(no shared filesystem). This module streams each upload into a stable
per-session location before the inbound event is appended::

    <workspace_root>/_attachments/<session_id>/<connector>/<event-ulid>-<filename>

That directory is bind-mounted read-only into the sandbox at
``/mnt/attachments``.

Replay-safe: a redelivered ``event_id`` finds the target path
already populated and skips the upload read. Files stranded by a
rolled-back transaction are reclaimed by
:mod:`aios.harness.attachment_gc` at worker startup.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, NamedTuple

from aios.sandbox.volumes import ensure_session_attachments_dir, safe_filename
from aios.services.files import UploadStream

_CHUNK_SIZE = 1 << 20  # 1 MiB — matches stage_upload's chunk size.


class AttachmentStagingError(Exception):
    """Per-attachment staging failure.  Caller drops the inbound with
    the canonical ``attachment_staging_failed`` reason."""


class InboundAttachment(NamedTuple):
    """One inbound attachment: an :class:`UploadStream` plus its metadata."""

    stream: UploadStream
    filename: str
    content_type: str


async def stage_inbound_attachments(
    *,
    session_id: str,
    connector_name: str,
    event_id: str,
    attachments: list[InboundAttachment],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Stream multipart bytes into per-session staging and return the
    structured records (for ``metadata.attachments``) plus the host
    paths newly created by this call (for the caller's compensating
    action when the dedup transaction fails downstream).

    Input record shape (one :class:`InboundAttachment` per uploaded file):
    a streamable body + its ``filename`` and ``content_type``.

    Output record shape (model-visible via context.py rendering)::

        {filename, content_type, size, in_sandbox_path}

    Returns ``(records, newly_staged_paths)``. ``records`` always
    matches ``attachments`` 1:1; ``newly_staged_paths`` lists only
    the paths this call materialized — replayed entries whose target
    already existed are skipped, so the caller's compensating unlink
    doesn't delete bytes referenced by a previously committed event.

    On any per-attachment failure: any files newly staged for this
    same inbound are removed and :class:`AttachmentStagingError` is
    raised so the caller drops the inbound atomically.

    Same-inbound collisions fail hard. If two attachments sanitize to
    the same ``<event_id>-<safe_name>``, we cannot satisfy both records
    with distinct bytes; silently appending a record pointing at the
    other attachment's bytes corrupts ``metadata.attachments``. The
    realistic trigger (e.g. Telegram album with two ``image.jpg``
    files) is the platform's responsibility to disambiguate via the
    SDK; on collision this function raises
    :class:`AttachmentStagingError`, the inbound is dropped, and the
    operator sees the canonical ``attachment_staging_failed`` reason.
    """
    if not attachments:
        return [], []

    connector_dir = ensure_session_attachments_dir(session_id) / connector_name
    connector_dir.mkdir(parents=True, exist_ok=True)

    staged_records: list[dict[str, Any]] = []
    newly_staged_paths: list[Path] = []
    target_names_seen: set[str] = set()

    try:
        for attachment in attachments:
            target_name = f"{event_id}-{safe_filename(attachment.filename)}"
            if target_name in target_names_seen:
                raise AttachmentStagingError(
                    f"two attachments in inbound {event_id!r} sanitize to the "
                    f"same staged name {target_name!r} ({attachment.filename!r}); the "
                    f"connector must disambiguate before delivery"
                )
            target_names_seen.add(target_name)
            target = connector_dir / target_name

            if target.exists():
                # Replay: target already populated, but we still need
                # its size for the record. ``stat`` is cheap and avoids
                # re-reading the stream.
                size = target.stat().st_size
            else:
                # Register the destination path with the cleanup list
                # *before* we touch the disk so a partial write
                # (e.g. mid-stream cancellation) is still reachable for
                # the compensating unlink.
                newly_staged_paths.append(target)
                size = await _stream_to_disk(attachment.stream, target)

            staged_records.append(
                {
                    "filename": attachment.filename,
                    "content_type": attachment.content_type,
                    "size": size,
                    "in_sandbox_path": f"/mnt/attachments/{connector_name}/{target_name}",
                }
            )

        return staged_records, newly_staged_paths
    except Exception:
        for path in newly_staged_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise


async def _stream_to_disk(stream: UploadStream, target: Path) -> int:
    """Stream ``stream`` to ``target`` and return total bytes written.

    Writes go to a ``.part`` sibling then rename atomically into place,
    matching :func:`aios.services.files.stage_upload`'s posture — half-
    written ``.part`` is a harmless orphan; the final path either exists
    fully or not at all.
    """
    temp_path = target.with_name(target.name + ".part")
    size = 0
    try:
        # ASYNC230: local-disk write, executor wrap isn't worth the per-chunk cost.
        with open(temp_path, "wb") as f:  # noqa: ASYNC230
            while True:
                chunk = await stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)
        os.rename(temp_path, target)
    except BaseException:
        # BaseException so partial state is cleaned up under task cancellation.
        # ASYNC240: tmpfs unlinks are sub-microsecond; threading them
        # would cost more than the syscall.
        temp_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)  # noqa: ASYNC240
        raise
    return size
