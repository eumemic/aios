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

Oversize images (above :data:`vision.INLINE_SIZE_CAP_BYTES`) get a
sibling ``<target>.inline.<ext>`` produced by :func:`maybe_downsample`;
the record carries an ``inline`` sub-record pointing at it so the
renderer can show the model pixels instead of a path marker.  The
original file is untouched — sandbox tools still see the unresized
bytes at ``/mnt/attachments/...``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image

from aios.harness.image_resize import ImageDownsampleError, maybe_downsample
from aios.harness.vision import INLINE_SIZE_CAP_BYTES, PRE_RESIZE_CEILING_BYTES
from aios.logging import get_logger
from aios.sandbox.volumes import ensure_session_attachments_dir, safe_filename
from aios.services.files import UploadStream

log = get_logger("aios.services.attachment_staging")

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
                # Register the destination AFTER ``_stream_to_disk``
                # successfully renames our ``.part`` into place. If we
                # registered before and ``_stream_to_disk`` raised mid-
                # stream, the caller's cleanup would unlink ``target`` —
                # but in the concurrent-same-event_id race (webhook
                # retries) ``target`` may have just been atomically
                # renamed into place by the winning invocation, so
                # unlinking would DELETE the winner's bytes the renderer
                # later expects. Post-rename is the only point where
                # this invocation truly owns ``target``.
                size = await _stream_to_disk(attachment.stream, target)
                newly_staged_paths.append(target)

            inline_record = await _maybe_stage_inline(
                original_path=target,
                original_size=size,
                content_type=attachment.content_type,
                connector_name=connector_name,
                newly_staged_paths=newly_staged_paths,
            )

            record: dict[str, Any] = {
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "size": size,
                "in_sandbox_path": f"/mnt/attachments/{connector_name}/{target_name}",
            }
            if inline_record is not None:
                record["inline"] = inline_record
            staged_records.append(record)

        return staged_records, newly_staged_paths
    except Exception:
        for path in newly_staged_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise


async def _maybe_stage_inline(
    *,
    original_path: Path,
    original_size: int,
    content_type: str,
    connector_name: str,
    newly_staged_paths: list[Path],
) -> dict[str, Any] | None:
    """Return an ``inline`` sub-record for the renderer (and possibly
    write a sibling ``.inline.<ext>`` file) when the original needs
    downsampling to fit the inline cap.

    Returns ``None`` when no inline copy is needed or possible:
      - Non-image content type.
      - Original already fits the inline cap.
      - Original exceeds the pre-resize ceiling (would be slow to decode).
      - Pillow can't decode or compress under the cap (warn-logged; the
        renderer falls through to the marker).

    Replay-safe: when an ``.inline.<ext>`` sibling already exists on
    disk (prior call of this same ``event_id`` succeeded but the dedup
    transaction is replaying), the record is reconstructed from disk
    without re-encoding.
    """
    if not content_type.startswith("image/"):
        return None
    if original_size <= INLINE_SIZE_CAP_BYTES:
        return None
    if original_size > PRE_RESIZE_CEILING_BYTES:
        return None

    # Inline siblings live next to the original.  We only ever write
    # ``.inline.jpg`` (non-transparent path) or ``.inline.png``
    # (transparency-preserving path); probe both on replay so we don't
    # re-encode an image whose bytes already exist on disk.
    for ext in ("jpg", "png"):
        candidate = original_path.with_name(f"{original_path.name}.inline.{ext}")
        if candidate.exists():
            return await asyncio.to_thread(
                _inline_record_from_existing,
                inline_path=candidate,
                connector_name=connector_name,
            )

    # ASYNC240: small disk read into an existing per-attachment async
    # boundary — wrapping in to_thread would add scheduling overhead
    # for no real concurrency win, matching the policy in
    # :func:`_stream_to_disk` above.  Pillow itself, called below via
    # ``maybe_downsample``, is the heavy work and is already in a
    # thread.
    original_bytes = original_path.read_bytes()  # noqa: ASYNC240
    try:
        resized = await maybe_downsample(original_bytes, content_type)
    except ImageDownsampleError as err:
        log.warning(
            "staging.downsample_failed",
            target=str(original_path),
            content_type=content_type,
            size=original_size,
            error=str(err),
        )
        return None
    if resized is None:
        # Unreachable in practice (original_size > cap above), but the
        # type narrowing wants the explicit branch.
        return None

    ext = "jpg" if resized.content_type == "image/jpeg" else "png"
    inline_path = original_path.with_name(f"{original_path.name}.inline.{ext}")
    inline_path.write_bytes(resized.data)
    newly_staged_paths.append(inline_path)
    return {
        "in_sandbox_path": f"/mnt/attachments/{connector_name}/{inline_path.name}",
        "content_type": resized.content_type,
        "size": len(resized.data),
        "width": resized.width,
        "height": resized.height,
    }


def _inline_record_from_existing(*, inline_path: Path, connector_name: str) -> dict[str, Any]:
    """Reconstruct an inline sub-record from an on-disk sibling.

    Pillow's ``Image.open`` is lazy — reading ``.size`` and ``.format``
    decodes only the header, not the pixel data.  Cheap even on a 4 MB
    image.
    """
    with Image.open(inline_path) as img:
        width, height = img.size
        fmt = img.format
    content_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return {
        "in_sandbox_path": f"/mnt/attachments/{connector_name}/{inline_path.name}",
        "content_type": content_type,
        "size": inline_path.stat().st_size,
        "width": width,
        "height": height,
    }


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
        # Only unlink ``temp_path`` — never ``target``. The ``target`` slot may
        # have been atomically renamed into by a concurrent invocation
        # (webhook retry with same event_id) that won the race past our
        # ``target.exists()`` check; unlinking it would delete the winner's
        # bytes the renderer later expects via the ``staged_records``
        # ``in_sandbox_path``. Cleanup of ``target`` for the failure path
        # in which THIS invocation owned it (rename succeeded, caller
        # raised later) is the caller's job via ``newly_staged_paths``.
        # tmpfs unlink is sub-microsecond; not worth threading.
        temp_path.unlink(missing_ok=True)
        raise
    return size
