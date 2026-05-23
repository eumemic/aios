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

from aios.harness.image_resize import ImageDownsampleError, maybe_downsample
from aios.harness.vision import PRE_RESIZE_CEILING_BYTES
from aios.logging import get_logger
from aios.sandbox.volumes import ensure_session_attachments_dir, safe_filename
from aios.services.files import UploadStream

log = get_logger("aios.services.attachment_staging")

_CHUNK_SIZE = 1 << 20  # 1 MiB — matches stage_upload's chunk size.

# Inline-sibling extensions that the writer in :func:`_maybe_stage_inline`
# can produce.  Used for replay probing AND for pre-reserving names in
# the same-inbound collision guard so a user can't upload a file literally
# named ``photo.jpg.inline.jpg`` alongside ``photo.jpg`` and have the
# second attachment silently inherit the first's downsampled bytes.
_INLINE_EXTS = ("jpg", "png")

# Pillow ``.format`` → MIME mapping for inline-sibling replay
# reconstruction.  Intentionally narrow: the writer only produces JPEG/PNG,
# so any other format on disk is either external tampering or a future-
# format regression — fall through to ``ImageDownsampleError`` rather
# than guess a wrong mime that providers will reject on magic-vs-declared
# mismatch.
_PILLOW_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png"}


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
            # Pre-reserve the auto-generated ``.inline.<ext>`` slots for
            # any image attachment so a SUBSEQUENT attachment in the
            # same inbound can't sanitize to a name that collides with
            # our downsampled sibling — e.g. ``photo.jpg`` (oversize,
            # triggers inline) + ``photo.jpg.inline.jpg`` (small,
            # uploaded by the same user) would otherwise have the
            # second hit ``target.exists()`` (the first's inline) and
            # silently inherit its bytes via the replay branch.
            if attachment.content_type.startswith("image/"):
                for ext in _INLINE_EXTS:
                    target_names_seen.add(f"{target_name}.inline.{ext}")
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
    except BaseException:
        # BaseException so cancellation (CancelledError is BaseException
        # on py3.13+) and other rare-but-real interruptions still run
        # the compensating unlink — otherwise freshly-staged files sit
        # orphaned inside the GC's 300s recent-file protection window
        # where webhook retries could pick them up as replay artifacts.
        # Matches ``_stream_to_disk``'s posture below.
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
    downsampling to fit the inline caps.

    Returns ``None`` when no inline copy is needed or possible:
      - Non-image content type.
      - Original exceeds the pre-resize ceiling (warn-logged so
        operators can spot a systematic class of inbound silently
        degrading to text markers).
      - Original already fits both byte and dimension caps (decided
        inside :func:`maybe_downsample` via a header-only Pillow open,
        so we never pay the full decode cost for the common case).
      - Pillow can't decode or compress under the cap (warn-logged; the
        renderer falls through to the marker).

    Replay-safe: when an ``.inline.<ext>`` sibling already exists on
    disk (prior call of this same ``event_id`` succeeded but the dedup
    transaction is replaying), the record is reconstructed from disk
    without re-encoding.  A corrupt or unrecognized sibling triggers
    the same marker fallback as a fresh-encode failure — never
    propagates an uncaught exception that would 500 the inbound.
    """
    if not content_type.startswith("image/"):
        return None
    if original_size > PRE_RESIZE_CEILING_BYTES:
        log.warning(
            "staging.inline_skipped_ceiling",
            target=str(original_path),
            content_type=content_type,
            size=original_size,
            ceiling=PRE_RESIZE_CEILING_BYTES,
        )
        return None

    # Replay probe.  Iterate in lock-step with the writer's ext choices
    # below.  Any decode failure on an existing sibling (truncated file
    # from a prior crash, FS corruption, unrecognized format) collapses
    # to the marker-fallback path — propagating it up would 500 the
    # inbound and the connector would retry forever against the same
    # bad bytes.
    for ext in _INLINE_EXTS:
        candidate = original_path.with_name(f"{original_path.name}.inline.{ext}")
        if candidate.exists():
            try:
                return await asyncio.to_thread(
                    _inline_record_from_existing,
                    inline_path=candidate,
                    connector_name=connector_name,
                )
            except ImageDownsampleError as err:
                log.warning(
                    "staging.inline_replay_failed",
                    target=str(candidate),
                    error=str(err),
                )
                return None

    # Read original bytes off the event loop — ``original_size`` can be
    # up to ``PRE_RESIZE_CEILING_BYTES`` (50 MiB), and a sync read here
    # would stall every other coroutine on the worker for the duration
    # on slow filesystems (NFS, tmpfs under pressure).  Pillow's CPU
    # work is already wrapped in :func:`maybe_downsample`'s internal
    # ``to_thread``.
    original_bytes = await asyncio.to_thread(original_path.read_bytes)
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
        # Original already fits both byte AND dimension caps —
        # :func:`maybe_downsample` decided this via a header-only check
        # without paying the full decode cost.  No sibling needed.
        return None

    ext = "jpg" if resized.content_type == "image/jpeg" else "png"
    inline_path = original_path.with_name(f"{original_path.name}.inline.{ext}")
    # Atomic .part+rename, matching :func:`_stream_to_disk`'s posture:
    # a crash mid-write leaves a harmless ``.part`` orphan that GC reaps,
    # never a partial file at the canonical name that future replays
    # would mistake (via Pillow's lazy-header parse) for a complete
    # sibling and ship to the model.  Wrap the write in ``to_thread``
    # for the same event-loop reason as the read above.
    inline_temp = inline_path.with_name(inline_path.name + ".part")
    try:
        await asyncio.to_thread(inline_temp.write_bytes, resized.data)
        os.rename(inline_temp, inline_path)
    except BaseException:
        with contextlib.suppress(OSError):
            inline_temp.unlink(missing_ok=True)
        raise
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

    Pillow's ``Image.open`` parses the header eagerly (``.size``,
    ``.format``, ``.mode`` populated; no pixel decode), so this is
    cheap even on a 4 MiB image.  Raises :class:`ImageDownsampleError`
    on any failure — corrupt file, truncated header (e.g. partial
    write from a prior staging crash), or a format we don't recognize
    (external tampering or a forward-compat hazard where the writer
    learned a new ext without updating :data:`_INLINE_EXTS`).  The
    caller turns the exception into a warn log + marker fallback.

    PIL import is deferred to match the codebase's heavy-import
    pattern (see ``vision.py:67``) so API processes that never run
    staging don't pay Pillow's ~200ms cold start.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(inline_path) as img:
            width, height = img.size
            fmt = img.format
    except (UnidentifiedImageError, OSError) as err:
        raise ImageDownsampleError(f"pillow could not open {inline_path}: {err}") from err

    content_type = _PILLOW_FORMAT_TO_MIME.get(fmt or "")
    if content_type is None:
        raise ImageDownsampleError(f"unrecognized inline-sibling format {fmt!r} at {inline_path}")
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
