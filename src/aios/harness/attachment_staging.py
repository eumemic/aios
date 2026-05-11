"""Inbound attachment staging.

The connector hands the SDK a host path it owns; bytes never
traverse stdio. The supervisor renames each attachment into a
stable per-session location before appending the inbound event::

    <workspace_root>/_attachments/<session_id>/<connector>/<event-ulid>-<filename>

That directory is bind-mounted read-only into the sandbox at
``/mnt/attachments``. Renames are atomic on same-FS; cross-FS falls
back to copy + unlink.

Replay-safe: a redelivered ``event_id`` finds the target path
already populated and skips the rename. Files stranded by a
rolled-back transaction are reclaimed by
:mod:`aios.harness.attachment_gc` at worker startup.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
from pathlib import Path
from typing import Any

from aios.sandbox.volumes import ensure_session_attachments_dir, safe_filename


class AttachmentStagingError(Exception):
    """Per-attachment staging failure.  Caller drops the inbound with
    the canonical ``attachment_staging_failed`` reason."""


def stage_inbound_attachments(
    *,
    session_id: str,
    connector_name: str,
    event_id: str,
    raw_attachments: Any,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Move connector-owned temp paths into per-session staging and
    return the structured records (for ``metadata.attachments``) plus
    the host paths newly created by this call (for the caller's
    compensating action when the dedup transaction fails downstream).

    Input record shape (from the SDK over JSON-RPC)::

        {host_path, filename, content_type, size}

    Output record shape (model-visible via context.py rendering)::

        {filename, content_type, size, in_sandbox_path}

    Returns ``(records, newly_staged_paths)``. ``records`` always
    matches ``raw_attachments`` 1:1; ``newly_staged_paths`` lists only
    the paths this call materialized — replayed entries whose target
    already existed are not included, so the caller's compensating
    unlink doesn't delete bytes referenced by a previously committed
    event.

    On any per-attachment failure: any files newly staged for this
    same inbound are removed and :class:`AttachmentStagingError` is
    raised so the caller drops the inbound atomically.

    Replay-safe: if the target path already exists (from a prior
    delivery of the same ``event_id``), the rename is skipped, the
    record is rebuilt from the input fields, and the path is omitted
    from ``newly_staged_paths``. Orphan GC handles files stranded by
    events that never made it past dedup.

    Same-inbound collisions fail hard. If two attachments sanitize to
    the same ``<event_id>-<safe_name>``, we cannot satisfy both records
    with distinct bytes; silently appending a record pointing at the
    other attachment's bytes corrupts ``metadata.attachments``. The
    realistic trigger (e.g. Telegram album with two ``image.jpg``
    files) is the platform's responsibility to disambiguate via the
    SDK; the supervisor drops the inbound and the operator sees the
    canonical ``attachment_staging_failed`` reason.
    """
    if not isinstance(raw_attachments, list) or not raw_attachments:
        return [], []

    connector_dir = ensure_session_attachments_dir(session_id) / connector_name
    connector_dir.mkdir(parents=True, exist_ok=True)

    staged_records: list[dict[str, Any]] = []
    newly_staged_paths: list[Path] = []
    target_names_seen: set[str] = set()

    try:
        for raw in raw_attachments:
            if not isinstance(raw, dict):
                raise AttachmentStagingError(f"attachment is not a dict: {raw!r}")
            host_path = raw.get("host_path")
            filename = raw.get("filename")
            content_type = raw.get("content_type")
            size = raw.get("size")
            if not (
                isinstance(host_path, str)
                and isinstance(filename, str)
                and isinstance(content_type, str)
                and isinstance(size, int)
            ):
                raise AttachmentStagingError(
                    f"attachment missing required fields "
                    f"{{host_path, filename, content_type, size}}: {raw!r}"
                )

            target_name = f"{event_id}-{safe_filename(filename)}"
            if target_name in target_names_seen:
                raise AttachmentStagingError(
                    f"two attachments in inbound {event_id!r} sanitize to the "
                    f"same staged name {target_name!r} ({filename!r}); the "
                    f"connector must disambiguate before delivery"
                )
            target_names_seen.add(target_name)
            target = connector_dir / target_name

            if not target.exists():
                # Register the destination path with the cleanup list
                # *before* we touch the disk so a partial copy (EXDEV
                # branch can fail mid-write) is still reachable for
                # the compensating unlink. ``unlink(missing_ok=True)``
                # makes early registration safe on the rename branch
                # too, where the target either exists fully or not at
                # all.
                newly_staged_paths.append(target)
                try:
                    os.rename(host_path, target)
                except FileNotFoundError as err:
                    raise AttachmentStagingError(
                        f"attachment temp path not found: {host_path!r}"
                    ) from err
                except OSError as err:
                    if err.errno != errno.EXDEV:
                        raise AttachmentStagingError(
                            f"failed to stage attachment {filename!r}: {err}"
                        ) from err
                    try:
                        shutil.copy2(host_path, target)
                    except OSError as copy_err:
                        raise AttachmentStagingError(
                            f"failed to copy attachment {filename!r} across filesystems: {copy_err}"
                        ) from copy_err
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(host_path)

            staged_records.append(
                {
                    "filename": filename,
                    "content_type": content_type,
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
