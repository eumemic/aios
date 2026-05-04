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
import re
import shutil
from pathlib import Path
from typing import Any

from aios.sandbox.volumes import ensure_session_attachments_dir


class AttachmentStagingError(Exception):
    """Per-attachment staging failure.  Caller drops the inbound with
    the canonical ``attachment_staging_failed`` reason."""


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\-]")
_MAX_FILENAME_LEN = 200


def _safe_filename(name: str) -> str:
    """Sanitize ``name`` for use as a path leaf.

    Strips directory separators (defeats ``../`` traversal), maps
    unsupported characters to ``_``, falls back to ``"unnamed"`` for
    empty or all-dot inputs, and caps length so a pathological
    filename combined with the ULID prefix can't exhaust the host
    FS's per-component limit.
    """
    base = os.path.basename(name)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", base)
    if not cleaned or cleaned.replace(".", "") == "":
        return "unnamed"
    return cleaned[:_MAX_FILENAME_LEN]


def stage_inbound_attachments(
    *,
    session_id: str,
    connector_name: str,
    event_id: str,
    raw_attachments: Any,
) -> list[dict[str, Any]]:
    """Move connector-owned temp paths into per-session staging and
    return the structured records that get stamped onto the inbound
    event's ``metadata.attachments``.

    Input record shape (from the SDK over JSON-RPC)::

        {host_path, filename, content_type, size}

    Output record shape (model-visible via context.py rendering)::

        {filename, content_type, size, in_sandbox_path}

    On any per-attachment failure: any files newly staged for this
    same inbound are removed and :class:`AttachmentStagingError` is
    raised so the caller drops the inbound atomically.

    Replay-safe: if the target path already exists (from a prior
    delivery of the same ``event_id``), the rename is skipped and the
    record is rebuilt from the input fields.  Orphan GC handles
    files stranded by events that never made it past dedup.
    """
    if not isinstance(raw_attachments, list) or not raw_attachments:
        return []

    connector_dir = ensure_session_attachments_dir(session_id) / connector_name
    connector_dir.mkdir(parents=True, exist_ok=True)

    staged_records: list[dict[str, Any]] = []
    newly_staged_paths: list[Path] = []

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

            target_name = f"{event_id}-{_safe_filename(filename)}"
            target = connector_dir / target_name

            if not target.exists():
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
                newly_staged_paths.append(target)

            staged_records.append(
                {
                    "filename": filename,
                    "content_type": content_type,
                    "size": size,
                    "in_sandbox_path": f"/mnt/attachments/{connector_name}/{target_name}",
                }
            )

        return staged_records
    except Exception:
        for path in newly_staged_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
