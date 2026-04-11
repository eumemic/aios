"""The read tool -- read a text file inside the session's sandbox.

Thin handler over
:class:`~aios.vendor.hermes_files.file_operations.ShellFileOperations`.
Obtains the per-session cached ``ShellFileOperations`` via
:mod:`aios.tools.file_session`, applies the Phase 4 safety gates
(device blocklist, binary extension check, char-count limit,
consecutive-read dedup + hard-block), and returns the structured
``ReadResult.to_dict()`` as the tool-role message content.

Error-shape convention (see PATCHES.md / phase4 plan):

- **Raise** for malformed arguments (``ReadArgumentError``) -- the
  tool dispatcher turns these into ``is_error=true`` tool messages AND
  evicts the container, which is what we want when the model produced
  garbage arguments.
- **Return a dict with ``error: ...``** for normal failure modes
  (blocked device, binary file, oversize, not-found, consecutive-loop
  block). The model sees these in the normal tool-result content and
  can adapt.
"""

from __future__ import annotations

import os
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.tools import file_session
from aios.tools.registry import registry
from aios.vendor.hermes_files.binary_extensions import has_binary_extension
from aios.vendor.hermes_files.file_operations import ShellFileOperations

# Device paths that would block or produce infinite output. Checked by
# literal path (no symlink resolution) because realpath follows
# /dev/stdin all the way to /dev/pts/0 which defeats the check.
_BLOCKED_DEVICE_PATHS = frozenset(
    {
        # Infinite output -- never reach EOF
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/full",
        # Block waiting for input
        "/dev/stdin",
        "/dev/tty",
        "/dev/console",
        # Nonsensical to read
        "/dev/stdout",
        "/dev/stderr",
        # fd aliases
        "/dev/fd/0",
        "/dev/fd/1",
        "/dev/fd/2",
    }
)


class ReadArgumentError(AiosError):
    """Raised when the read tool is called with malformed arguments."""

    error_type = "read_argument_error"
    status_code = 400


READ_DESCRIPTION = (
    "Read a text file inside the session's sandbox. Returns content "
    "with line numbers in `LINE_NUM|CONTENT` format. Use `offset` "
    "(1-indexed) and `limit` to paginate large files -- default limit "
    "is 500 lines, max 2000. Reads that produce more than ~100KB of "
    "content are rejected with a hint to narrow the range using "
    "offset/limit. Binary files (images, executables, archives, etc) "
    "are rejected by extension. Use this instead of `cat`/`head`/`tail` "
    "via the bash tool -- it handles binary detection, line numbering, "
    "and dedup of repeated reads. Paths are relative to /workspace or "
    "absolute."
)

READ_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read. Absolute, or relative to /workspace.",
        },
        "offset": {
            "type": "integer",
            "description": "1-indexed line number to start reading from. Default 1.",
            "minimum": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum lines to return. Default 500, max 2000.",
            "minimum": 1,
            "maximum": 2000,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def _is_blocked_device(path: str) -> bool:
    """Return True if the path is a device file that would hang the process."""
    normalized = os.path.expanduser(path)
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio.
    return normalized.startswith("/proc/") and normalized.endswith(("/fd/0", "/fd/1", "/fd/2"))


async def _get_mtime(file_ops: ShellFileOperations, path: str) -> float | None:
    """Return the mtime of ``path`` inside the sandbox, or None if stat fails."""
    result = await file_ops._exec(f"stat -c %Y {file_ops._escape_shell_arg(path)} 2>/dev/null")
    if result.exit_code != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


async def read_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the read tool. See module docstring for error-shape rules."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ReadArgumentError("read tool requires a non-empty 'path' string")

    raw_offset = arguments.get("offset", 1)
    if not isinstance(raw_offset, int) or raw_offset < 1:
        raise ReadArgumentError("offset must be a positive integer")

    raw_limit = arguments.get("limit", 500)
    if not isinstance(raw_limit, int) or raw_limit < 1:
        raise ReadArgumentError("limit must be a positive integer")

    # Device path guard (pure path check, no I/O).
    if _is_blocked_device(path):
        return {
            "error": (
                f"Cannot read '{path}': this is a device file that would "
                "block or produce infinite output."
            )
        }

    # Binary extension guard (pure).
    if has_binary_extension(path):
        ext = os.path.splitext(path)[1].lower()
        return {
            "error": (
                f"Cannot read binary file '{path}' ({ext}). "
                "Binary files cannot be displayed as text."
            )
        }

    sess = await file_session.get_or_create(session_id)

    async with sess.lock:
        # Dedup key -- we use the path as supplied, normalized with
        # os.path.normpath. This intentionally does NOT symlink-resolve
        # (which would require a realpath shell-out inside the
        # container). Deliberate Phase 4 simplification.
        dedup_key = (os.path.normpath(path), raw_offset, raw_limit)

        # Dedup check.
        cached_mtime = sess.dedup.get(dedup_key)
        if cached_mtime is not None:
            current_mtime = await _get_mtime(sess.file_ops, path)
            if current_mtime is not None and current_mtime == cached_mtime:
                return {
                    "content": (
                        "File unchanged since last read. The content from the "
                        "earlier read result in this conversation is still "
                        "current -- refer to that instead of re-reading."
                    ),
                    "path": path,
                    "dedup": True,
                }

        # Perform the read.
        result = await sess.file_ops.read_file(path, raw_offset, raw_limit)
        result_dict = result.to_dict()

        # Char-count guard.
        content_len = len(result.content or "")
        settings = get_settings()
        max_chars = settings.file_read_max_chars
        if content_len > max_chars:
            return {
                "error": (
                    f"Read produced {content_len:,} characters which exceeds "
                    f"the safety limit ({max_chars:,} chars). Use offset and "
                    "limit to read a smaller range."
                ),
                "path": path,
                "total_lines": result_dict.get("total_lines", "unknown"),
                "file_size": result_dict.get("file_size", 0),
            }

        # Consecutive-read tracking.
        read_key = (os.path.normpath(path), raw_offset, raw_limit)
        sess.read_history.add(read_key)
        if sess.last_key == read_key:
            sess.consecutive += 1
        else:
            sess.last_key = read_key
            sess.consecutive = 1
        count = sess.consecutive

        # Refresh dedup mtime + read_timestamps (for the write-side
        # staleness warning).
        mtime = await _get_mtime(sess.file_ops, path)
        if mtime is not None:
            sess.dedup[dedup_key] = mtime
            sess.read_timestamps[os.path.normpath(path)] = mtime

        if count >= 4:
            return {
                "error": (
                    f"BLOCKED: you have read this exact file region {count} "
                    "times in a row. The content has NOT changed. You already "
                    "have this information. STOP re-reading and proceed with "
                    "your task."
                ),
                "path": path,
                "already_read": count,
            }
        if count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times "
                "consecutively. The content has not changed since your last "
                "read. Use the information you already have."
            )

        return result_dict


def _register() -> None:
    registry.register(
        name="read",
        description=READ_DESCRIPTION,
        parameters_schema=READ_PARAMETERS_SCHEMA,
        handler=read_handler,
    )


_register()
