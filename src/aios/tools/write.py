"""The write tool -- write content to a file inside the session's sandbox.

Thin handler over
:class:`~aios.vendor.hermes_files.file_operations.ShellFileOperations`'s
``write_file`` method. Applies the Phase 4 safety gates: sensitive-path
check (hermes's ``/etc/``, ``/boot/``, ``/usr/lib/systemd/`` prefixes
+ ``/var/run/docker.sock`` exact match), char-count limit, and
staleness warning if the file was modified after the most recent read.

Error-shape convention:

- **Raise** for malformed arguments (``WriteArgumentError``).
- **Return a dict with ``error: ...``** for deny-listed paths,
  sensitive paths, oversize, and write failures. The model sees these
  in the normal tool-result content.
"""

from __future__ import annotations

import os
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.tools import file_session
from aios.tools.registry import registry
from aios.vendor.hermes_files.file_operations import ShellFileOperations

# Sensitive path prefixes and exact paths -- refuse to write even if the
# primary deny list misses them. Above-and-beyond the
# ``_is_write_denied`` check inside ShellFileOperations.
_SENSITIVE_PATH_PREFIXES = ("/etc/", "/boot/", "/usr/lib/systemd/")
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}


class WriteArgumentError(AiosError):
    """Raised when the write tool is called with malformed arguments."""

    error_type = "write_argument_error"
    status_code = 400


WRITE_DESCRIPTION = (
    "Write content to a file inside the session's sandbox, completely "
    "replacing any existing content. Parent directories are created "
    "automatically. Content is piped through stdin (no ARG_MAX limit "
    "on file size). Writes to sensitive system paths (/etc, /boot, "
    "~/.ssh, etc) are denied. Use this instead of heredocs or `echo >` "
    "via the bash tool -- it's simpler and supports arbitrary binary-"
    "safe content. For targeted changes to existing files, use the "
    "edit tool instead to avoid overwriting the whole file."
)

WRITE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "File path to write. Absolute, or relative to /workspace. "
                "Parent directories are created automatically."
            ),
        },
        "content": {
            "type": "string",
            "description": "Complete file content. Overwrites any existing file.",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def _check_sensitive_path(path: str) -> str | None:
    """Return an error message if the path targets a sensitive system location.

    Checks BOTH the raw (expanded-but-not-resolved) path and the realpath
    because macOS's ``/etc`` is a symlink to ``/private/etc`` — realpath
    alone would convert ``/etc/passwd`` to ``/private/etc/passwd`` and
    miss the ``/etc/`` prefix. The goal here is to catch model requests
    for sensitive paths at a semantic level, not symlink-escape
    attempts, so checking both is the correct behaviour.
    """
    expanded = os.path.expanduser(path)
    try:
        resolved = os.path.realpath(expanded)
    except (OSError, ValueError):
        resolved = expanded

    msg = (
        f"Refusing to write to sensitive system path: {path}. "
        "Use the bash tool if you need to modify system files."
    )

    for candidate in (expanded, resolved):
        for prefix in _SENSITIVE_PATH_PREFIXES:
            if candidate.startswith(prefix):
                return msg
        if candidate in _SENSITIVE_EXACT_PATHS:
            return msg
    return None


async def _get_mtime(file_ops: ShellFileOperations, path: str) -> float | None:
    """Return mtime of ``path`` inside the sandbox, or None if stat fails."""
    result = await file_ops._exec(f"stat -c %Y {file_ops._escape_shell_arg(path)} 2>/dev/null")
    if result.exit_code != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


async def write_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the write tool. See module docstring for error-shape rules."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise WriteArgumentError("write tool requires a non-empty 'path' string")

    content = arguments.get("content")
    if not isinstance(content, str):
        raise WriteArgumentError("write tool requires a 'content' string")

    # Sensitive-path check (pure).
    sensitive_msg = _check_sensitive_path(path)
    if sensitive_msg is not None:
        return {"error": sensitive_msg}

    # Char-count guard (pure).
    settings = get_settings()
    if len(content) > settings.file_write_max_chars:
        return {
            "error": (
                f"Write content is {len(content):,} characters which exceeds "
                f"the safety limit ({settings.file_write_max_chars:,} chars). "
                "Split the content across multiple files."
            ),
            "path": path,
        }

    sess = await file_session.get_or_create(session_id)

    async with sess.lock:
        # Staleness warning: if the model read this file earlier and
        # the file has been modified externally since, warn on write
        # (but still proceed -- this is advisory, not blocking).
        staleness_warning: str | None = None
        normpath = os.path.normpath(path)
        previous_mtime = sess.read_timestamps.get(normpath)
        if previous_mtime is not None:
            current_mtime = await _get_mtime(sess.file_ops, path)
            if current_mtime is not None and current_mtime != previous_mtime:
                staleness_warning = (
                    f"File {path} was modified externally since you last "
                    f"read it. The write succeeded but you should re-read "
                    "the file before making further edits."
                )

        # Perform the write.
        result = await sess.file_ops.write_file(path, content)
        result_dict = result.to_dict()

        if staleness_warning is not None:
            result_dict["_warning"] = staleness_warning

        # Refresh read_timestamps so subsequent writes from this session
        # don't re-warn about the same mtime.
        new_mtime = await _get_mtime(sess.file_ops, path)
        if new_mtime is not None:
            sess.read_timestamps[normpath] = new_mtime

        return result_dict


def _register() -> None:
    registry.register(
        name="write",
        description=WRITE_DESCRIPTION,
        parameters_schema=WRITE_PARAMETERS_SCHEMA,
        handler=write_handler,
    )


_register()
