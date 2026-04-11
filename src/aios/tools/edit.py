"""The edit tool -- targeted modifications to files inside the session's sandbox.

Two modes:

- ``mode='replace'`` -- find a unique string in ``path`` and replace it
  with ``new_string``. Uses the 8-strategy fuzzy match chain from
  :mod:`aios.vendor.hermes_files.fuzzy_match`, so minor whitespace or
  indentation differences between ``old_string`` and the actual file
  contents won't break the match.

- ``mode='patch'`` -- apply a V4A-format multi-file patch via
  :mod:`aios.vendor.hermes_files.patch_parser`. Supports
  ``*** Update File:``, ``*** Add File:``, ``*** Delete File:``, and
  ``*** Move File:`` operations in a single call.

Error-shape convention:

- **Raise** on malformed arguments (``EditArgumentError``).
- **Return a dict with ``error: ...``** for sensitive-path denial,
  fuzzy-match miss, unparseable patch, and ``patch_replace`` /
  ``patch_v4a`` failures.
"""

from __future__ import annotations

import os
import re
from typing import Any

from aios.errors import AiosError
from aios.tools import file_session
from aios.tools.registry import registry
from aios.tools.write import _check_sensitive_path
from aios.vendor.hermes_files.file_operations import ShellFileOperations


class EditArgumentError(AiosError):
    """Raised when the edit tool is called with malformed arguments."""

    error_type = "edit_argument_error"
    status_code = 400


EDIT_DESCRIPTION = (
    "Targeted edits to files inside the session's sandbox. Two modes:\n"
    "\n"
    '- `mode: "replace"` -- find a unique string in `path` and replace '
    "it with `new_string`. Uses an 8-strategy fuzzy match chain so "
    "minor whitespace/indentation differences won't break the match. "
    "If `old_string` matches multiple locations, set `replace_all: "
    "true` or include more context to disambiguate.\n"
    "\n"
    '- `mode: "patch"` -- apply a V4A-format multi-file patch '
    "(`*** Begin Patch` / `*** End Patch` envelope with `*** Update "
    "File:`, `*** Add File:`, `*** Delete File:`, `*** Move File:` "
    "operations). Pass the whole patch string in `patch`.\n"
    "\n"
    "Returns a unified diff showing the change. Use this instead of "
    "`sed`/`awk` via the bash tool -- fuzzy matching handles "
    "whitespace gracefully, multi-file patches land atomically."
)

EDIT_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["replace", "patch"],
            "description": (
                "'replace' for single-string find-and-replace; 'patch' for V4A multi-file patches."
            ),
        },
        "path": {
            "type": "string",
            "description": "File path (replace mode only).",
        },
        "old_string": {
            "type": "string",
            "description": (
                "String to find (replace mode only). Must be unique unless replace_all is true."
            ),
        },
        "new_string": {
            "type": "string",
            "description": "Replacement string (replace mode only).",
        },
        "replace_all": {
            "type": "boolean",
            "description": "If true, replace every occurrence (replace mode only).",
        },
        "patch": {
            "type": "string",
            "description": "V4A-format patch string (patch mode only).",
        },
    },
    "required": ["mode"],
    "additionalProperties": False,
}


def _patch_targets(patch_content: str) -> list[str]:
    """Return every file path referenced in a V4A patch header.

    Used for the sensitive-path check on ``mode='patch'`` -- we reject
    the whole patch if any of its target files are sensitive, before
    any modification happens.
    """
    paths: list[str] = []
    for line in patch_content.split("\n"):
        for pattern in (
            r"\*\*\*\s*Update\s+File:\s*(.+)",
            r"\*\*\*\s*Add\s+File:\s*(.+)",
            r"\*\*\*\s*Delete\s+File:\s*(.+)",
        ):
            m = re.match(pattern, line)
            if m:
                paths.append(m.group(1).strip())
                break
        # Move operations have two paths; record both.
        m = re.match(r"\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)", line)
        if m:
            paths.append(m.group(1).strip())
            paths.append(m.group(2).strip())
    return paths


async def _get_mtime(file_ops: ShellFileOperations, path: str) -> float | None:
    """Return mtime of ``path`` inside the sandbox, or None if stat fails."""
    result = await file_ops._exec(f"stat -c %Y {file_ops._escape_shell_arg(path)} 2>/dev/null")
    if result.exit_code != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


async def edit_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the edit tool. See module docstring for error-shape rules."""
    mode = arguments.get("mode")
    if mode not in ("replace", "patch"):
        raise EditArgumentError("edit tool requires mode='replace' or mode='patch'")

    sess = await file_session.get_or_create(session_id)

    if mode == "replace":
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EditArgumentError("edit (replace) requires a non-empty 'path' string")
        old_string = arguments.get("old_string")
        if not isinstance(old_string, str):
            raise EditArgumentError("edit (replace) requires an 'old_string' string")
        new_string = arguments.get("new_string")
        if not isinstance(new_string, str):
            raise EditArgumentError("edit (replace) requires a 'new_string' string")
        replace_all = bool(arguments.get("replace_all", False))

        sensitive_msg = _check_sensitive_path(path)
        if sensitive_msg is not None:
            return {"error": sensitive_msg}

        async with sess.lock:
            result = await sess.file_ops.patch_replace(
                path, old_string, new_string, replace_all=replace_all
            )
            result_dict = result.to_dict()

            # Refresh read_timestamps so subsequent writes don't re-warn.
            if result.success:
                mtime = await _get_mtime(sess.file_ops, path)
                if mtime is not None:
                    sess.read_timestamps[os.path.normpath(path)] = mtime

            return result_dict

    # mode == "patch"
    patch_content = arguments.get("patch")
    if not isinstance(patch_content, str) or not patch_content.strip():
        raise EditArgumentError("edit (patch) requires a non-empty 'patch' string")

    # Sensitive-path check on every target.
    targets = _patch_targets(patch_content)
    for target in targets:
        sensitive_msg = _check_sensitive_path(target)
        if sensitive_msg is not None:
            return {
                "error": (
                    f"Patch rejected: target '{target}' is a sensitive system path. {sensitive_msg}"
                )
            }

    async with sess.lock:
        result = await sess.file_ops.patch_v4a(patch_content)
        result_dict = result.to_dict()

        # Refresh read_timestamps for every touched path.
        if result.success:
            for touched in targets:
                mtime = await _get_mtime(sess.file_ops, touched)
                if mtime is not None:
                    sess.read_timestamps[os.path.normpath(touched)] = mtime

        return result_dict


def _register() -> None:
    registry.register(
        name="edit",
        description=EDIT_DESCRIPTION,
        parameters_schema=EDIT_PARAMETERS_SCHEMA,
        handler=edit_handler,
    )


_register()
