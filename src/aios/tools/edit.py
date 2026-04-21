"""The edit tool — find-and-replace in a file inside the session's sandbox.

Reads the file via ``cat``, strict ``str.find`` against ``old_string``,
writes the modified content back via base64 stdin, returns a unified
diff computed in-process by :mod:`difflib`.

**Strict matching only.** If ``old_string`` doesn't appear byte-for-byte
in the file, the tool returns an error telling the model to re-read the
file and try again with exact content. No fuzzy matching, no whitespace
normalization, no heuristics. Models that produce imprecise arguments
will see an error, read the file, and retry — that's the session log
doing its job as the source of truth.

If ``old_string`` matches multiple locations, the tool rejects the call
and asks for more context (or ``replace_all: true``). This prevents
silent wrong-region edits.

Return shape on success::

    {"path": "/workspace/foo.py", "diff": "--- ...\\n+++ ...\\n..."}

On failure, returns ``{"error": "...", "path": path}``.
"""

from __future__ import annotations

import base64
import difflib
import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.registry import registry


class EditArgumentError(AiosError):
    """Raised when the edit tool is called with malformed arguments."""

    error_type = "edit_argument_error"
    status_code = 400


EDIT_DESCRIPTION = (
    "Find and replace text in a file inside the session's sandbox. "
    "Uses strict byte-for-byte matching: `old_string` must appear "
    "exactly as-is in the file, or the call fails with a hint to "
    "re-read the file. If `old_string` appears in multiple locations, "
    "the call fails unless you set `replace_all: true` or include "
    "more surrounding context to disambiguate. Returns a unified "
    "diff of the change. Prefer this over `sed`/`awk` via the bash "
    "tool: strict matching is more predictable than shell pattern "
    "escaping, and the diff output is easier to verify."
)

EDIT_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file. Absolute or relative to /workspace.",
        },
        "old_string": {
            "type": "string",
            "description": (
                "Exact text to find. Must match byte-for-byte. If it "
                "matches multiple locations, either add surrounding "
                "context to disambiguate or set replace_all=true."
            ),
        },
        "new_string": {
            "type": "string",
            "description": "Replacement text.",
        },
        "replace_all": {
            "type": "boolean",
            "description": (
                "If true, replace every occurrence of old_string. "
                "Default false (requires a unique match)."
            ),
        },
    },
    "required": ["path", "old_string", "new_string"],
    "additionalProperties": False,
}


async def edit_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the edit tool. See module docstring for the return shape."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise EditArgumentError("edit tool requires a non-empty 'path' string")

    old_string = arguments.get("old_string")
    if not isinstance(old_string, str) or not old_string:
        raise EditArgumentError("edit tool requires a non-empty 'old_string' string")

    new_string = arguments.get("new_string")
    if not isinstance(new_string, str):
        raise EditArgumentError("edit tool requires a 'new_string' string")

    replace_all = bool(arguments.get("replace_all", False))

    if old_string == new_string:
        return {
            "error": "old_string and new_string are identical; nothing to change",
            "path": path,
        }

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())

    quoted_path = shlex.quote(path)

    # Step 1: read the current content.
    read_result = await handle.run_command(
        f"cat -- {quoted_path}",
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )
    if read_result.exit_code != 0:
        return {
            "error": read_result.stderr.strip() or f"could not read {path}",
            "path": path,
        }

    original = read_result.stdout
    match_count = original.count(old_string)

    if match_count == 0:
        return {
            "error": (
                f"old_string not found in {path}. Use the read tool to see "
                "the exact file content, then retry edit with text that "
                "matches byte-for-byte."
            ),
            "path": path,
        }
    if match_count > 1 and not replace_all:
        return {
            "error": (
                f"old_string matches {match_count} locations in {path}. "
                "Add surrounding context to make it unique, or set "
                "replace_all=true to replace every occurrence."
            ),
            "path": path,
            "matches": match_count,
        }

    if replace_all:
        modified = original.replace(old_string, new_string)
    else:
        modified = original.replace(old_string, new_string, 1)

    # Step 2: write the modified content back via the same base64
    # stdin mechanism the write tool uses.
    modified_bytes = modified.encode("utf-8")
    b64 = base64.b64encode(modified_bytes).decode("ascii")
    write_cmd = f"base64 -d <<< '{b64}' > {quoted_path}"

    write_result = await handle.run_command(
        write_cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )
    if write_result.exit_code != 0:
        return {
            "error": (
                write_result.stderr.strip()
                or f"write-back failed with exit code {write_result.exit_code}"
            ),
            "path": path,
        }

    # Step 3: compute the diff in-process. splitlines(keepends=True)
    # preserves trailing newlines so difflib produces a clean unified diff.
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a{path}",
            tofile=f"b{path}",
        )
    )

    return {
        "path": path,
        "diff": diff,
        "replaced": match_count if replace_all else 1,
    }


def _register() -> None:
    registry.register(
        name="edit",
        description=EDIT_DESCRIPTION,
        parameters_schema=EDIT_PARAMETERS_SCHEMA,
        handler=edit_handler,
    )


_register()
