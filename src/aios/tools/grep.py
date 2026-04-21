"""The grep tool — search file contents inside the session's sandbox.

A thin wrapper over ``rg`` (ripgrep). Returns matching lines with file paths
and line numbers, capped at 250 lines by default.

Supports multiple output modes (content, files_with_matches, count),
context lines, case-insensitive search, multiline regex, and file type
filtering.

Return shape::

    {"matches": "path/foo.py:10:def hello()\\npath/bar.py:3:import os"}

On failure (nonzero exit other than 1, which means no matches), returns
``{"error": "..."}``. Exit code 1 (no matches) returns an empty string.
"""

from __future__ import annotations

import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.registry import registry


class GrepArgumentError(AiosError):
    """Raised when the grep tool is called with malformed arguments."""

    error_type = "grep_argument_error"
    status_code = 400


GREP_DESCRIPTION = (
    "Search file contents for a regex pattern inside the session's sandbox. "
    "Returns matching lines with file paths and line numbers by default. "
    "Supports output modes: 'content' (default), 'files_with_matches', "
    "'count'. Respects .gitignore automatically. "
    "`pattern` is a regex pattern. `path` is the directory or file to "
    "search in (default /workspace). `include` is an optional file glob "
    "filter like '*.py'. Prefer this over `grep` via the bash tool."
)

GREP_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in. Default /workspace.",
        },
        "include": {
            "type": "string",
            "description": "Optional file glob filter, e.g. '*.py'.",
        },
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
            "description": (
                "Output mode: 'content' (matching lines, default), "
                "'files_with_matches' (file paths only), "
                "'count' (match counts per file)."
            ),
        },
        "context": {
            "type": "integer",
            "description": "Lines of context around matches (content mode only).",
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Case-insensitive search.",
        },
        "multiline": {
            "type": "boolean",
            "description": "Enable multiline regex mode.",
        },
        "file_type": {
            "type": "string",
            "description": "File type filter, e.g. 'py', 'js', 'rust'.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum output lines (default 250).",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


async def grep_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the grep tool. See module docstring for the return shape."""
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise GrepArgumentError("grep tool requires a non-empty 'pattern' string")

    path = arguments.get("path", "/workspace")
    if not isinstance(path, str) or not path.strip():
        raise GrepArgumentError("path must be a non-empty string")

    include = arguments.get("include")
    if include is not None and (not isinstance(include, str) or not include.strip()):
        raise GrepArgumentError("include must be a non-empty string if provided")

    output_mode = arguments.get("output_mode", "content")
    context = arguments.get("context")
    case_insensitive = arguments.get("case_insensitive", False)
    multiline = arguments.get("multiline", False)
    file_type = arguments.get("file_type")
    limit = arguments.get("limit", 250)

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())

    parts = ["rg"]

    # Output mode flags
    if output_mode == "files_with_matches":
        parts.append("-l")
    elif output_mode == "count":
        parts.append("-c")

    # Line numbers in content mode
    if output_mode == "content":
        parts.append("-n")

    # Optional flags
    if case_insensitive:
        parts.append("-i")
    if multiline:
        parts.extend(["-U", "--multiline-dotall"])
    if context and output_mode == "content":
        parts.extend(["-C", str(context)])
    if include:
        parts.extend(["--glob", shlex.quote(include)])
    if file_type:
        parts.extend(["--type", shlex.quote(file_type)])

    # Prevent base64/minified line noise
    parts.append("--max-columns=500")

    parts.append(shlex.quote(pattern))
    parts.append(shlex.quote(path))
    parts.append("2>/dev/null")
    parts.append(f"| head -{limit}")

    cmd = " ".join(parts)

    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )

    # rg exit code 1 means no matches — not an error.
    if result.exit_code not in (0, 1):
        return {
            "error": result.stderr.strip() or f"grep failed with exit code {result.exit_code}",
        }

    return {"matches": result.stdout}


def _register() -> None:
    registry.register(
        name="grep",
        description=GREP_DESCRIPTION,
        parameters_schema=GREP_PARAMETERS_SCHEMA,
        handler=grep_handler,
    )


_register()
