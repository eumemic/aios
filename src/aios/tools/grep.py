"""The grep tool — search file contents inside the session's sandbox.

A thin wrapper over ``grep -rn``. Returns matching lines with file paths
and line numbers, capped at 200 lines.

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
    "Returns up to 200 matching lines with file paths and line numbers. "
    "`pattern` is a regex pattern. `path` is the directory or file to "
    "search in (default /workspace). `include` is an optional file glob "
    "filter like '*.py'. Prefer this over `grep` via the bash tool for "
    "simple content searches."
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

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id)

    include_flag = f" --include={shlex.quote(include)}" if include else ""
    cmd = (
        f"grep -rn{include_flag} {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null | head -200"
    )

    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )

    # grep exit code 1 means no matches — not an error.
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
