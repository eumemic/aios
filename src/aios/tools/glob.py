"""The glob tool — find files by pattern inside the session's sandbox.

A thin wrapper over ``rg --files --glob``. Returns a list of matching
file paths, capped at 500 results. Respects ``.gitignore`` automatically.

Return shape::

    {"matches": ["/workspace/foo.py", "/workspace/bar.py"]}

On failure (nonzero exit from ``rg``), returns ``{"error": "..."}``.
"""

from __future__ import annotations

import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.registry import registry


class GlobArgumentError(AiosError):
    """Raised when the glob tool is called with malformed arguments."""

    error_type = "glob_argument_error"
    status_code = 400


GLOB_DESCRIPTION = (
    "Find files matching a glob pattern inside the session's sandbox. "
    "Returns up to 500 matching file paths. Respects .gitignore. "
    "`path` is the directory to search in (default /workspace). "
    "`pattern` is a glob pattern like '*.py' or '**/*.test.ts'. "
    "Prefer this over `find` via the bash tool for simple file searches."
)

GLOB_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match file names against, e.g. '*.py'.",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in. Default /workspace.",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


async def glob_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the glob tool. See module docstring for the return shape."""
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise GlobArgumentError("glob tool requires a non-empty 'pattern' string")

    path = arguments.get("path", "/workspace")
    if not isinstance(path, str) or not path.strip():
        raise GlobArgumentError("path must be a non-empty string")

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id)

    cmd = f"rg --files --glob {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null | head -500"

    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )

    if result.exit_code != 0:
        return {
            "error": result.stderr.strip() or f"glob failed with exit code {result.exit_code}",
        }

    matches = [line for line in result.stdout.splitlines() if line.strip()]
    return {"matches": matches}


def _register() -> None:
    registry.register(
        name="glob",
        description=GLOB_DESCRIPTION,
        parameters_schema=GLOB_PARAMETERS_SCHEMA,
        handler=glob_handler,
    )


_register()
