"""The write tool — write a file inside the session's sandbox.

A thin wrapper that base64-encodes the content on the host and pipes
it into ``base64 -d > path`` inside the container via a here-string.
Base64 output uses only ``[A-Za-z0-9+/=]``, so it's safe to embed
inside a shell here-string with a single-quoted payload — no
delimiter collision, no quoting surprises on NUL bytes or newlines.

This is the one capability ``bash`` alone can't provide cleanly: for
content beyond trivial size, heredocs break on delimiter-in-content
and ``echo -n '...'`` breaks on single quotes inside the payload. The
write tool gives the model a reliable channel for arbitrary text.

No deny list, no staleness warning, no char ceiling. The sandbox
filesystem boundary IS the security boundary — writes inside the
container are isolated from the host. If the model writes something
harmful to ``/etc/passwd`` inside its own container, the only thing
it hurts is its own container, which gets torn down at turn end
anyway. Writes larger than the host's ``ARG_MAX`` fail naturally
with a shell error surfaced back to the model.

Return shape::

    {"path": "/workspace/foo.py", "bytes_written": 42}

On failure, returns ``{"error": "...", "path": path}``.
"""

from __future__ import annotations

import base64
import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.registry import registry


class WriteArgumentError(AiosError):
    """Raised when the write tool is called with malformed arguments."""

    error_type = "write_argument_error"
    status_code = 400


WRITE_DESCRIPTION = (
    "Write a text file inside the session's sandbox. Creates parent "
    "directories as needed and overwrites any existing file at the "
    "path. Paths may be absolute or relative to /workspace. Prefer "
    "this over `echo > file` or heredoc tricks via the bash tool: "
    "this tool handles arbitrary content safely (quotes, newlines, "
    "NUL bytes, special characters) via base64 stdin piping."
)

WRITE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file. Absolute or relative to /workspace.",
        },
        "content": {
            "type": "string",
            "description": "Full file content to write. Overwrites any existing file.",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


async def write_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the write tool. See module docstring for the return shape."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise WriteArgumentError("write tool requires a non-empty 'path' string")

    content = arguments.get("content")
    if not isinstance(content, str):
        raise WriteArgumentError("write tool requires a 'content' string")

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())

    content_bytes = content.encode("utf-8")
    b64 = base64.b64encode(content_bytes).decode("ascii")

    quoted_path = shlex.quote(path)
    # Here-string ('<<<') feeds the single-quoted base64 payload to
    # base64 -d's stdin. Safe because base64's alphabet excludes the
    # single quote. `mkdir -p "$(dirname ...)"` creates parent dirs.
    cmd = f"mkdir -p -- \"$(dirname -- {quoted_path})\" && base64 -d <<< '{b64}' > {quoted_path}"

    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )

    if result.exit_code != 0:
        return {
            "error": result.stderr.strip() or f"write failed with exit code {result.exit_code}",
            "path": path,
        }

    return {"path": path, "bytes_written": len(content_bytes)}


def _register() -> None:
    registry.register(
        name="write",
        description=WRITE_DESCRIPTION,
        parameters_schema=WRITE_PARAMETERS_SCHEMA,
        handler=write_handler,
    )


_register()
