"""The read tool — read a file inside the session's sandbox.

A thin wrapper over ``cat -n`` piped through ``sed``. Returns file
content with ``LINE_NUM<TAB>CONTENT`` line numbers, windowed by
``offset`` (1-indexed) and ``limit`` so the model can page through
large files without blowing its context window in one shot.

The tool shells out once per call via :meth:`ContainerHandle.run_command`
and trusts the model for everything else: no binary-file guard, no
device blocklist, no char ceiling, no dedup. If the model reads
``/dev/zero``, the container timeout (``bash_default_timeout_seconds``)
kills it and the model sees an error. If the model re-reads the same
range 10 times, that shows up in the session log and the model pays
the token cost.

Return shape::

    {"path": "/workspace/foo.py", "content": "     1\\thello\\n     2\\tworld"}

On failure (nonzero exit from ``cat``), returns ``{"error": "..."}``.
"""

from __future__ import annotations

import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.memory_intercept import resolve_memory_target
from aios.tools.registry import registry


class ReadArgumentError(AiosError):
    """Raised when the read tool is called with malformed arguments."""

    error_type = "read_argument_error"
    status_code = 400


READ_DESCRIPTION = (
    "Read a text file inside the session's sandbox. Returns content "
    "prefixed with 1-indexed line numbers in `LINE_NUM<TAB>CONTENT` "
    "format. Use `offset` and `limit` to page through large files — "
    "`offset` is a 1-indexed line number to start from (default 1), "
    "`limit` is the max number of lines to return (default 2000). "
    "Paths may be absolute or relative to /workspace. Prefer this "
    "over `cat`/`head`/`tail` via the bash tool when you want "
    "structured line-numbered output; use bash for one-off inspection."
)

READ_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file. Absolute or relative to /workspace.",
        },
        "offset": {
            "type": "integer",
            "description": "1-indexed line number to start reading from. Default 1.",
            "minimum": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum lines to return. Default 2000.",
            "minimum": 1,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


async def read_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the read tool. See module docstring for the return shape."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ReadArgumentError("read tool requires a non-empty 'path' string")

    offset = arguments.get("offset", 1)
    if not isinstance(offset, int) or offset < 1:
        raise ReadArgumentError("offset must be a positive integer")

    limit = arguments.get("limit", 2000)
    if not isinstance(limit, int) or limit < 1:
        raise ReadArgumentError("limit must be a positive integer")

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())
    target = resolve_memory_target(session_id, path)

    end = offset + limit - 1
    quoted_path = shlex.quote(path)
    sed_arg = shlex.quote(f"{offset},{end}p")
    # cat -n numbers lines (1-indexed) with the format `   N\tCONTENT`;
    # sed -n 'START,ENDp' slices by line number against the already-
    # numbered output so the visible numbers are the file's actual line
    # numbers, not relative to the slice.
    if target is None:
        cmd = f"cat -n -- {quoted_path} | sed -n {sed_arg}"
    else:
        # Memory mounts: prepend the raw-file sha (one line, 64 hex chars)
        # so the write-tool precondition can be gated against the FS state
        # the model just observed. One docker-exec round-trip total.
        cmd = (
            f"sha256sum -- {quoted_path} | cut -d' ' -f1 && "
            f"cat -n -- {quoted_path} | sed -n {sed_arg}"
        )

    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )

    if result.exit_code != 0:
        return {
            "error": result.stderr.strip() or f"read failed with exit code {result.exit_code}",
            "path": path,
        }

    if target is None:
        return {"path": path, "content": result.stdout}

    sha_line, _, content = result.stdout.partition("\n")
    runtime.set_read_sha(session_id, target.store_id, target.store_path, sha_line.strip())
    return {"path": path, "content": content}


def _register() -> None:
    registry.register(
        name="read",
        description=READ_DESCRIPTION,
        parameters_schema=READ_PARAMETERS_SCHEMA,
        handler=read_handler,
    )


_register()
