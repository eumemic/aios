"""The read tool — read a file inside the session's sandbox.

Text files return ``LINE_NUM<TAB>CONTENT`` windowed by ``offset`` /
``limit`` via ``cat -n | sed``.  Image files (extensions in
:data:`_EXT_TO_MIME`) return a content-parts list with an
``image_url`` block when the bound mind supports vision and the file
fits the inline cap; otherwise an explanatory ``ToolResult``.
"""

from __future__ import annotations

import base64
import os
import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.harness.vision import (
    can_inline_image,
    human_size,
    make_image_url_part,
    supports_vision,
)
from aios.sandbox.container import ContainerHandle
from aios.sandbox.volumes import resolve_to_host_path
from aios.services import sessions as sessions_service
from aios.tools.memory_intercept import resolve_memory_target
from aios.tools.registry import ToolResult, registry

_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


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


async def read_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any] | ToolResult:
    """Handler for the read tool. See module docstring for return shapes."""
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
    pool = runtime.require_pool()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=pool)

    if _looks_like_image(path):
        return await _read_image(session_id=session_id, path=path, handle=handle, pool=pool)

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


def _looks_like_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _EXT_TO_MIME


async def _read_image(
    *,
    session_id: str,
    path: str,
    handle: ContainerHandle,
    pool: Any,
) -> ToolResult:
    mime = _EXT_TO_MIME[os.path.splitext(path)[1].lower()]
    model = await sessions_service.get_session_model(pool, session_id)

    host_path = resolve_to_host_path(session_id, path)
    if host_path is not None:
        try:
            data = host_path.read_bytes()
        except FileNotFoundError:
            return ToolResult(content=f"file not found: {path}", is_error=True)
        except OSError as err:
            return ToolResult(content=f"read failed: {err}", is_error=True)
        size = len(data)
    else:
        result = await _stat_and_read_via_exec(handle, path)
        if result is None:
            return ToolResult(
                content=f"read failed: file not readable inside sandbox: {path}",
                is_error=True,
            )
        data, size = result

    if not can_inline_image(model=model, content_type=mime, size_bytes=size):
        vision = "yes" if supports_vision(model) else "no"
        return ToolResult(
            content=(
                f"Image at {path} exists ({human_size(size)}, {mime}) but cannot "
                f"be inlined. Mind vision support: {vision}. Inline cap: 2 MiB."
            ),
            is_error=False,
        )

    encoded = base64.b64encode(data).decode("ascii")
    return ToolResult(
        content=[
            {
                "type": "text",
                "text": f"Image: {os.path.basename(path)} ({mime}, {human_size(size)})",
            },
            make_image_url_part(content_type=mime, data_b64=encoded),
        ],
    )


async def _stat_and_read_via_exec(handle: ContainerHandle, path: str) -> tuple[bytes, int] | None:
    """Fetch ``(bytes, size)`` for non-bind-mount image paths via one docker-exec.

    Returns ``None`` on any read failure (missing path, exec error,
    unparseable size).  Combines stat + base64 into a single shell so
    we don't pay two docker-exec round-trips.
    """
    settings = get_settings()
    quoted = shlex.quote(path)
    cmd = f"stat -c %s -- {quoted} && base64 -w0 -- {quoted}"
    result = await handle.run_command(
        cmd,
        timeout_seconds=settings.bash_default_timeout_seconds,
        max_output_bytes=settings.bash_max_output_bytes,
    )
    if result.exit_code != 0:
        return None
    size_line, _, b64 = result.stdout.partition("\n")
    try:
        size = int(size_line.strip())
        data = base64.b64decode(b64.strip())
    except (ValueError, TypeError):
        return None
    return data, size


def _register() -> None:
    registry.register(
        name="read",
        description=READ_DESCRIPTION,
        parameters_schema=READ_PARAMETERS_SCHEMA,
        handler=read_handler,
    )


_register()
