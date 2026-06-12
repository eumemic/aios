"""The read tool — read a file inside the session's sandbox.

Text files return ``LINE_NUM<TAB>CONTENT`` windowed by ``offset`` /
``limit`` via ``cat -n | sed``.  Image files — detected by extension
(:data:`_EXT_TO_MIME`) or, for paths whose extension is not a known
image type (no extension, or a non-image extension such as a
connector-staged chat attachment), by a magic-byte sniff of the leading
bytes — return a content-parts list with an ``image_url`` block when the
bound model supports vision and the file fits the inline cap; otherwise
an explanatory ``ToolResult``.
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
    INLINE_SIZE_CAP_BYTES,
    can_inline_image,
    human_size,
    make_image_url_part,
    supports_vision,
)
from aios.sandbox.backends.base import SandboxHandle
from aios.sandbox.spec import resolve_bash_timeout_ceiling
from aios.sandbox.volumes import resolve_to_host_path
from aios.services import sessions as sessions_service
from aios.tools.memory_intercept import resolve_memory_target
from aios.tools.registry import ToolResult, registry
from aios_connector_http.mime import sniff_image_mime

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

    # Resolve the per-environment bash-timeout ceiling once (#725) and
    # thread it through every sandbox-exec this handler may make (the
    # text read below plus the image probe/stat helpers) so a raised
    # env limit applies to read just as it does to bash. Falls back to
    # the global default when no env override is set.
    exec_timeout = await resolve_bash_timeout_ceiling(session_id)

    image_mime = await _detect_image_mime(
        session_id, path, handle=handle, exec_timeout=exec_timeout
    )
    if image_mime is not None:
        return await _read_image(
            session_id=session_id,
            path=path,
            mime=image_mime,
            handle=handle,
            pool=pool,
            exec_timeout=exec_timeout,
        )

    target = resolve_memory_target(session_id, path)

    end = offset + limit - 1
    quoted_path = shlex.quote(path)
    sed_arg = shlex.quote(f"{offset},{end}p")
    # ``cat -n`` numbers lines (1-indexed) with the format ``   N\tCONTENT``;
    # ``sed -n 'START,ENDp'`` slices by line number against the already-
    # numbered output so the visible numbers are the file's actual line
    # numbers, not relative to the slice.
    #
    # ``set -o pipefail`` is load-bearing: without it the pipe's exit code
    # is ``sed``'s (0) even when ``cat`` failed (e.g., missing path), and
    # the existing ``exit_code != 0`` branch silently returns empty content
    # to the model. For memory targets it's worse — the empty sha line gets
    # cached into the read-sha map and poisons the next write-tool
    # precondition. ``bash -c`` is the sandbox exec shell so the option is
    # supported.
    if target is None:
        cmd = f"set -o pipefail; cat -n -- {quoted_path} | sed -n {sed_arg}"
    else:
        # Memory mounts: prepend the raw-file sha (one line, 64 hex chars)
        # so the write-tool precondition can be gated against the FS state
        # the model just observed. One docker-exec round-trip total.
        cmd = (
            f"set -o pipefail; "
            f"sha256sum -- {quoted_path} | cut -d' ' -f1 && "
            f"cat -n -- {quoted_path} | sed -n {sed_arg}"
        )

    result = await sandbox.exec(
        handle,
        cmd,
        timeout_seconds=exec_timeout,
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


async def _detect_image_mime(
    session_id: str, path: str, *, handle: SandboxHandle, exec_timeout: int
) -> str | None:
    """Return the image mime for ``path``, or ``None`` to read it as text.

    The extension decides when it names a known image type — the common,
    zero-IO case.  Otherwise (no extension, or a non-image extension:
    connector-staged chat attachments arrive with arbitrary or no
    extension) sniff the leading bytes.  The probe is read locally from the
    bind mount when ``path`` resolves into one and falls back to one
    docker-exec only for paths outside any mount.  The sniff decides routing
    only — the final declared mime is reconciled against the bytes actually
    inlined by :func:`make_image_url_part`, so a sniff/read race cannot ship
    a PNG/JPEG/GIF/WebP mime that disagrees with the inlined bytes.
    """
    ext = os.path.splitext(path)[1].lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime is not None:
        return mime
    head = await _read_probe_bytes(session_id, path, handle=handle, n=16, exec_timeout=exec_timeout)
    if head is None:
        return None
    return sniff_image_mime(head)


async def _read_image(
    *,
    session_id: str,
    path: str,
    mime: str,
    handle: SandboxHandle,
    pool: Any,
    exec_timeout: int,
) -> ToolResult:
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    model = await sessions_service.get_session_model(pool, session_id, account_id=account_id)

    host_path = resolve_to_host_path(session_id, path, workspace_path=handle.workspace_path)
    if host_path is not None:
        try:
            data = host_path.read_bytes()
        except FileNotFoundError:
            return ToolResult(content=f"file not found: {path}", is_error=True)
        except OSError as err:
            return ToolResult(content=f"read failed: {err}", is_error=True)
        size = len(data)
    else:
        result = await _stat_and_read_via_exec(handle, path, exec_timeout=exec_timeout)
        if result is None:
            return ToolResult(
                content=f"read failed: file not readable inside sandbox: {path}",
                is_error=True,
            )
        data, size = result

    if not can_inline_image(model=model, content_type=mime, size_bytes=size):
        vision = "yes" if supports_vision(model) else "no"
        # Render the cap with higher precision than ``human_size`` to
        # avoid the model getting "Image is 3.8MB, cap is 3.8MB" when
        # the truth is 3.93 MB > 3.75 MiB — ``human_size`` rounds both
        # to the same string.
        cap_mib = INLINE_SIZE_CAP_BYTES / (1024 * 1024)
        return ToolResult(
            content=(
                f"Image at {path} exists ({human_size(size)}, {mime}) but cannot "
                f"be inlined. Mind vision support: {vision}. "
                f"Inline cap: {cap_mib:.2f} MiB ({INLINE_SIZE_CAP_BYTES} bytes)."
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


async def _stat_and_read_via_exec(
    handle: SandboxHandle, path: str, *, exec_timeout: int
) -> tuple[bytes, int] | None:
    """Fetch ``(bytes, size)`` for non-bind-mount image paths via one docker-exec.

    Returns ``None`` on any read failure (missing path, exec error,
    unparseable size).  Combines stat + base64 into a single shell so
    we don't pay two docker-exec round-trips.
    """
    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    quoted = shlex.quote(path)
    cmd = f"stat -c %s -- {quoted} && base64 -w0 -- {quoted}"
    result = await sandbox.exec(
        handle,
        cmd,
        timeout_seconds=exec_timeout,
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


async def _read_probe_bytes(
    session_id: str, path: str, *, handle: SandboxHandle, n: int, exec_timeout: int
) -> bytes | None:
    """First ``n`` bytes of ``path`` for magic-byte image detection.

    Reads locally from the bind-mount source when ``path`` resolves into one
    (no docker-exec — the full read in :func:`_read_image` uses the same host
    fast path, so detection is free for the common ``/workspace`` and
    ``/mnt/attachments`` cases).  Falls back to one docker-exec for paths
    outside any mount.  Returns ``None`` when the file is unreadable — the
    caller treats that as "not an image" and the text path surfaces the real
    error.
    """
    host_path = resolve_to_host_path(session_id, path, workspace_path=handle.workspace_path)
    if host_path is not None:
        try:
            with host_path.open("rb") as fh:
                return fh.read(n)
        except OSError:
            return None
    return await _read_head_bytes(handle, path, n=n, exec_timeout=exec_timeout)


async def _read_head_bytes(
    handle: SandboxHandle, path: str, *, n: int, exec_timeout: int
) -> bytes | None:
    """Read the first ``n`` bytes of ``path`` via one docker-exec, base64-framed
    so binary survives the str stdout boundary.  Used as the out-of-mount
    fallback for magic-byte image detection.  ``set -o pipefail`` makes a
    failing ``head`` propagate (without it ``base64`` masks it with exit 0);
    returns ``None`` on any failure — the caller treats that as "not an image"
    and falls through to the text path.
    """
    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    quoted = shlex.quote(path)
    cmd = f"set -o pipefail; head -c {n} -- {quoted} | base64 -w0"
    result = await sandbox.exec(
        handle,
        cmd,
        timeout_seconds=exec_timeout,
        max_output_bytes=settings.bash_max_output_bytes,
    )
    if result.exit_code != 0:
        return None
    try:
        return base64.b64decode(result.stdout.strip())
    except (ValueError, TypeError):
        return None


def _register() -> None:
    registry.register(
        name="read",
        description=READ_DESCRIPTION,
        parameters_schema=READ_PARAMETERS_SCHEMA,
        handler=read_handler,
        transport="agent_tool",
        executes="sandbox",
    )


_register()
