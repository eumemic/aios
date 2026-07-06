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
from dataclasses import dataclass
from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.harness.image_resize import ImageDownsampleError, maybe_downsample
from aios.harness.vision import (
    INLINE_SIZE_CAP_BYTES,
    PROVIDER_INLINE_IMAGE_FORMATS,
    can_inline_image,
    human_size,
    inline_image_format,
    make_image_url_part,
    supports_vision,
)
from aios.sandbox.backends.base import SandboxHandle
from aios.sandbox.spec import resolve_bash_timeout_ceiling
from aios.sandbox.volumes import resolve_to_host_path
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
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
    "Read a file inside the session's sandbox. For text files, returns "
    "content prefixed with 1-indexed line numbers in `LINE_NUM<TAB>CONTENT` "
    "format. Use `offset` and `limit` to page through large files — "
    "`offset` is a 1-indexed line number to start from (default 1), "
    "`limit` is the max number of lines to return (default 2000). "
    "Paths may be absolute or relative to /workspace. Prefer this "
    "over `cat`/`head`/`tail` via the bash tool when you want "
    "structured line-numbered output; use bash for one-off inspection. "
    "For images (PNG/JPEG/GIF/WEBP), reading the file inlines the image "
    "into your visual context so you can actually see it — use this to "
    "view screenshots and other sandbox images directly (no need to "
    "describe or measure them indirectly). This works when the bound "
    "model supports vision and the file fits the inline cap (~3.75 MiB); "
    "a larger image returns an explanatory text result instead of the "
    "pixels, so capture at viewport size or downscale/JPEG to fit."
)

READ_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Path to the file. Absolute or relative to /workspace. An "
                "image path (PNG/JPEG/GIF/WEBP) returns the image into your "
                "visual context instead of text (vision-capable models, "
                "images up to ~3.75 MiB)."
            ),
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
        raise ToolBail(
            result.stderr.strip() or f"read failed with exit code {result.exit_code}",
            detail={"path": path},
        )

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
        exec_result = await _stat_and_read_via_exec(handle, path, exec_timeout=exec_timeout)
        match exec_result:
            case _ExecImageOk(data=d, size=s):
                data, size = d, s
            case _ExecImageTruncated():
                cap = get_settings().bash_max_output_bytes
                return ToolResult(
                    content=(
                        f"Image at {path} is too large to read through the sandbox: "
                        f"its base64 transfer exceeded the {cap}-byte exec output cap. "
                        f"Stage it under /workspace, or process it in place with another tool."
                    ),
                    is_error=True,
                )
            case _ExecImageFailed(detail=detail):
                return ToolResult(
                    content=f"read failed inside sandbox: {path}: {detail}",
                    is_error=True,
                )

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

    image_format = inline_image_format(data)
    if image_format is None or image_format not in PROVIDER_INLINE_IMAGE_FORMATS:
        # Passes the mime+size+vision gate but the provider will reject it:
        # undecodable (corrupt/truncated body behind a valid magic prefix) or a
        # decodable-but-unsupported format (e.g. a TIFF/BMP saved as .png — the
        # .png extension skips the magic re-sniff). Inlining persists the bad
        # data URI into the tool_result event, which build_messages replays on
        # every wake → a 400 that terminally errors the turn the model can't
        # see. Same decode+format gate the attachment render path applies.
        detail = (
            "its bytes do not decode as an image"
            if image_format is None
            else f"its format ({image_format}) is not one a vision model accepts "
            "(only JPEG/PNG/GIF/WEBP)"
        )
        return ToolResult(
            content=(
                f"Image at {path} ({human_size(size)}, {mime}) cannot be inlined: "
                f"{detail}. The file is present; use other tools to process it if needed."
            ),
            is_error=False,
        )

    # Downscale full-resolution bytes to the inline dimension cap before
    # encoding. ``read()`` previously inlined the raw bytes at full
    # resolution; once >20 such images accumulate in a window and at least
    # one exceeds 2000px on a side, Anthropic HARD-REJECTS the many-image
    # request (400, no server-side resize), wedging every subsequent wake
    # (the bytes are frozen in the tool_result event and replayed verbatim).
    # ``maybe_downsample`` returns ``None`` for the common case (already
    # <=2000px on each side and <=cap) via a header-only check, so the hot
    # path pays no decode/re-encode cost and loses no quality.
    try:
        resized = await maybe_downsample(data, mime)
    except ImageDownsampleError as err:
        # Oversize-beyond-ceiling or undecodable: degrade to a text marker
        # the model can still act on — same stance as the non-inlinable
        # branches above. Do NOT inline the oversize original; that is the
        # wedge vector.
        return ToolResult(
            content=(
                f"Image at {path} ({human_size(size)}, {mime}) is present but could "
                f"not be prepared for inline viewing ({err}). Use other tools to "
                f"process it, or capture/downscale it to <=2000px on the longest edge."
            ),
            is_error=False,
        )
    if resized is not None:
        # Use the RETURNED content_type — ``maybe_downsample`` switches an
        # opaque PNG to JPEG, so reusing the original ``mime`` would mislabel
        # the data URI.
        data, mime, size = resized.data, resized.content_type, len(resized.data)

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


@dataclass(frozen=True, slots=True)
class _ExecImageOk:
    data: bytes
    size: int


@dataclass(frozen=True, slots=True)
class _ExecImageTruncated:
    """base64 stream exceeded the exec output cap (exit 0, truncated host-side)."""


@dataclass(frozen=True, slots=True)
class _ExecImageFailed:
    """stat/base64 failed or its output was unparseable; carries the real stderr."""

    detail: str


type _ExecImageResult = _ExecImageOk | _ExecImageTruncated | _ExecImageFailed


async def _stat_and_read_via_exec(
    handle: SandboxHandle, path: str, *, exec_timeout: int
) -> _ExecImageResult:
    """Fetch image bytes for non-bind-mount paths via one docker-exec.

    Returns a kind-tagged :data:`_ExecImageResult` so the caller can render
    each failure mode truthfully:

    * :class:`_ExecImageOk` — bytes + size read successfully.
    * :class:`_ExecImageTruncated` — the base64 stream exceeded the exec
      output cap (host-side truncation, exit 0); the file is readable but
      too large to ship through the cap.
    * :class:`_ExecImageFailed` — stat/base64 failed or produced
      unparseable output; carries the real stderr.

    Combines stat + base64 into a single shell so we don't pay two
    docker-exec round-trips.
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
    # Ordering is load-bearing: a truncated base64 stream both has
    # ``exit_code == 0`` and fails ``b64decode`` (the host appends an
    # ``[output truncated]`` trailer).  Checking ``truncated`` first makes
    # mis-routing it into the parse-failure arm unrepresentable.
    if result.truncated:
        return _ExecImageTruncated()
    if result.exit_code != 0:
        return _ExecImageFailed(result.stderr.strip() or f"exit code {result.exit_code}")
    size_line, _, b64 = result.stdout.partition("\n")
    try:
        size = int(size_line.strip())
        data = base64.b64decode(b64.strip())
    except (ValueError, TypeError):
        return _ExecImageFailed(
            result.stderr.strip() or "stat/base64 output was not parseable (corrupt or unexpected)"
        )
    return _ExecImageOk(data=data, size=size)


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
