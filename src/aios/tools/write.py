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

On an expected failure, raises :class:`~aios.tools.invoke.ToolBail`
(one typed failure channel — #1680) whose ``detail`` carries the
``{path[, detail, matches]}`` context; the single event writer stamps
``is_error`` and merges the detail into the tool-result ``{"error": ...}``
content.
"""

from __future__ import annotations

import base64
import hashlib
import shlex
from typing import Any

from aios.config import get_settings
from aios.errors import (
    AiosError,
    MemoryPathConflictError,
    MemoryPreconditionFailedError,
    MemoryStoreArchivedError,
)
from aios.harness import runtime
from aios.models.memory_stores import MAX_CONTENT_BYTES
from aios.sandbox.spec import resolve_bash_timeout_ceiling
from aios.services import memory_stores as memory_service
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.memory_intercept import resolve_memory_target
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
    account_id = await sessions_service.load_session_account_id(runtime.require_pool(), session_id)
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise WriteArgumentError("write tool requires a non-empty 'path' string")

    content = arguments.get("content")
    if not isinstance(content, str):
        raise WriteArgumentError("write tool requires a 'content' string")

    target = resolve_memory_target(session_id, path)
    if target is not None:
        if target.access == "read_only":
            raise ToolBail(
                f"memory store {target.store_name!r} is mounted read_only; cannot write",
                detail={"path": path},
            )
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ToolBail(
                f"content exceeds memory store cap of {MAX_CONTENT_BYTES} "
                f"bytes (got {len(content.encode('utf-8'))})",
                detail={"path": path},
            )

    settings = get_settings()
    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())

    if target is not None:
        # Durable DB write first; the FS mirror below is for in-container
        # visibility. The cached read sha gates the update so concurrent
        # modifications surface as a typed precondition error.
        pool = runtime.require_pool()
        precondition_sha = runtime.get_read_sha(session_id, target.store_id, target.store_path)
        try:
            existing = await memory_service.get_memory_by_path(
                pool,
                target.store_id,
                target.store_path,
                include_content=False,
                account_id=account_id,
            )
            if existing is None:
                await memory_service.create_memory(
                    pool,
                    store_id=target.store_id,
                    path=target.store_path,
                    content=content,
                    actor=memory_service.SessionActor(session_id=session_id),
                    account_id=account_id,
                )
            else:
                await memory_service.update_memory(
                    pool,
                    store_id=target.store_id,
                    memory_id=existing.id,
                    new_content=content,
                    precondition_sha256=precondition_sha,
                    actor=memory_service.SessionActor(session_id=session_id),
                    account_id=account_id,
                )
        except MemoryPathConflictError as exc:
            raise ToolBail(exc.message, detail={"path": path, "detail": exc.detail}) from exc
        except MemoryPreconditionFailedError as exc:
            # Curated retry guidance the raw ``exc.message`` does not carry — keep it
            # verbatim through the typed channel rather than letting _classify_tool_error's
            # ``to_message()`` render the terse store-level message (#1680).
            raise ToolBail(
                f"the file at {path} changed since your last read; re-read it and retry the write",
                detail={"path": path, "detail": exc.detail},
            ) from exc
        except MemoryStoreArchivedError as exc:
            raise ToolBail(exc.message, detail={"path": path}) from exc

        # Refresh the read-sha cache so a subsequent write tool call against
        # this same path doesn't fail its precondition with the now-stale
        # pre-write sha. Mirrors ``edit.py``'s post-update refresh; without
        # this the model gets a misleading "file changed since your last
        # read" error on its OWN second write.
        runtime.set_read_sha(
            session_id,
            target.store_id,
            target.store_path,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    content_bytes = content.encode("utf-8")
    b64 = base64.b64encode(content_bytes).decode("ascii")

    quoted_path = shlex.quote(path)
    # Here-string ('<<<') feeds the single-quoted base64 payload to
    # base64 -d's stdin. Safe because base64's alphabet excludes the
    # single quote. `mkdir -p "$(dirname ...)"` creates parent dirs.
    cmd = f"mkdir -p -- \"$(dirname -- {quoted_path})\" && base64 -d <<< '{b64}' > {quoted_path}"

    result = await sandbox.exec(
        handle,
        cmd,
        # Honor the session's per-environment bash-timeout ceiling (#725)
        # so an env that raised the limit applies to every sandbox-exec
        # tool, not just bash. Falls back to the global default when no
        # env override is set.
        timeout_seconds=await resolve_bash_timeout_ceiling(session_id),
        max_output_bytes=settings.bash_max_output_bytes,
    )

    if result.exit_code != 0:
        # FS mirror failed AFTER the durable DB write committed. Tell the
        # model exactly that — the bytes are persisted, but the in-container
        # view of the file won't reflect them until the session restarts.
        if target is not None:
            raise ToolBail(
                "durable write succeeded but the in-container mirror "
                f"failed: {result.stderr.strip() or f'exit {result.exit_code}'}. "
                "Subsequent reads in this session may return stale "
                "content until the session restarts.",
                detail={"path": path},
            )
        raise ToolBail(
            result.stderr.strip() or f"write failed with exit code {result.exit_code}",
            detail={"path": path},
        )

    return {"path": path, "bytes_written": len(content_bytes)}


def _register() -> None:
    registry.register(
        name="write",
        description=WRITE_DESCRIPTION,
        parameters_schema=WRITE_PARAMETERS_SCHEMA,
        handler=write_handler,
        transport="agent_tool",
        executes="sandbox",
    )


_register()
