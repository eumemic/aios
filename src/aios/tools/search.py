"""The search tool -- search file contents or find files.

Thin handler over
:class:`~aios.vendor.hermes_files.file_operations.ShellFileOperations`'s
``search`` method. Two targets:

- ``target='content'`` (default) -- regex search inside files using
  ripgrep (with grep fallback). Output modes: ``content`` (matches
  with line numbers), ``files_only`` (unique file paths), ``count``
  (matches per file). Optional ``context`` for surrounding lines.
- ``target='files'`` -- find files by glob pattern. Uses ripgrep's
  ``--files`` mode when available (respects .gitignore, skips hidden
  directories); falls back to ``find``.

Applies a consecutive-search guard: the same search key repeated 3
times in a row returns a warning, 4 times hard-blocks.
"""

from __future__ import annotations

from typing import Any

from aios.errors import AiosError
from aios.tools import file_session
from aios.tools.registry import registry


class SearchArgumentError(AiosError):
    """Raised when the search tool is called with malformed arguments."""

    error_type = "search_argument_error"
    status_code = 400


SEARCH_DESCRIPTION = (
    "Search file contents or find files inside the session's sandbox. "
    "Ripgrep-backed when available (fast, .gitignore-aware). Two "
    "targets via `target`:\n"
    "\n"
    '- `target: "content"` (default) -- regex search inside files. '
    'Filter by `file_glob` (e.g. `"*.py"`). Output modes: `content` '
    "(default; matches with line numbers), `files_only` (unique file "
    "paths containing a match), `count` (matches per file). Optional "
    "`context` for surrounding lines.\n"
    "\n"
    '- `target: "files"` -- find files by glob pattern (e.g. '
    '`"*.py"`). Sorted by modification time when ripgrep is '
    "available.\n"
    "\n"
    "Use this instead of `grep`/`rg`/`find`/`ls` via the bash tool. "
    "Results are paginated via `limit` (default 50) and `offset`. "
    "The default path `.` means `/workspace`."
)

SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": ("Regex (content target) or glob (files target) to search for."),
        },
        "target": {
            "type": "string",
            "enum": ["content", "files"],
            "description": "What to search: file contents or file names. Default 'content'.",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in. Default '.' (which is /workspace).",
        },
        "file_glob": {
            "type": "string",
            "description": ("Optional glob to filter which files are searched (content target)."),
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results to return. Default 50.",
            "minimum": 1,
        },
        "offset": {
            "type": "integer",
            "description": "Skip the first N results. Default 0.",
            "minimum": 0,
        },
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_only", "count"],
            "description": (
                "How to format results (content target): 'content' shows "
                "matching lines with line numbers, 'files_only' shows only "
                "the unique file paths, 'count' shows match counts per file."
            ),
        },
        "context": {
            "type": "integer",
            "description": "Lines of context around each match (content target).",
            "minimum": 0,
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


async def search_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the search tool. See module docstring for error-shape rules."""
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise SearchArgumentError("search tool requires a non-empty 'pattern' string")

    target = arguments.get("target", "content")
    if target not in ("content", "files"):
        raise SearchArgumentError("target must be 'content' or 'files'")

    path = arguments.get("path", ".")
    if not isinstance(path, str):
        raise SearchArgumentError("path must be a string")

    file_glob = arguments.get("file_glob")
    if file_glob is not None and not isinstance(file_glob, str):
        raise SearchArgumentError("file_glob must be a string")

    limit = arguments.get("limit", 50)
    if not isinstance(limit, int) or limit < 1:
        raise SearchArgumentError("limit must be a positive integer")

    offset = arguments.get("offset", 0)
    if not isinstance(offset, int) or offset < 0:
        raise SearchArgumentError("offset must be a non-negative integer")

    output_mode = arguments.get("output_mode", "content")
    if output_mode not in ("content", "files_only", "count"):
        raise SearchArgumentError("output_mode must be 'content', 'files_only', or 'count'")

    context = arguments.get("context", 0)
    if not isinstance(context, int) or context < 0:
        raise SearchArgumentError("context must be a non-negative integer")

    sess = await file_session.get_or_create(session_id)

    async with sess.lock:
        # Consecutive-search tracking. Use a distinct key shape from
        # read's to avoid cross-contaminating the counter.
        search_key = (
            "search",
            pattern,
            target,
            path,
            file_glob or "",
            limit,
            offset,
            output_mode,
            context,
        )
        # We piggyback on last_key / consecutive (which read also uses).
        # Mixing read and search resets the counter across tool types,
        # which is the right behavior -- read-then-search is progress,
        # not a loop.
        if sess.last_key == search_key:
            sess.consecutive += 1
        else:
            sess.last_key = search_key
            sess.consecutive = 1
        count = sess.consecutive

        if count >= 4:
            return {
                "error": (
                    f"BLOCKED: you have run this exact search {count} times in "
                    "a row. The results have NOT changed. STOP re-searching and "
                    "proceed with your task."
                ),
                "pattern": pattern,
            }

        result = await sess.file_ops.search(
            pattern,
            path=path,
            target=target,
            file_glob=file_glob,
            limit=limit,
            offset=offset,
            output_mode=output_mode,
            context=context,
        )
        result_dict = result.to_dict()

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "Use the information you already have."
            )

        # Pagination hint: if truncated, suggest the next offset.
        if result.truncated:
            next_offset = offset + limit
            result_dict["_next_offset_hint"] = (
                f"Results truncated. Use offset={next_offset} to see more."
            )

        return result_dict


def _register() -> None:
    registry.register(
        name="search",
        description=SEARCH_DESCRIPTION,
        parameters_schema=SEARCH_PARAMETERS_SCHEMA,
        handler=search_handler,
    )


_register()
