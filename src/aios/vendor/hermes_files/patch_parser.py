"""V4A patch format parser (vendored from hermes-agent).

Vendored from hermes-agent ``tools/patch_parser.py`` at commit 34d06a9.
See ``PATCHES.md`` for the list of substantive changes:

- Import paths rewritten from ``tools.*`` to
  ``aios.vendor.hermes_files.*``.
- ``apply_v4a_operations`` and its ``_apply_*`` helpers are now
  ``async def``, awaiting on ``file_ops.read_file``,
  ``file_ops.write_file``, and ``file_ops._exec`` — matching the full
  async refactor of ``ShellFileOperations``.
- The auto-lint loop after applying operations (hermes lines 267-272)
  has been deleted per Phase 4's "skip auto-lint" decision.
- Typing modernized to PEP 695 / builtin generics.

The circular import between this module and ``file_operations.py`` (the
latter defines ``PatchResult`` which this module constructs; this module
defines functions that ``file_operations.patch_v4a`` calls) is resolved
by keeping the ``PatchResult`` import lazy inside ``apply_v4a_operations``
and using ``TYPE_CHECKING`` for the type annotation only.

V4A Format::

    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional context hint @@
     context line (space prefix)
    -removed line (minus prefix)
    +added line (plus prefix)
    *** Add File: path/to/new.py
    +new file content
    +line 2
    *** Delete File: path/to/old.py
    *** Move File: old/path.py -> new/path.py
    *** End Patch
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from aios.vendor.hermes_files.fuzzy_match import fuzzy_find_and_replace

if TYPE_CHECKING:
    from aios.vendor.hermes_files.file_operations import PatchResult, ShellFileOperations


class OperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class HunkLine:
    """A single line in a patch hunk."""

    prefix: str  # ' ', '-', or '+'
    content: str


@dataclass
class Hunk:
    """A group of changes within a file."""

    context_hint: str | None = None
    lines: list[HunkLine] = field(default_factory=list)


@dataclass
class PatchOperation:
    """A single operation in a V4A patch."""

    operation: OperationType
    file_path: str
    new_path: str | None = None  # For move operations
    hunks: list[Hunk] = field(default_factory=list)
    content: str | None = None  # For add file operations


def parse_v4a_patch(patch_content: str) -> tuple[list[PatchOperation], str | None]:
    """Parse a V4A format patch.

    Pure — no I/O, stays sync.

    Returns:
        ``(operations, error_message)``. On success,
        ``(list_of_operations, None)``. On failure, ``([], error_description)``.
    """
    lines = patch_content.split("\n")
    operations: list[PatchOperation] = []

    start_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        if "*** Begin Patch" in line or "***Begin Patch" in line:
            start_idx = i
        elif "*** End Patch" in line or "***End Patch" in line:
            end_idx = i
            break

    if start_idx is None:
        start_idx = -1

    if end_idx is None:
        end_idx = len(lines)

    i = start_idx + 1
    current_op: PatchOperation | None = None
    current_hunk: Hunk | None = None

    while i < end_idx:
        line = lines[i]

        update_match = re.match(r"\*\*\*\s*Update\s+File:\s*(.+)", line)
        add_match = re.match(r"\*\*\*\s*Add\s+File:\s*(.+)", line)
        delete_match = re.match(r"\*\*\*\s*Delete\s+File:\s*(.+)", line)
        move_match = re.match(r"\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)", line)

        if update_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)

            current_op = PatchOperation(
                operation=OperationType.UPDATE,
                file_path=update_match.group(1).strip(),
            )
            current_hunk = None

        elif add_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)

            current_op = PatchOperation(
                operation=OperationType.ADD,
                file_path=add_match.group(1).strip(),
            )
            current_hunk = Hunk()

        elif delete_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)

            current_op = PatchOperation(
                operation=OperationType.DELETE,
                file_path=delete_match.group(1).strip(),
            )
            operations.append(current_op)
            current_op = None
            current_hunk = None

        elif move_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)

            current_op = PatchOperation(
                operation=OperationType.MOVE,
                file_path=move_match.group(1).strip(),
                new_path=move_match.group(2).strip(),
            )
            operations.append(current_op)
            current_op = None
            current_hunk = None

        elif line.startswith("@@"):
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)

                hint_match = re.match(r"@@\s*(.+?)\s*@@", line)
                hint = hint_match.group(1) if hint_match else None
                current_hunk = Hunk(context_hint=hint)

        elif current_op and line:
            if current_hunk is None:
                current_hunk = Hunk()

            if line.startswith("+"):
                current_hunk.lines.append(HunkLine("+", line[1:]))
            elif line.startswith("-"):
                current_hunk.lines.append(HunkLine("-", line[1:]))
            elif line.startswith(" "):
                current_hunk.lines.append(HunkLine(" ", line[1:]))
            elif line.startswith("\\"):
                # "\ No newline at end of file" marker - skip
                pass
            else:
                # Treat as context line (implicit space prefix)
                current_hunk.lines.append(HunkLine(" ", line))

        i += 1

    if current_op:
        if current_hunk and current_hunk.lines:
            current_op.hunks.append(current_hunk)
        operations.append(current_op)

    return operations, None


async def apply_v4a_operations(
    operations: list[PatchOperation],
    file_ops: ShellFileOperations,
) -> PatchResult:
    """Apply V4A patch operations using a file operations interface.

    Async because every ``_apply_*`` helper awaits on ``file_ops``.

    Returns:
        ``PatchResult`` with diffs and errors for every operation.
    """
    # Lazy import to resolve the circular: file_operations.py imports
    # this module to access parse_v4a_patch / apply_v4a_operations from
    # patch_v4a, and this module needs PatchResult to construct results.
    from aios.vendor.hermes_files.file_operations import PatchResult

    files_modified: list[str] = []
    files_created: list[str] = []
    files_deleted: list[str] = []
    all_diffs: list[str] = []
    errors: list[str] = []

    for op in operations:
        try:
            if op.operation == OperationType.ADD:
                ok, diff_or_err = await _apply_add(op, file_ops)
                if ok:
                    files_created.append(op.file_path)
                    all_diffs.append(diff_or_err)
                else:
                    errors.append(f"Failed to add {op.file_path}: {diff_or_err}")

            elif op.operation == OperationType.DELETE:
                ok, diff_or_err = await _apply_delete(op, file_ops)
                if ok:
                    files_deleted.append(op.file_path)
                    all_diffs.append(diff_or_err)
                else:
                    errors.append(f"Failed to delete {op.file_path}: {diff_or_err}")

            elif op.operation == OperationType.MOVE:
                ok, diff_or_err = await _apply_move(op, file_ops)
                if ok:
                    files_modified.append(f"{op.file_path} -> {op.new_path}")
                    all_diffs.append(diff_or_err)
                else:
                    errors.append(f"Failed to move {op.file_path}: {diff_or_err}")

            elif op.operation == OperationType.UPDATE:
                ok, diff_or_err = await _apply_update(op, file_ops)
                if ok:
                    files_modified.append(op.file_path)
                    all_diffs.append(diff_or_err)
                else:
                    errors.append(f"Failed to update {op.file_path}: {diff_or_err}")

        except Exception as e:
            errors.append(f"Error processing {op.file_path}: {e!s}")

    # Phase 4 drops hermes's auto-lint loop — see PATCHES.md.

    combined_diff = "\n".join(all_diffs)

    if errors:
        return PatchResult(
            success=False,
            diff=combined_diff,
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            error="; ".join(errors),
        )

    return PatchResult(
        success=True,
        diff=combined_diff,
        files_modified=files_modified,
        files_created=files_created,
        files_deleted=files_deleted,
    )


async def _apply_add(op: PatchOperation, file_ops: ShellFileOperations) -> tuple[bool, str]:
    """Apply an add-file operation."""
    content_lines: list[str] = []
    for hunk in op.hunks:
        for line in hunk.lines:
            if line.prefix == "+":
                content_lines.append(line.content)

    content = "\n".join(content_lines)

    result = await file_ops.write_file(op.file_path, content)
    if result.error:
        return False, result.error

    diff = f"--- /dev/null\n+++ b/{op.file_path}\n"
    diff += "\n".join(f"+{line}" for line in content_lines)

    return True, diff


async def _apply_delete(op: PatchOperation, file_ops: ShellFileOperations) -> tuple[bool, str]:
    """Apply a delete-file operation."""
    read_result = await file_ops.read_file(op.file_path)

    if read_result.error and "not found" in read_result.error.lower():
        return True, f"# {op.file_path} already deleted or doesn't exist"

    rm_result = await file_ops._exec(f"rm -f {file_ops._escape_shell_arg(op.file_path)}")

    if rm_result.exit_code != 0:
        return False, rm_result.stdout

    diff = f"--- a/{op.file_path}\n+++ /dev/null\n# File deleted"
    return True, diff


async def _apply_move(op: PatchOperation, file_ops: ShellFileOperations) -> tuple[bool, str]:
    """Apply a move-file operation."""
    assert op.new_path is not None, "move operation requires new_path"
    mv_result = await file_ops._exec(
        f"mv {file_ops._escape_shell_arg(op.file_path)} {file_ops._escape_shell_arg(op.new_path)}"
    )

    if mv_result.exit_code != 0:
        return False, mv_result.stdout

    diff = f"# Moved: {op.file_path} -> {op.new_path}"
    return True, diff


async def _apply_update(op: PatchOperation, file_ops: ShellFileOperations) -> tuple[bool, str]:
    """Apply an update-file operation."""
    read_result = await file_ops.read_file(op.file_path, limit=10000)

    if read_result.error:
        return False, f"Cannot read file: {read_result.error}"

    # Parse content (strip line numbers added by read_file).
    current_lines: list[str] = []
    for line in read_result.content.split("\n"):
        if re.match(r"^\s*\d+\|", line):
            # Line format: "    123|content"
            parts = line.split("|", 1)
            if len(parts) == 2:
                current_lines.append(parts[1])
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    current_content = "\n".join(current_lines)

    new_content = current_content

    for hunk in op.hunks:
        search_lines: list[str] = []
        replace_lines: list[str] = []

        for hunk_line in hunk.lines:
            if hunk_line.prefix == " ":
                search_lines.append(hunk_line.content)
                replace_lines.append(hunk_line.content)
            elif hunk_line.prefix == "-":
                search_lines.append(hunk_line.content)
            elif hunk_line.prefix == "+":
                replace_lines.append(hunk_line.content)

        if search_lines:
            search_pattern = "\n".join(search_lines)
            replacement = "\n".join(replace_lines)

            new_content, count, error = fuzzy_find_and_replace(
                new_content, search_pattern, replacement, replace_all=False
            )

            if error and count == 0:
                # Retry with context hint window if one was provided.
                if hunk.context_hint:
                    hint_pos = new_content.find(hunk.context_hint)
                    if hint_pos != -1:
                        window_start = max(0, hint_pos - 500)
                        window_end = min(len(new_content), hint_pos + 2000)
                        window = new_content[window_start:window_end]

                        window_new, count, error = fuzzy_find_and_replace(
                            window, search_pattern, replacement, replace_all=False
                        )

                        if count > 0:
                            new_content = (
                                new_content[:window_start] + window_new + new_content[window_end:]
                            )
                            error = None

                if error:
                    return False, f"Could not apply hunk: {error}"
        else:
            # Addition-only hunk (no context or removed lines).
            insert_text = "\n".join(replace_lines)
            if hunk.context_hint:
                hint_pos = new_content.find(hunk.context_hint)
                if hint_pos != -1:
                    eol = new_content.find("\n", hint_pos)
                    if eol != -1:
                        new_content = (
                            new_content[: eol + 1] + insert_text + "\n" + new_content[eol + 1 :]
                        )
                    else:
                        new_content = new_content + "\n" + insert_text
                else:
                    new_content = new_content.rstrip("\n") + "\n" + insert_text + "\n"
            else:
                new_content = new_content.rstrip("\n") + "\n" + insert_text + "\n"

    write_result = await file_ops.write_file(op.file_path, new_content)
    if write_result.error:
        return False, write_result.error

    import difflib

    diff_lines = difflib.unified_diff(
        current_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{op.file_path}",
        tofile=f"b/{op.file_path}",
    )
    diff = "".join(diff_lines)

    return True, diff
