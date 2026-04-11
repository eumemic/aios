"""File operations module (vendored from hermes-agent).

Vendored from hermes-agent ``tools/file_operations.py`` at commit
34d06a9. See ``PATCHES.md`` for the full list of substantive changes.
Highlights:

- Every method in :class:`ShellFileOperations` that touches
  ``self._exec`` is now ``async def``. Pure helpers
  (``_is_likely_binary``, ``_add_line_numbers``, ``_escape_shell_arg``,
  ``_unified_diff``) stay sync.
- The ``_exec`` helper awaits on ``self.env.run_command`` and returns
  the shared :class:`aios.tools.adapters.ExecuteResult`. Hermes's local
  ``ExecuteResult`` dataclass has been dropped; there is now a single
  source of truth in ``aios.tools.adapters``.
- ``hermes_constants.get_hermes_home`` import removed. The one place
  that used it (the ``$HERMES_HOME/.env`` deny path) is dropped because
  aios has no equivalent home directory concept -- writes happen inside
  the sandbox container at ``/workspace``.
- ``_get_safe_write_root`` and the ``HERMES_WRITE_SAFE_ROOT`` env var
  are removed. The aios sandbox mounts only ``/workspace`` into the
  container; any "safe root" concept is enforced by the bind mount, not
  by a Python check.
- ``LINTERS``, ``LintResult``, ``_check_lint`` are all deleted.
  Auto-lint after patches was explicitly dropped per the Phase 4 locked
  decision (revisit in Phase 6 polish).
- ``IMAGE_EXTENSIONS`` and the image-handling branch of ``read_file``
  are removed. aios v1 doesn't ship a vision tool, so images are
  treated as binary and rejected by ``read_file`` via the
  binary-extension check.
- The ``__init__`` cwd heuristic (``cwd or getattr(env, 'cwd', ...) or
  os.getcwd()``) is collapsed to a simple default of ``/workspace``.
  Inside the container, that's the only cwd we ever want.
- Typing modernized to PEP 695 / builtin generics.

Provides file manipulation capabilities (read, write, patch, search)
that work through the sandbox's ``ContainerHandle.run_command`` via the
``SandboxTerminalEnv`` adapter.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aios.tools.adapters import ExecuteResult
from aios.vendor.hermes_files.binary_extensions import BINARY_EXTENSIONS

if TYPE_CHECKING:
    from aios.tools.adapters import SandboxTerminalEnv


# ─── Write-path deny list ─────────────────────────────────────────────────────
#
# Blocks writes to sensitive system/credential files. Defence in depth --
# the primary safety boundary is the sandbox bind mount: only ``/workspace``
# is writable from inside the container, period. These paths are unlikely
# to even exist inside the standard aios sandbox image, but the deny list
# stays in case a future deployment loosens the bind mount.

_HOME = str(Path.home())

WRITE_DENIED_PATHS: set[str] = {
    os.path.realpath(p)
    for p in [
        os.path.join(_HOME, ".ssh", "authorized_keys"),
        os.path.join(_HOME, ".ssh", "id_rsa"),
        os.path.join(_HOME, ".ssh", "id_ed25519"),
        os.path.join(_HOME, ".ssh", "config"),
        os.path.join(_HOME, ".bashrc"),
        os.path.join(_HOME, ".zshrc"),
        os.path.join(_HOME, ".profile"),
        os.path.join(_HOME, ".bash_profile"),
        os.path.join(_HOME, ".zprofile"),
        os.path.join(_HOME, ".netrc"),
        os.path.join(_HOME, ".pgpass"),
        os.path.join(_HOME, ".npmrc"),
        os.path.join(_HOME, ".pypirc"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
}

WRITE_DENIED_PREFIXES: list[str] = [
    os.path.realpath(p) + os.sep
    for p in [
        os.path.join(_HOME, ".ssh"),
        os.path.join(_HOME, ".aws"),
        os.path.join(_HOME, ".gnupg"),
        os.path.join(_HOME, ".kube"),
        "/etc/sudoers.d",
        "/etc/systemd",
        os.path.join(_HOME, ".docker"),
        os.path.join(_HOME, ".azure"),
        os.path.join(_HOME, ".config", "gh"),
    ]
]


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list. Pure path-string logic."""
    resolved = os.path.realpath(os.path.expanduser(str(path)))

    if resolved in WRITE_DENIED_PATHS:
        return True
    return any(resolved.startswith(prefix) for prefix in WRITE_DENIED_PREFIXES)


# ─── Result data classes ──────────────────────────────────────────────────────


@dataclass
class ReadResult:
    """Result from reading a file."""

    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    hint: str | None = None
    is_binary: bool = False
    error: str | None = None
    similar_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class WriteResult:
    """Result from writing a file."""

    bytes_written: int = 0
    dirs_created: bool = False
    error: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PatchResult:
    """Result from patching a file."""

    success: bool = False
    diff: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.diff:
            result["diff"] = self.diff
        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.files_created:
            result["files_created"] = self.files_created
        if self.files_deleted:
            result["files_deleted"] = self.files_deleted
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class SearchMatch:
    """A single search match."""

    path: str
    line_number: int
    content: str
    mtime: float = 0.0  # Modification time for sorting


@dataclass
class SearchResult:
    """Result from searching."""

    matches: list[SearchMatch] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"total_count": self.total_count}
        if self.matches:
            result["matches"] = [
                {"path": m.path, "line": m.line_number, "content": m.content} for m in self.matches
            ]
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
        if self.error:
            result["error"] = self.error
        return result


# ─── Abstract interface ───────────────────────────────────────────────────────


class FileOperations(ABC):
    """Abstract interface for file operations across terminal backends.

    Every public method is async -- the terminal backend is assumed to
    be async (which matches aios's :class:`ContainerHandle`).
    """

    @abstractmethod
    async def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        """Read a file with pagination support."""
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> WriteResult:
        """Write content to a file, creating directories as needed."""
        ...

    @abstractmethod
    async def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> PatchResult:
        """Replace text in a file using fuzzy matching."""
        ...

    @abstractmethod
    async def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""
        ...

    @abstractmethod
    async def search(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ) -> SearchResult:
        """Search for content or files."""
        ...


# ─── Shell-based implementation ───────────────────────────────────────────────

# Max limits for read operations
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 50 * 1024  # 50KB -- beyond this, reads still succeed but we warn


class ShellFileOperations(FileOperations):
    """File operations implemented via shell commands.

    Works with any async terminal backend exposing a
    ``run_command(command, *, cwd, timeout, stdin_data) -> ExecuteResult``
    method. In aios that backend is
    :class:`aios.tools.adapters.SandboxTerminalEnv`, which in turn
    delegates to :class:`aios.sandbox.container.ContainerHandle`.
    """

    def __init__(self, terminal_env: SandboxTerminalEnv, cwd: str = "/workspace") -> None:
        self.env = terminal_env
        self.cwd = cwd
        # Cache for command-availability checks. Survives across tool
        # calls within one FileToolSession, so `rg` detection runs once
        # per session.
        self._command_cache: dict[str, bool] = {}

    async def _exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,  # noqa: ASYNC109  # hermes API contract, not asyncio cancellation
        stdin_data: str | None = None,
    ) -> ExecuteResult:
        """Run a command via the terminal backend and return the result.

        Thin pass-through to ``self.env.run_command``. Exists as an
        indirection point so future changes (retry logic, logging) can
        hook in without touching every call site.
        """
        return await self.env.run_command(
            command,
            cwd=cwd or self.cwd,
            timeout=timeout,
            stdin_data=stdin_data,
        )

    async def _has_command(self, cmd: str) -> bool:
        """Check if a command exists in the environment (cached)."""
        if cmd not in self._command_cache:
            result = await self._exec(f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'")
            self._command_cache[cmd] = result.stdout.strip() == "yes"
        return self._command_cache[cmd]

    def _is_likely_binary(self, path: str, content_sample: str | None = None) -> bool:
        """Check if a file is likely binary.

        Uses extension check (fast) + content analysis (fallback). Pure
        -- no I/O.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True

        # Content analysis: >30% non-printable chars = binary
        if content_sample:
            non_printable = sum(
                1 for c in content_sample[:1000] if ord(c) < 32 and c not in "\n\r\t"
            )
            return non_printable / min(len(content_sample), 1000) > 0.30

        return False

    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Add line numbers to content in ``LINE_NUM|CONTENT`` format. Pure."""
        lines = content.split("\n")
        numbered: list[str] = []
        for i, line in enumerate(lines, start=start_line):
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "... [truncated]"
            numbered.append(f"{i:6d}|{line}")
        return "\n".join(numbered)

    async def _expand_path(self, path: str) -> str:
        """Expand shell-style paths like ``~`` and ``~user`` to absolute paths.

        Expansion runs inside the sandbox container, so ``~`` resolves
        to the container's ``$HOME`` (typically ``/root`` in the
        python:3.13-slim base), not the host user's home.
        """
        if not path:
            return path

        if path.startswith("~"):
            result = await self._exec("echo $HOME")
            if result.exit_code == 0 and result.stdout.strip():
                home = result.stdout.strip()
                if path == "~":
                    return home
                if path.startswith("~/"):
                    return home + path[1:]
                # ~username form -- validate before expanding to prevent
                # shell injection via paths like "~; rm -rf /".
                rest = path[1:]
                slash_idx = rest.find("/")
                username = rest[:slash_idx] if slash_idx >= 0 else rest
                if username and re.fullmatch(r"[a-zA-Z0-9._-]+", username):
                    expand_result = await self._exec(f"echo ~{username}")
                    if expand_result.exit_code == 0 and expand_result.stdout.strip():
                        user_home = expand_result.stdout.strip()
                        suffix = path[1 + len(username) :]
                        return user_home + suffix

        return path

    def _escape_shell_arg(self, arg: str) -> str:
        """Escape a string for safe use in shell commands. Pure."""
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """Generate unified diff between old and new content. Pure."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        return "".join(diff)

    # ─── READ ─────────────────────────────────────────────────────────────

    async def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        """Read a file with pagination, binary detection, and line numbers."""
        path = await self._expand_path(path)

        limit = min(limit, MAX_LINES)

        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = await self._exec(stat_cmd)

        if stat_result.exit_code != 0:
            return await self._suggest_similar_files(path)

        try:
            file_size = int(stat_result.stdout.strip())
        except ValueError:
            file_size = 0

        # Phase 4 drops the image branch -- aios has no vision tool.
        # Images are rejected as binary via the extension check below.

        sample_cmd = f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null"
        sample_result = await self._exec(sample_cmd)

        if self._is_likely_binary(path, sample_result.stdout):
            return ReadResult(
                is_binary=True,
                file_size=file_size,
                error=(
                    "Binary file - cannot display as text. "
                    "Use appropriate tools to handle this file type."
                ),
            )

        end_line = offset + limit - 1
        read_cmd = f"sed -n '{offset},{end_line}p' {self._escape_shell_arg(path)}"
        read_result = await self._exec(read_cmd)

        if read_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {read_result.stdout}")

        wc_cmd = f"wc -l < {self._escape_shell_arg(path)}"
        wc_result = await self._exec(wc_cmd)
        try:
            total_lines = int(wc_result.stdout.strip())
        except ValueError:
            total_lines = 0

        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = (
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)"
            )

        return ReadResult(
            content=self._add_line_numbers(read_result.stdout, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint,
        )

    async def _suggest_similar_files(self, path: str) -> ReadResult:
        """Suggest similar files when the requested file is not found."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)

        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -20"
        ls_result = await self._exec(ls_cmd)

        similar: list[str] = []
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            files = ls_result.stdout.strip().split("\n")
            for f in files:
                common = set(filename.lower()) & set(f.lower())
                if len(common) >= len(filename) * 0.5:
                    similar.append(os.path.join(dir_path, f))

        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar[:5],
        )

    # ─── WRITE ────────────────────────────────────────────────────────────

    async def write_file(self, path: str, content: str) -> WriteResult:
        """Write content to a file, creating parent directories as needed.

        Pipes content through stdin via the adapter's base64 wrapping to
        avoid OS ``ARG_MAX`` limits on large files. The content never
        appears in the shell command string -- only the file path does.
        """
        path = await self._expand_path(path)

        if _is_write_denied(path):
            return WriteResult(
                error=f"Write denied: '{path}' is a protected system/credential file."
            )

        parent = os.path.dirname(path)
        dirs_created = False

        if parent:
            mkdir_cmd = f"mkdir -p {self._escape_shell_arg(parent)}"
            mkdir_result = await self._exec(mkdir_cmd)
            if mkdir_result.exit_code == 0:
                dirs_created = True

        write_cmd = f"cat > {self._escape_shell_arg(path)}"
        write_result = await self._exec(write_cmd, stdin_data=content)

        if write_result.exit_code != 0:
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")

        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = await self._exec(stat_cmd)

        try:
            bytes_written = int(stat_result.stdout.strip())
        except ValueError:
            bytes_written = len(content.encode("utf-8"))

        return WriteResult(bytes_written=bytes_written, dirs_created=dirs_created)

    # ─── PATCH (replace mode) ─────────────────────────────────────────────

    async def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> PatchResult:
        """Replace text in a file using fuzzy matching."""
        from aios.vendor.hermes_files.fuzzy_match import fuzzy_find_and_replace

        path = await self._expand_path(path)

        if _is_write_denied(path):
            return PatchResult(
                error=f"Write denied: '{path}' is a protected system/credential file."
            )

        read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        read_result = await self._exec(read_cmd)

        if read_result.exit_code != 0:
            return PatchResult(error=f"Failed to read file: {path}")

        content = read_result.stdout

        new_content, match_count, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )

        if error:
            return PatchResult(error=error)

        if match_count == 0:
            return PatchResult(error=f"Could not find match for old_string in {path}")

        write_result = await self.write_file(path, new_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")

        diff = self._unified_diff(content, new_content, path)

        return PatchResult(success=True, diff=diff, files_modified=[path])

    async def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""
        from aios.vendor.hermes_files.patch_parser import (
            apply_v4a_operations,
            parse_v4a_patch,
        )

        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            return PatchResult(error=f"Failed to parse patch: {parse_error}")

        result = await apply_v4a_operations(operations, self)
        return result

    # ─── SEARCH ───────────────────────────────────────────────────────────

    async def search(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ) -> SearchResult:
        """Search for content or files."""
        path = await self._expand_path(path)

        check = await self._exec(
            f"test -e {self._escape_shell_arg(path)} && echo exists || echo not_found"
        )
        if "not_found" in check.stdout:
            return SearchResult(
                error=f"Path not found: {path}. Verify the path exists.",
                total_count=0,
            )

        if target == "files":
            return await self._search_files(pattern, path, limit, offset)
        return await self._search_content(
            pattern, path, file_glob, limit, offset, output_mode, context
        )

    async def _search_files(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name pattern (glob-like)."""
        if not pattern.startswith("**/") and "/" not in pattern:
            search_pattern = pattern
        else:
            search_pattern = pattern.split("/")[-1]

        if await self._has_command("rg"):
            return await self._search_files_rg(search_pattern, path, limit, offset)

        if not await self._has_command("find"):
            return SearchResult(
                error=(
                    "File search requires 'rg' (ripgrep) or 'find'. "
                    "Install ripgrep for best results: "
                    "https://github.com/BurntSushi/ripgrep#installation"
                )
            )

        hidden_exclude = "-not -path '*/.*'"

        cmd = (
            f"find {self._escape_shell_arg(path)} {hidden_exclude} -type f "
            f"-name {self._escape_shell_arg(search_pattern)} "
            f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn | tail -n +{offset + 1} "
            f"| head -n {limit}"
        )

        result = await self._exec(cmd, timeout=60)

        if not result.stdout.strip():
            # Fallback for BSD find (macOS) which lacks -printf.
            cmd_simple = (
                f"find {self._escape_shell_arg(path)} {hidden_exclude} -type f "
                f"-name {self._escape_shell_arg(search_pattern)} 2>/dev/null "
                f"| head -n {limit + offset} | tail -n +{offset + 1}"
            )
            result = await self._exec(cmd_simple, timeout=60)

        files: list[str] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].replace(".", "").isdigit():
                files.append(parts[1])
            else:
                files.append(line)

        return SearchResult(files=files, total_count=len(files))

    async def _search_files_rg(
        self, pattern: str, path: str, limit: int, offset: int
    ) -> SearchResult:
        """Search for files using ripgrep's ``--files`` mode."""
        if "/" not in pattern and not pattern.startswith("*"):
            glob_pattern = f"*{pattern}"
        else:
            glob_pattern = pattern

        fetch_limit = limit + offset
        cmd = (
            f"rg --files -g {self._escape_shell_arg(glob_pattern)} "
            f"{self._escape_shell_arg(path)} 2>/dev/null "
            f"| head -n {fetch_limit}"
        )
        result = await self._exec(cmd, timeout=60)

        all_files = [f for f in result.stdout.strip().split("\n") if f]
        page = all_files[offset : offset + limit]

        return SearchResult(
            files=page,
            total_count=len(all_files),
            truncated=len(all_files) >= fetch_limit,
        )

    async def _search_content(
        self,
        pattern: str,
        path: str,
        file_glob: str | None,
        limit: int,
        offset: int,
        output_mode: str,
        context: int,
    ) -> SearchResult:
        """Search for content inside files (grep-like)."""
        if await self._has_command("rg"):
            return await self._search_with_rg(
                pattern, path, file_glob, limit, offset, output_mode, context
            )
        if await self._has_command("grep"):
            return await self._search_with_grep(
                pattern, path, file_glob, limit, offset, output_mode, context
            )
        return SearchResult(
            error=(
                "Content search requires ripgrep (rg) or grep. "
                "Install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
            )
        )

    async def _search_with_rg(
        self,
        pattern: str,
        path: str,
        file_glob: str | None,
        limit: int,
        offset: int,
        output_mode: str,
        context: int,
    ) -> SearchResult:
        """Search using ripgrep."""
        cmd_parts: list[str] = ["rg", "--line-number", "--no-heading", "--with-filename"]

        if context > 0:
            cmd_parts.extend(["-C", str(context)])

        if file_glob:
            cmd_parts.extend(["--glob", self._escape_shell_arg(file_glob)])

        if output_mode == "files_only":
            cmd_parts.append("-l")
        elif output_mode == "count":
            cmd_parts.append("-c")

        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))

        # Fetch extra rows so we can report the true total before slicing.
        # Context mode adds "--" separator lines; grab generously.
        fetch_limit = limit + offset + 200 if context > 0 else limit + offset
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])

        cmd = " ".join(cmd_parts)
        result = await self._exec(cmd, timeout=60)

        # rg exit codes: 0=matches, 1=no matches, 2=error.
        if result.exit_code == 2 and not result.stdout.strip():
            error_msg = result.stderr.strip() or "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)

        if output_mode == "files_only":
            all_files = [f for f in result.stdout.strip().split("\n") if f]
            total = len(all_files)
            page = all_files[offset : offset + limit]
            return SearchResult(files=page, total_count=total)

        if output_mode == "count":
            counts: dict[str, int] = {}
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    parts = line.rsplit(":", 1)
                    if len(parts) == 2:
                        with contextlib.suppress(ValueError):
                            counts[parts[0]] = int(parts[1])
            return SearchResult(counts=counts, total_count=sum(counts.values()))

        # Content mode: parse match + optional context lines.
        match_re = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")
        ctx_re = re.compile(r"^([A-Za-z]:)?(.*?)-(\d+)-(.*)$")
        matches: list[SearchMatch] = []
        for line in result.stdout.strip().split("\n"):
            if not line or line == "--":
                continue

            m = match_re.match(line)
            if m:
                matches.append(
                    SearchMatch(
                        path=(m.group(1) or "") + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500],
                    )
                )
                continue

            if context > 0:
                m = ctx_re.match(line)
                if m:
                    matches.append(
                        SearchMatch(
                            path=(m.group(1) or "") + m.group(2),
                            line_number=int(m.group(3)),
                            content=m.group(4)[:500],
                        )
                    )

        total = len(matches)
        page_m = matches[offset : offset + limit]
        return SearchResult(
            matches=page_m,
            total_count=total,
            truncated=total > offset + limit,
        )

    async def _search_with_grep(
        self,
        pattern: str,
        path: str,
        file_glob: str | None,
        limit: int,
        offset: int,
        output_mode: str,
        context: int,
    ) -> SearchResult:
        """Fallback search using grep."""
        cmd_parts: list[str] = ["grep", "-rnH"]

        # Exclude hidden directories to match ripgrep's default behavior.
        cmd_parts.append("--exclude-dir='.*'")

        if context > 0:
            cmd_parts.extend(["-C", str(context)])

        if file_glob:
            cmd_parts.extend(["--include", self._escape_shell_arg(file_glob)])

        if output_mode == "files_only":
            cmd_parts.append("-l")
        elif output_mode == "count":
            cmd_parts.append("-c")

        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))

        fetch_limit = limit + offset + (200 if context > 0 else 0)
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])

        cmd = " ".join(cmd_parts)
        result = await self._exec(cmd, timeout=60)

        if result.exit_code == 2 and not result.stdout.strip():
            error_msg = result.stderr.strip() or "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)

        if output_mode == "files_only":
            all_files = [f for f in result.stdout.strip().split("\n") if f]
            total = len(all_files)
            page = all_files[offset : offset + limit]
            return SearchResult(files=page, total_count=total)

        if output_mode == "count":
            counts: dict[str, int] = {}
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    parts = line.rsplit(":", 1)
                    if len(parts) == 2:
                        with contextlib.suppress(ValueError):
                            counts[parts[0]] = int(parts[1])
            return SearchResult(counts=counts, total_count=sum(counts.values()))

        match_re = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")
        ctx_re = re.compile(r"^([A-Za-z]:)?(.*?)-(\d+)-(.*)$")
        matches: list[SearchMatch] = []
        for line in result.stdout.strip().split("\n"):
            if not line or line == "--":
                continue

            m = match_re.match(line)
            if m:
                matches.append(
                    SearchMatch(
                        path=(m.group(1) or "") + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500],
                    )
                )
                continue

            if context > 0:
                m = ctx_re.match(line)
                if m:
                    matches.append(
                        SearchMatch(
                            path=(m.group(1) or "") + m.group(2),
                            line_number=int(m.group(3)),
                            content=m.group(4)[:500],
                        )
                    )

        total = len(matches)
        page_m = matches[offset : offset + limit]
        return SearchResult(
            matches=page_m,
            total_count=total,
            truncated=total > offset + limit,
        )
