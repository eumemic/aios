# Vendored hermes-agent file tools — patch log

This directory holds the standard filesystem tool implementations lifted from
[hermes-agent](https://github.com/eumemic/hermes-agent). The code is
**not** maintained as a silent fork — every change we apply is listed below,
and re-vendoring is a manual process: copy the new files in, re-apply these
patches by hand, run the full test suite.

The user owns hermes-agent and has explicitly authorized this vendoring.

## Source

- Repo: `~/code/hermes-agent`
- Commit: `34d06a980244fb5bd88345765708cd7e36e3f118`
- Files vendored:
  - `tools/binary_extensions.py` → `src/aios/vendor/hermes_files/binary_extensions.py`
  - `tools/fuzzy_match.py` → `src/aios/vendor/hermes_files/fuzzy_match.py`
  - `tools/patch_parser.py` → `src/aios/vendor/hermes_files/patch_parser.py`
  - `tools/file_operations.py` → `src/aios/vendor/hermes_files/file_operations.py`

## Substantive changes per file

### `binary_extensions.py`

No behavioural changes. Style cleanups only:

- Added `from __future__ import annotations`.
- Added the aios vendoring header to the module docstring.
- Ruff-formatted (one extension per line).

### `fuzzy_match.py`

No behavioural changes. Style cleanups only:

- Added `from __future__ import annotations`.
- Replaced `from typing import Tuple, Optional, List, Callable` with
  builtin generics (`tuple`, `list`) and `Callable` from
  `collections.abc`.
- Signature types updated to PEP 695 style (`tuple[str, int, str | None]`
  etc.).
- Ruff-formatted.

### `patch_parser.py`

Substantive changes:

- Import paths rewritten: `tools.file_operations` →
  `aios.vendor.hermes_files.file_operations`; `tools.fuzzy_match` →
  `aios.vendor.hermes_files.fuzzy_match`.
- `fuzzy_find_and_replace` is now imported at module top level (was
  lazily imported inside `_apply_update` in hermes).
- `PatchResult` and `ShellFileOperations` are imported under
  `TYPE_CHECKING` only; `PatchResult` is still lazy-imported at runtime
  inside `apply_v4a_operations` to break the circular with
  `file_operations.py`.
- `apply_v4a_operations`, `_apply_add`, `_apply_delete`, `_apply_move`,
  `_apply_update` are all `async def` now, awaiting on
  `file_ops.read_file`, `file_ops.write_file`, and `file_ops._exec`.
- The auto-lint loop (hermes lines 267-272, inspecting `file_ops._check_lint`)
  is **deleted**. Phase 4 explicitly drops auto-lint (revisit in Phase 6).
- Type annotations modernized to PEP 695 / builtin generics.

### `file_operations.py`

The biggest diff. Substantive changes:

1. **Full async refactor.** Every method on `ShellFileOperations` that
   calls `self._exec` is now `async def`. Pure helpers
   (`_is_likely_binary`, `_add_line_numbers`, `_escape_shell_arg`,
   `_unified_diff`, `_is_write_denied`) stay synchronous. The abstract
   `FileOperations` ABC also has all methods declared `async`.
2. **Shared `ExecuteResult` dataclass.** Hermes's local `ExecuteResult`
   is deleted. The vendored code now imports
   `aios.tools.adapters.ExecuteResult`, which is a superset (adds
   `stderr`, `timed_out`, `truncated`). This makes
   `SandboxTerminalEnv` the single bridge between aios's
   `ContainerHandle` and the vendored code.
3. **`hermes_constants` import removed.** The only use was
   `get_hermes_home() / ".env"` in the write deny list; dropped because
   aios has no equivalent home directory concept.
4. **`_get_safe_write_root` and `HERMES_WRITE_SAFE_ROOT` deleted.** The
   aios sandbox bind mount (only `/workspace` is writable from inside
   the container) is the primary safety boundary; an env-var safety
   root is redundant and adds configuration surface.
5. **`LINTERS`, `LintResult`, `_check_lint` deleted.** Auto-lint after
   patches was explicitly dropped per the Phase 4 locked decision.
   Per-language linters would require adding compilers / runtimes to
   the sandbox image, which is out of scope for v1.
6. **`IMAGE_EXTENSIONS` and the image-handling branch of `read_file`
   deleted.** aios v1 doesn't ship a vision tool; images are treated
   as binary and rejected via the standard binary-extension check.
7. **`MAX_FILE_SIZE` kept as a module constant** but no longer used to
   branch behaviour inside `read_file` — the old code computed it and
   passed through anyway. Kept for clarity.
8. **Constructor signature simplified.** Hermes's
   `__init__(self, terminal_env, cwd=None)` had a five-level heuristic
   fallback chain for `cwd` (`cwd` arg → `terminal_env.cwd` →
   `terminal_env.config.cwd` → `"/"`). aios always runs inside a
   sandbox container with a fixed bind mount, so the default is now
   just `"/workspace"`.
9. **`patch_replace`'s in-method lazy import of `fuzzy_find_and_replace`
   kept** — it's already cheap and avoids loading `fuzzy_match` at
   module import time in scripts that only want `PatchResult`
   definitions.
10. **Typing modernized to PEP 695** and all `Optional[X]` replaced
    with `X | None`.
11. **`contextlib.suppress(ValueError)`** replaces `try/except ValueError: pass`
    in the rg/grep count-mode parsing loops (ruff SIM105).
12. **`any(...)` instead of `for ... return True`** in `_is_write_denied`
    (ruff SIM110).
13. **`# noqa: ASYNC109`** on `_exec`'s `timeout` parameter because
    ruff flags async functions with a `timeout` argument. Here the
    parameter is part of the hermes API contract and gets forwarded
    through the adapter to `docker exec`, not used for asyncio
    cancellation.

## Re-vendoring procedure

There is no automated sync. When hermes adds features worth pulling:

1. `git -C ~/code/hermes-agent log --oneline --since="<date>" -- tools/file_operations.py tools/fuzzy_match.py tools/patch_parser.py tools/binary_extensions.py` to see what changed.
2. Re-copy the upstream files.
3. Re-apply every patch listed in this document, in order.
4. Update the "Source" section with the new commit SHA.
5. Run the full test suite: `uv run pytest tests/unit -q && uv run mypy src`.
6. Run the live verification micro-prompts against the playground (see
   Phase 4 section of `aios_v1_progress.md` in memory for the exact
   prompts).

If hermes makes sweeping changes — e.g. a second major refactor of
`file_operations.py` — it may be cheaper to re-vendor from scratch than
to re-apply patches. In that case, delete the files in this directory,
copy the new versions, and walk through this document to re-apply the
necessary changes by hand.

## What we did NOT vendor from hermes

- `tools/file_tools.py` (836 lines) — the LLM-facing tool wrappers.
  aios has its own tool-handler layer under `src/aios/tools/` that
  builds on the vendored primitives; re-implementing the handlers
  from scratch lets us preserve aios conventions (async-native,
  structlog, `AiosError`, the tool registry singleton pattern).
- `tools/registry.py` — aios already has its own
  `src/aios/tools/registry.py` from Phase 3. Hermes's registry has
  extra features (dispatch, availability checks) we don't need in v1.
- `tests/tools/test_file_tools.py` — split into per-handler test files
  in aios under `tests/unit/test_file_{read,write,edit,search}_handler.py`
  matching aios's one-test-file-per-handler discipline.
- Anything under `tests/tools/` that touches hermes internals directly
  (e.g. `test_file_tools_live.py`).
