"""Structural guard: ``open_tool_call_floor_seq`` is written in exactly ONE
place (#1746).

The floor is a PROVEN lower bound on the oldest open tool_call's ``seq`` only
because it is advanced ``GREATEST``-only, and ONLY by the ghost sweep's
reconciliation (:func:`aios.harness.sweep._advance_open_tool_call_floor`) —
never by the write path (``append_event``), never by a manual/backfill
``UPDATE`` (the #925 manual-UPDATE-zombie class this guards against), and
never by a second code path that some future change might add. A second
writer could stamp an unsafe (non-lower-bound) value and silently reintroduce
the exact permanent-wedge class the sweep-maintained design forecloses.

This test greps the whole ``src/aios`` tree for any SQL statement that writes
the column and asserts there is exactly one such site — the one inside
``sweep.py``'s ``_ADVANCE_OPEN_TOOL_CALL_FLOOR_SQL``. It also re-affirms the
sibling guard for ``open_tool_call_count`` (#841's single-increment /
single-decrement invariant), which remains load-bearing for invariant 0 (the
counter being a sound-but-not-exact lower-bound signal is exactly what makes
the sweep-derived floor trustworthy in turn) — pinning BOTH counters' writer
sets in one place keeps a future "helpful" backfill script from adding an
unguarded third site to either.

Pure source-text scan — no DB, no Docker, no imports of the scanned modules
(so a syntax-breaking edit to the target file still fails this test loudly
rather than raising at import time in an unrelated caller).
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "aios"

# Matches a SQL assignment of the column: ``<column> = <expr>`` where the
# single ``=`` is NOT part of a ``==`` comparison (which would match a prose
# aside like "the ``open_tool_call_count == 0`` transition"). Non-comment
# lines only (enforced by ``_matching_lines`` below) — this is enough to
# distinguish an actual SQL ``SET``/assignment from a docstring mention: no
# non-comment line in the tree writes prose containing "<column> = " other
# than an actual write.
_FLOOR_WRITE_RE = re.compile(r"open_tool_call_floor_seq\s*=(?!=)")
_COUNT_WRITE_RE = re.compile(r"\bopen_tool_call_count\s*=(?!=)")


def _matching_lines(pattern: re.Pattern[str]) -> list[tuple[Path, int, str]]:
    """Lines that write the column via actual SQL syntax.

    Requires ``SET <column> =`` (case-insensitive, whitespace-tolerant) so a
    prose comment merely discussing the column (e.g. "the
    ``open_tool_call_count == 0`` transition") is not mistaken for a write
    site — only a literal ``SET`` assignment counts.
    """
    hits: list[tuple[Path, int, str]] = []
    for path in _SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                hits.append((path, lineno, stripped))
    return hits


def test_open_tool_call_floor_seq_has_exactly_one_writer() -> None:
    """The floor column is assigned in exactly one source line: the sweep's
    ``GREATEST``-only advance. A second hit means a new writer landed without
    updating this guard — stop and re-derive its safety before merging."""
    hits = _matching_lines(_FLOOR_WRITE_RE)
    assert len(hits) == 1, (
        f"expected exactly ONE write site for open_tool_call_floor_seq, found "
        f"{len(hits)}: {hits}. The floor is a proven lower bound ONLY because "
        f"it has a single, GREATEST-only writer (the ghost sweep's "
        f"reconciliation, #1746) — a second writer can stamp an unsafe value "
        f"and silently reintroduce the permanent-wedge class this design "
        f"forecloses."
    )
    path, _lineno, line = hits[0]
    assert path.name == "sweep.py", (
        f"open_tool_call_floor_seq's sole writer moved out of sweep.py to "
        f"{path} — the floor must be maintained by the ghost sweep's "
        f"reconciliation, not the write path (append_event) or any other "
        f"call site (#1746's rejected write-path-stamp design)."
    )
    assert "GREATEST" in line, (
        f"open_tool_call_floor_seq's writer must be GREATEST-only (monotonic, "
        f"race-safe advance-only) — got: {line!r}"
    )


def test_open_tool_call_count_writer_set_is_unchanged() -> None:
    """Sibling guard: ``open_tool_call_count`` keeps its exactly-two writers
    (the id-blind increment in ``append_event`` and the compensating
    decrement in ``decrement_open_tool_call_count``, #841) — both GREATEST-
    clamped. This invariant is load-bearing for #1746: the floor's soundness
    argument (invariant 0) rests on the counter's ``count > 0 ⇒ genuinely
    open`` guarantee, which in turn rests on there being no unguarded third
    writer of the counter."""
    hits = _matching_lines(_COUNT_WRITE_RE)
    assert len(hits) == 2, (
        f"expected exactly TWO write sites for open_tool_call_count (the "
        f"append_event increment and the decrement_open_tool_call_count "
        f"compensation, #841), found {len(hits)}: {hits}. A new writer changes "
        f"the counter's soundness argument that #1746's floor relies on."
    )
    filenames = {path.name for path, _lineno, _line in hits}
    assert filenames == {"events.py", "sessions.py"}, (
        f"open_tool_call_count writers moved to unexpected files: {filenames}"
    )
    for _path, _lineno, line in hits:
        assert "GREATEST" in line, (
            f"open_tool_call_count writer must stay GREATEST-clamped — got: {line!r}"
        )
