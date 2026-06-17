"""Render the detection-residue gauge to ``found-by-finder-ledger.md`` (#1328).

``found-by-finder-ledger.md`` stops being a hand-appended markdown file and
becomes the RENDERED QUERY of the ``residue_events`` table. This module holds the
PURE render function (deterministic over already-fetched data: no DB, no clock,
no I/O) so the whole render is verifiable on fixtures with no live model. The
deterministic ``render-ledger`` workflow (a ``CronSource → WorkflowAction``) does
the I/O — pulls each axis SEPARATELY and the two denominators — then calls this.

THE DE-GOODHART, RENDERED INTO THE INSTRUMENT:

- axis-1 (machine-found mechanical waste) and axis-2 (class migration) appear
  under TWO distinct ``##`` headers, each with its OWN denominator, and are
  visually un-summable.
- A hard-coded BANNER forbids axis-summing AND forbids a rising axis-1 as an
  autonomy-increment justification — so a reader cannot accidentally combine the
  two axes.
- The ``other``-bucket count and the ``cannot-determine`` count are prominent
  lines (the load-bearing growth alarm + the fail-loud absence), never buried.
- A merged-PR denominator that came back ``cannot-determine`` (a truncated GitHub
  list read, #1323) renders ``cannot-determine`` for axis-2 — NEVER an implicit
  short count.
- The historical hand-written ledger rows are appended below the live query as a
  ``## Pre-instrument baseline`` section (never-delete).
"""

from __future__ import annotations

from typing import NamedTuple

from aios.db.queries.residue import FINDERS

# The de-Goodhart banner, rendered into the instrument itself. The exact strings
# are asserted by a render test, so a future edit that weakens the prohibition is
# caught.
BANNER = (
    "> **axis-1 (machine-found mechanical waste) and axis-2 (class migration) are "
    "NEVER summed; a rising axis-1 is FORBIDDEN as justification for any autonomy "
    "increment.** axis-1 measures coverage of named, mechanically-legible classes "
    "only; novel-class first-detection (axis-2) is structurally lagging, so a "
    "rising axis-1 is a false safety signal. The chairman's irreducible residue is "
    "MEASURED and BOUNDED (the `other`-bucket growth alarm below), never claimed "
    "eliminated."
)

AXIS1_HEADER = "## axis-1 — machine-found mechanical waste (found-by-finder)"
AXIS2_HEADER = "## axis-2 — class migration (found-by-finder)"
BASELINE_HEADER = "## Pre-instrument baseline"


class AxisView(NamedTuple):
    """The fully-fetched, axis-SCOPED inputs to the render for ONE axis.

    ``denominator`` is the uncorrelated universe (terminal-run count for axis-1;
    merged-PR count for axis-2), or ``None`` when that axis's denominator came
    back ``cannot-determine``. ``found_by_finder`` is ``{finder: count}`` for
    THIS axis only. ``other_count`` / ``cannot_determine_count`` are the
    alarm/fail-loud bucket sizes for THIS axis. There is no field that combines
    the two axes — the segregation is structural in the data shape itself."""

    denominator: int | None
    found_by_finder: dict[str, int]
    other_count: int
    cannot_determine_count: int


def _denominator_cell(denominator: int | None) -> str:
    return "`cannot-determine`" if denominator is None else str(denominator)


def _ratio_cell(found: int, denominator: int | None) -> str:
    if denominator is None:
        return "`cannot-determine`"
    if denominator == 0:
        return "n/a (0 in denominator)"
    return "%.1f%%" % (100.0 * found / denominator)


def _axis_section(header: str, denominator_label: str, view: AxisView) -> str:
    lines = [header, ""]
    lines.append("Denominator (%s): **%s**" % (denominator_label, _denominator_cell(view.denominator)))
    lines.append("")
    lines.append("| finder | count | of denominator |")
    lines.append("| --- | --- | --- |")
    total_found = 0
    for finder in FINDERS:
        n = view.found_by_finder.get(finder, 0)
        total_found += n
        lines.append("| %s | %d | %s |" % (finder, n, _ratio_cell(n, view.denominator)))
    lines.append("")
    # The load-bearing growth alarm + the fail-loud absence line — prominent, not
    # buried in the table.
    lines.append("- **`other`-bucket (chairman irreducible residue — GROWTH ALARM):** %d"
                 % view.other_count)
    lines.append("- **`cannot-determine` (fail-loud; null telemetry, NOT counted clean):** %d"
                 % view.cannot_determine_count)
    return "\n".join(lines)


def render_ledger(
    *,
    axis1: AxisView,
    axis2: AxisView,
    baseline_markdown: str = "",
) -> str:
    """Render the full ``found-by-finder-ledger.md`` body.

    The two axes are rendered under two distinct ``##`` headers with two distinct
    denominators; the banner sits at the top forbidding any sum. ``baseline_markdown``
    is the historical hand-written ledger, appended verbatim below the live query
    (never-delete). Pure: deterministic over its inputs."""
    parts = [
        "# Found-by-finder ledger",
        "",
        "_This file is the RENDERED QUERY of the `residue_events` table (#1328). "
        "Do not hand-edit the live sections below — append a `residue_events` row "
        "instead. The `Pre-instrument baseline` section preserves the historical "
        "hand-written rows._",
        "",
        BANNER,
        "",
        _axis_section(AXIS1_HEADER, "terminal wf_runs over window", axis1),
        "",
        _axis_section(AXIS2_HEADER, "merged PRs over window", axis2),
    ]
    if baseline_markdown.strip():
        parts.append("")
        parts.append(BASELINE_HEADER)
        parts.append("")
        parts.append(baseline_markdown.strip())
    return "\n".join(parts) + "\n"
