"""Render tests for the found-by-finder ledger (#1328, AC5/AC9/AC11).

The render is a PURE function over already-fetched, axis-scoped data, so the whole
markdown is verifiable here with no DB and no live model.
"""

from __future__ import annotations

from aios.workflows.residue_render import (
    AXIS1_HEADER,
    AXIS2_HEADER,
    BANNER,
    BASELINE_HEADER,
    AxisView,
    render_ledger,
)


def _axis(denominator, found=None, other=0, cd=0):
    return AxisView(
        denominator=denominator,
        found_by_finder=found or {},
        other_count=other,
        cannot_determine_count=cd,
    )


def test_banner_forbids_summing_and_autonomy_increment() -> None:
    md = render_ledger(axis1=_axis(10), axis2=_axis(5))
    assert BANNER in md
    assert "NEVER summed" in md
    assert "FORBIDDEN as justification for any autonomy increment" in md


def test_two_distinct_axis_headers_with_two_distinct_denominators() -> None:
    md = render_ledger(
        axis1=_axis(10, {"internal-armed-check": 3}),
        axis2=_axis(5, {"chairman": 2}),
    )
    assert AXIS1_HEADER in md
    assert AXIS2_HEADER in md
    # Two distinct denominators, each under its own header.
    assert "terminal wf_runs over window): **10**" in md
    assert "merged PRs over window): **5**" in md
    # The headers are distinct ## blocks.
    assert md.count("\n## ") >= 2


def test_other_bucket_and_cannot_determine_lines_are_prominent() -> None:
    md = render_ledger(
        axis1=_axis(10, other=0, cd=1),
        axis2=_axis(5, {"chairman": 1}, other=1, cd=0),
    )
    # The load-bearing growth alarm + fail-loud absence — both surface.
    assert "`other`-bucket (chairman irreducible residue — GROWTH ALARM):** 1" in md
    assert "`cannot-determine` (fail-loud; null telemetry, NOT counted clean):** 1" in md


def test_cannot_determine_denominator_renders_cannot_determine_not_a_short_count() -> None:
    # axis-2 merged-PR read was truncated → denominator is None → the render shows
    # cannot-determine, NEVER an implicit short count (AC9, the look-green guard).
    md = render_ledger(
        axis1=_axis(10, {"internal-armed-check": 2}),
        axis2=_axis(None, {"chairman": 3}),
    )
    assert "merged PRs over window): **`cannot-determine`**" in md
    # The finder ratio cells for axis-2 also degrade to cannot-determine, never a
    # fabricated percentage against an unknown denominator.
    assert md.count("`cannot-determine`") >= 2


def test_baseline_section_is_appended_below_live_query() -> None:
    md = render_ledger(
        axis1=_axis(1),
        axis2=_axis(1),
        baseline_markdown="| 2025-01-01 | chairman | found a thing |",
    )
    assert BASELINE_HEADER in md
    # The baseline comes AFTER both live axis sections (never-delete, appended below).
    assert md.index(AXIS1_HEADER) < md.index(BASELINE_HEADER)
    assert md.index(AXIS2_HEADER) < md.index(BASELINE_HEADER)
    assert "found a thing" in md


def test_zero_denominator_does_not_divide_by_zero() -> None:
    md = render_ledger(axis1=_axis(0, {"internal-armed-check": 0}), axis2=_axis(0))
    assert "n/a (0 in denominator)" in md
