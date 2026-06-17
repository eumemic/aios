"""Axis segregation is the de-Goodhart of the residue gauge (#1328, AC4/AC5).

axis-1 (machine-found mechanical waste) and axis-2 (class migration) must be
queried and rendered SEPARATELY by construction: no aggregate may combine
``axis=1`` and ``axis=2`` rows into a single number, because a rising axis-1 is a
false safety signal that is FORBIDDEN as an autonomy-increment justification.

This test greps the residue query/render modules and FAILS if it finds:
- a ``SUM``/``count`` aggregate that is not scoped by an ``axis`` filter, or
- a query that aggregates ``residue_events`` without an ``axis = `` predicate.

Pure-Python: reads the module source off disk. No DB.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "aios"
_QUERY = _SRC / "db" / "queries" / "residue.py"
_RENDER = _SRC / "workflows" / "residue_render.py"

# A SQL statement that aggregates residue_events. We split the query module into
# individual SQL string literals and check each aggregate touches residue_events
# under an axis filter.
_SQL_RE = re.compile(r'"""(.*?)"""', re.DOTALL)


def _sql_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _SQL_RE.finditer(text)]


def test_every_residue_aggregate_is_axis_scoped() -> None:
    text = _QUERY.read_text()
    aggregated = []
    for block in _sql_blocks(text):
        low = block.lower()
        if "residue_events" not in low:
            continue
        is_aggregate = ("count(" in low) or ("sum(" in low) or ("group by" in low)
        if not is_aggregate:
            continue
        # Every aggregate over residue_events MUST filter to a single axis.
        assert "axis = $" in low or "axis = " in low, (
            "an aggregate over residue_events is NOT axis-scoped — a cross-axis "
            "aggregate would combine axis-1 and axis-2 (forbidden, #1328):\n%s" % block
        )
        aggregated.append(block)
    # Sanity: there ARE aggregates (the test is actually exercising something).
    assert aggregated, "expected at least one axis-scoped aggregate in residue.py"


def test_no_sum_across_axes_anywhere() -> None:
    """No SQL block may GROUP BY axis (which would put both axes in one result
    set to be summed), and no aggregate may select both axes."""
    for path in (_QUERY, _RENDER):
        text = path.read_text()
        for block in _sql_blocks(text):
            low = block.lower()
            if "residue_events" not in low:
                continue
            # GROUP BY axis would surface both axes in one aggregate result —
            # exactly the cross-axis combination the de-Goodhart forbids.
            assert "group by axis" not in low, (
                "GROUP BY axis combines both axes into one aggregate (forbidden, "
                "#1328): %s in %s" % (block, path.name)
            )


def test_render_keeps_axes_in_separate_data_structures() -> None:
    """The render takes two distinct AxisView inputs — there is no field that
    fuses the two axes, so the data shape itself enforces segregation."""
    from aios.workflows.residue_render import AxisView, render_ledger

    # render_ledger's signature takes axis1 and axis2 separately.
    import inspect

    params = inspect.signature(render_ledger).parameters
    assert "axis1" in params and "axis2" in params
    # AxisView carries only single-axis fields (no combined/total field).
    assert set(AxisView._fields) == {
        "denominator",
        "found_by_finder",
        "other_count",
        "cannot_determine_count",
    }
