"""Test-side utilities shared across tiers.

Kept at the top level of ``tests/`` (not inside ``unit/`` or ``e2e/``) so
either tier can import it without crossing boundaries.
"""

from __future__ import annotations

import re
from typing import Any


def find_subplans_over_events(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk a Postgres EXPLAIN (FORMAT JSON) plan tree. Return every node
    whose ancestry includes a ``Parent Relationship: SubPlan`` **and**
    whose scan target is the ``events`` relation.

    Used by the sweep perf regression guard (#140): the N+1 pathology we
    fixed was exactly a correlated SubPlan re-scanning ``events`` per
    outer row. If the fix ever regresses, this detector lights up.
    """
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], in_subplan: bool) -> None:
        here_subplan = node.get("Parent Relationship") == "SubPlan" or in_subplan
        if here_subplan and node.get("Relation Name") == "events":
            found.append(node)
        for child in node.get("Plans", []):
            walk(child, here_subplan)

    walk(plan_node, False)
    return found


# Aggregate node types that, when they scan ``events`` unbounded, are the
# O(session-size) read-tax shape issue #1657 removes: a WindowAgg / Aggregate /
# GroupAggregate that consumes *every* message row in the session slate.
_AGGREGATE_NODE_TYPES = frozenset({"WindowAgg", "Aggregate", "GroupAggregate"})


def _scan_has_events_lower_bound(node: dict[str, Any]) -> bool:
    """True if this scan node is an ``events`` scan bounded from below on
    ``cumulative_tokens`` or ``seq`` — i.e. it reads only a trailing window,
    not the whole session slate.

    A "lower bound" is an ``Index Cond`` / ``Recheck Cond`` / ``Filter`` that
    constrains ``cumulative_tokens`` or ``seq`` with ``>`` / ``>=`` (the drop
    boundary the windowed read scans past). A pure equality on ``session_id``
    (which every per-session scan has) is NOT a lower bound — the whole point
    of #1657 is that ``session_id = $1`` alone is O(session-size).
    """
    if node.get("Relation Name") != "events":
        return False
    cond_text = " ".join(
        str(node.get(key, "")) for key in ("Index Cond", "Recheck Cond", "Filter", "Index Name")
    )
    # A ``>``/``>=`` bound on either ordinal column. ``events_session_cumtokens_idx``
    # range scans surface as ``(cumulative_tokens > $2)``; a seq window as
    # ``(seq > ...)``. ``ORDER BY ... DESC LIMIT 1`` boundary seeks (the O(1)
    # index seeks the fix uses) are NOT aggregates and never reach this check.
    for col in ("cumulative_tokens", "seq"):
        if f"{col} >" in cond_text or f"{col} >=" in cond_text:
            return True
    return False


def _collect_scan_relations(node: dict[str, Any]) -> list[dict[str, Any]]:
    """All scan nodes at/under ``node`` that name a ``Relation Name``."""
    out: list[dict[str, Any]] = []
    if "Relation Name" in node:
        out.append(node)
    for child in node.get("Plans", []):
        out.extend(_collect_scan_relations(child))
    return out


def find_unbounded_events_aggregates(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every aggregate node (WindowAgg / Aggregate / GroupAggregate)
    that scans ``events`` **without** a ``cumulative_tokens`` / ``seq``
    lower-bound Index Cond on the scan feeding it.

    This is the plan-shape oracle for the per-turn context read-tax regression
    guard (issue #1657): the pre-fix ``_retained_class_mass`` ran an unbounded
    ``LAG() OVER (ORDER BY seq)`` — a WindowAgg over the entire message slate,
    O(session-size). The fix stores per-class running sums at append time so
    that node ceases to exist and the read is an O(1) index seek. RED before
    (the WindowAgg is present and unbounded), GREEN after (no such node) — an
    unambiguous present-vs-absent verdict, no wall-clock, no threshold.

    The detector fires only when BOTH hold for an aggregate node: (a) it is one
    of ``_AGGREGATE_NODE_TYPES``, and (b) some ``events`` scan beneath it lacks
    a ``cumulative_tokens``/``seq`` ``>``/``>=`` bound. A windowed aggregate
    that IS bounded (a legitimate trailing-window scan) does not trip it.
    """
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("Node Type") in _AGGREGATE_NODE_TYPES:
            scans = _collect_scan_relations(node)
            events_scans = [s for s in scans if s.get("Relation Name") == "events"]
            if events_scans and not any(_scan_has_events_lower_bound(s) for s in events_scans):
                found.append(node)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_node)
    return found


def find_unbounded_events_scan_over_seq(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every aggregate node (WindowAgg / Aggregate / GroupAggregate)
    whose input subtree scans the ``events`` relation *without* a
    ``cumulative_tokens`` range bound and *without* a ``seq`` lower bound.

    This is the PRIMARY plan-shape oracle for the read-path complexity harness
    (issue #1661), foreclosing the whole O(session-size) hot-path class — not
    just the single #1657 shape. The pathology it names is precisely what
    detonated Ultron: an unbounded ``LAG(cumulative_tokens) OVER (ORDER BY
    seq)`` WindowAgg (``_retained_class_mass``, born in #1611) that consumes
    *every* message row in the session slate before the fits-in-window
    short-circuit even runs. The fix stores per-class running sums at append
    time, so the aggregate node ceases to exist and the read collapses to an
    O(1) index seek.

    The verdict is asymptotic SHAPE, present-vs-absent — no wall-clock, no
    row-count threshold, no ``EXPLAIN ANALYZE``. It fires only when BOTH hold
    for an aggregate node: (a) its type is in ``_AGGREGATE_NODE_TYPES``, and
    (b) some ``events`` scan beneath it carries neither a ``cumulative_tokens``
    ``>``/``>=`` range bound nor a ``seq`` ``>``/``>=`` lower bound (i.e. it is
    keyed on ``session_id`` alone, which is O(session-size)). A windowed
    aggregate that IS bounded (a legitimate trailing-window range scan) does
    not trip it.

    Kept beside :func:`find_subplans_over_events` and
    :func:`find_unbounded_events_aggregates`, mirroring their shape-only,
    row-count-free discipline. It shares the ``events``-lower-bound predicate
    (``cumulative_tokens``/``seq`` ``>``/``>=``) with the latter; the two are
    intentionally near-synonyms — this one is the harness's named entry point,
    the other predates it as the #1657-specific guard, and keeping both makes
    the harness's registry read as a self-describing catalogue.
    """
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("Node Type") in _AGGREGATE_NODE_TYPES:
            scans = _collect_scan_relations(node)
            events_scans = [s for s in scans if s.get("Relation Name") == "events"]
            if events_scans and not any(_scan_has_events_lower_bound(s) for s in events_scans):
                found.append(node)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_node)
    return found


def find_seq_scans_over_events(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every ``Seq Scan`` node in the plan tree whose scan target is the
    ``events`` relation.

    The plan-shape oracle for partial-index regression guards (issue #1707): the
    ``model_workflow_park`` crash-recovery sweep seq-scanned ``events`` on every
    30s cross-session pass because its ``kind='span' AND data->>'event'=
    'model_workflow_park'`` predicate had no supporting index. With the migration
    0131 partial indexes in place the planner picks an index scan; a dropped /
    unusable index regresses back to a ``Seq Scan on events`` and lights this up.

    Present-vs-absent, no wall-clock, no row-count threshold — same discipline as
    :func:`find_subplans_over_events`.
    """
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("Node Type") == "Seq Scan" and node.get("Relation Name") == "events":
            found.append(node)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_node)
    return found


# The JSONB-over-column index-predicate-mismatch smell (issue #1734/#1750):
# an ``events`` Seq Scan whose ``Filter`` carries a SELECTIVE equality on
# ``data->>'role'`` or ``data->>'tool_call_id'`` — exactly the two JSONB
# expressions that a partial index (``events_tool_result_idx``,
# ``events_tool_confirmed_allow_idx``, etc., migrations 0011/0023/0065/0097)
# is predicated on via the BACKFILLED plain column instead (``role``). The
# planner cannot prove the JSONB predicate implies the column predicate, so a
# query written against the JSONB expression falls back to a bare
# ``Seq Scan on events`` keyed on ``session_id`` alone — the #1734 defect
# class (``find_tool_result_event`` pre-fix).
#
# "Selective equality" means compared against a bind parameter (``$N``) or a
# literal — NOT another ``data->>`` expression (a column-to-column join
# condition, e.g. ``tr.data->>'tool_call_id' = lc.data->>'tool_call_id'``, is
# not a scan-narrowing predicate the way a literal/parameter equality is, and
# is common in legitimate anti-join Filters this oracle must not flag).
_JSONB_PREDICATE_MISMATCH_RE = re.compile(
    r"data\s*->>\s*'(role|tool_call_id)'(::text)?\)?\s*=\s*(\$\d+(::text)?|'[^']*'(::text)?)"
)


def _has_jsonb_over_column_predicate(text: str) -> bool:
    return bool(_JSONB_PREDICATE_MISMATCH_RE.search(text))


def find_predicate_mismatch_events_scan(plan_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every ``Seq Scan on events`` whose ``Filter`` carries a selective
    ``data->>'role'``/``data->>'tool_call_id'`` equality — the index-predicate
    mismatch class that defeats a partial index predicated on the backfilled
    plain column (issue #1734, the #1 violation in the #1733 epic).

    This is a SIBLING of :func:`find_unbounded_events_scan_over_seq`, not a
    replacement: that oracle catches an unbounded AGGREGATE (WindowAgg /
    Aggregate / GroupAggregate) over ``events``; this one catches a bare
    ``Seq Scan`` (no aggregate at all — a plain ``SELECT ... LIMIT 1``) whose
    defect is an unserved selective predicate, not an unbounded aggregate.

    Deliberately narrow: it does NOT flag "any events Seq Scan keyed on
    session_id alone" (that would false-positive on legitimately-O(1)
    single-session reads and small/empty-table plans where the planner
    reasonably prefers a seq scan). It keys on the PRESENCE of the specific
    JSONB-over-column predicate smell in the scan's own ``Filter`` — the
    signal that a partial index exists to serve this exact condition (via the
    normalized column) but the query is written against the un-servable JSONB
    expression instead, so the planner falls back to scanning the relation
    (which, being a ``Seq Scan`` node itself, by definition used no index seek
    for this predicate).

    Present-vs-absent, no wall-clock, no row-count threshold — same
    discipline as :func:`find_unbounded_events_scan_over_seq` and
    :func:`find_seq_scans_over_events`.
    """
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("Node Type") == "Seq Scan" and node.get("Relation Name") == "events":
            filter_text = str(node.get("Filter", ""))
            if _has_jsonb_over_column_predicate(filter_text):
                found.append(node)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_node)
    return found


def has_external_disk_sort(plan_node: dict[str, Any]) -> bool:
    """True if any ``Sort`` node in the plan spilled to disk via an
    external merge.

    This is the SECONDARY structural fingerprint of the O(session-size)
    read-tax (issue #1661): Ultron's detonation surfaced not only as a
    multi-second WindowAgg but as a ``Sort Method: external merge  Disk:
    45,736kB`` — the sort of the whole slate could not fit ``work_mem`` and
    spilled. A trailing-window read never sorts enough rows to spill; a
    full-slate aggregate over 90k rows does.

    ``EXPLAIN`` *without* ``ANALYZE`` reports the planner's *estimated* sort
    method in ``Sort Method`` only when it can foresee the spill; the runtime
    ``Actual Rows`` / measured spill size live under ``ANALYZE`` and are
    deliberately NOT consulted here (no row-count, no wall-clock in any gating
    path). This helper is therefore an *advisory* backstop signal — used to
    enrich diagnostics, never as a standalone gate — and reads only the
    ``Sort Method`` string Postgres already emits in the plan tree.
    """

    def walk(node: dict[str, Any]) -> bool:
        method = str(node.get("Sort Method", ""))
        if "external" in method.lower() and "disk" in method.lower():
            return True
        # Also honor the bare ``external merge`` / ``external sort`` phrasing,
        # which some Postgres versions emit without the literal word "disk".
        if "external merge" in method.lower() or "external sort" in method.lower():
            return True
        return any(walk(child) for child in node.get("Plans", []))

    return walk(plan_node)
