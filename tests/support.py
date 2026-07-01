"""Test-side utilities shared across tiers.

Kept at the top level of ``tests/`` (not inside ``unit/`` or ``e2e/``) so
either tier can import it without crossing boundaries.
"""

from __future__ import annotations

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
