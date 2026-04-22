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
