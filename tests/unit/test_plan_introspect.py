"""Unit tests for the ``find_subplans_over_events`` detector used by the
sweep perf regression guard (issue #140).

The e2e test (``tests/e2e/test_sweep_perf.py``) only proves the current
production SQL is SubPlan-free — it cannot show that the detector would
actually catch a regression. These tests fill that gap: hand-crafted plan
fixtures (one pathological, one clean) demonstrate the detector fires on
the former and stays silent on the latter.

This is the "RED evidence" in the TDD cycle that never rots: even after
the production SQL is fixed, these tests still prove the guard works.
"""

from __future__ import annotations

from tests.support import find_subplans_over_events

# Fixture: an EXPLAIN (FORMAT JSON) "Plan" subtree matching the pre-#140
# shape — outer Index Scan on events whose Filter contains a SubPlan
# that itself scans events. This is the pathology the e2e test guards
# against; the detector must flag it.
_BAD_PLAN = {
    "Node Type": "Unique",
    "Plans": [
        {
            "Node Type": "Nested Loop",
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "events",
                    "Plans": [
                        {
                            "Node Type": "Bitmap Heap Scan",
                            "Relation Name": "events",
                            "Parent Relationship": "SubPlan",
                            "Subplan Name": "SubPlan 3",
                            "Plans": [
                                {
                                    "Node Type": "Bitmap Index Scan",
                                    "Parent Relationship": "Outer",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}

# Fixture: the post-#140 shape — Hash Join over a HashAggregate on events.
# No SubPlan over events anywhere in the tree; the detector must not fire.
_GOOD_PLAN = {
    "Node Type": "Unique",
    "Plans": [
        {
            "Node Type": "Hash Join",
            "Plans": [
                {
                    "Node Type": "Hash Left Join",
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "events",
                            "Parent Relationship": "Outer",
                        },
                        {
                            "Node Type": "Hash",
                            "Plans": [
                                {
                                    "Node Type": "HashAggregate",
                                    "Plans": [
                                        {
                                            "Node Type": "Seq Scan",
                                            "Relation Name": "events",
                                            "Parent Relationship": "Outer",
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    ],
}


class TestFindSubplansOverEvents:
    def test_detects_correlated_subplan_over_events(self) -> None:
        """The pre-#140 shape (SubPlan re-scanning events) must be caught."""
        found = find_subplans_over_events(_BAD_PLAN)
        assert len(found) == 1
        assert found[0]["Parent Relationship"] == "SubPlan"
        assert found[0]["Relation Name"] == "events"

    def test_ignores_hoisted_cte_shape(self) -> None:
        """The post-#140 shape (HashAggregate-driven) must NOT trip the guard."""
        found = find_subplans_over_events(_GOOD_PLAN)
        assert found == []

    def test_ignores_subplans_over_non_events_relations(self) -> None:
        """A SubPlan scanning some other table is fine — we only care about
        events, the load-bearing hot table."""
        plan = {
            "Node Type": "Seq Scan",
            "Relation Name": "sessions",
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "agents",
                    "Parent Relationship": "SubPlan",
                }
            ],
        }
        assert find_subplans_over_events(plan) == []

    def test_inherits_subplan_context_into_descendants(self) -> None:
        """A deeper Bitmap Heap Scan inside a SubPlan subtree still counts —
        Postgres often labels the outer node with the Parent Relationship
        and its children get 'Outer' / 'Inner' relationships."""
        plan = {
            "Node Type": "Index Scan",
            "Relation Name": "events",
            "Plans": [
                {
                    "Node Type": "Aggregate",
                    "Parent Relationship": "SubPlan",
                    "Plans": [
                        {
                            "Node Type": "Bitmap Heap Scan",
                            "Relation Name": "events",
                            "Parent Relationship": "Outer",
                        }
                    ],
                }
            ],
        }
        found = find_subplans_over_events(plan)
        assert len(found) == 1
        assert found[0]["Node Type"] == "Bitmap Heap Scan"
