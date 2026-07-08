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

from tests.support import find_predicate_mismatch_events_scan, find_subplans_over_events

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


# ─── find_predicate_mismatch_events_scan (issue #1750) ───────────────────────
#
# Hand-crafted fixtures — the same "RED evidence that never rots" discipline
# as the SubPlan detector above: these pin the detector's behavior
# independent of any live Postgres plan, so they keep proving the oracle
# works even after the production SQL is fixed.

# The pre-#1734 shape: ``find_tool_result_event`` filtered on
# ``data->>'role' = 'tool'`` (a JSONB expression), which the partial index
# ``events_tool_result_idx`` (predicated on the ``role`` COLUMN) cannot serve
# — so the planner falls back to a bare Seq Scan on events.
_BAD_PREDICATE_MISMATCH_PLAN = {
    "Node Type": "Seq Scan",
    "Relation Name": "events",
    "Filter": (
        "((session_id = 'sess_1'::text) AND (account_id = 'acc_1'::text) "
        "AND (kind = 'message'::text) AND ((data ->> 'role'::text) = 'tool'::text) "
        "AND ((data ->> 'tool_call_id'::text) = 'tc_1'::text))"
    ),
}

# The post-#1734 fix: the SAME query, re-predicated onto the normalized
# ``role`` column, planned as an Index Scan (the partial index applies) —
# no Seq Scan node at all.
_GOOD_INDEX_SCAN_PLAN = {
    "Node Type": "Index Scan",
    "Relation Name": "events",
    "Index Name": "events_tool_result_idx",
    "Index Cond": "((session_id = 'sess_1'::text) AND ((data ->> 'tool_call_id'::text) = 'tc_1'::text))",
    "Filter": "((account_id = 'acc_1'::text) AND (kind = 'message'::text) AND (role = 'tool'::text))",
}

# A legitimately-O(1) single-session Seq Scan with NO selective JSONB
# equality in its Filter (e.g. a tiny/empty table where the planner
# reasonably prefers a full scan over an index probe). Must NOT trip the
# detector — it is not the index-predicate-mismatch class.
_BENIGN_SEQ_SCAN_PLAN = {
    "Node Type": "Seq Scan",
    "Relation Name": "events",
    "Filter": "(session_id = 'sess_1'::text)",
}

# A Seq Scan on some OTHER relation carrying the same JSONB predicate text —
# must not fire; the detector is events-specific (the load-bearing hot table).
_OTHER_RELATION_SEQ_SCAN_PLAN = {
    "Node Type": "Seq Scan",
    "Relation Name": "sessions",
    "Filter": "((data ->> 'role'::text) = 'tool'::text)",
}


class TestFindPredicateMismatchEventsScan:
    def test_detects_jsonb_role_predicate_mismatch(self) -> None:
        """The pre-#1734 ``data->>'role'`` Seq Scan must be caught."""
        found = find_predicate_mismatch_events_scan(_BAD_PREDICATE_MISMATCH_PLAN)
        assert len(found) == 1
        assert found[0]["Node Type"] == "Seq Scan"
        assert found[0]["Relation Name"] == "events"

    def test_ignores_post_fix_index_scan(self) -> None:
        """The post-#1734 ``role`` column form (an Index Scan, no Seq Scan
        node at all) must NOT trip the guard."""
        found = find_predicate_mismatch_events_scan(_GOOD_INDEX_SCAN_PLAN)
        assert found == []

    def test_ignores_benign_seq_scan_without_selective_predicate(self) -> None:
        """A Seq Scan keyed on session_id alone, with no JSONB role/tool_call_id
        equality, is a different (possibly legitimate) shape — not this
        detector's target."""
        found = find_predicate_mismatch_events_scan(_BENIGN_SEQ_SCAN_PLAN)
        assert found == []

    def test_ignores_other_relations(self) -> None:
        """The detector only fires on the ``events`` relation."""
        found = find_predicate_mismatch_events_scan(_OTHER_RELATION_SEQ_SCAN_PLAN)
        assert found == []

    def test_ignores_column_to_column_join_condition(self) -> None:
        """A ``data->>'tool_call_id' = data->>'tool_call_id'`` JOIN/anti-join
        condition (comparing two JSONB expressions, not a bind param/literal)
        is a normal correlation predicate, not the index-predicate-mismatch
        smell — must NOT trip the guard."""
        plan = {
            "Node Type": "Seq Scan",
            "Relation Name": "events",
            "Filter": ("((tr.data ->> 'tool_call_id'::text) = (lc.data ->> 'tool_call_id'::text))"),
        }
        found = find_predicate_mismatch_events_scan(plan)
        assert found == []

    def test_detects_tool_call_id_variant_nested(self) -> None:
        """The ``data->>'tool_call_id'`` half of the smell is also caught,
        including when nested under other plan nodes."""
        plan = {
            "Node Type": "Nested Loop",
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "events",
                    "Filter": "((data ->> 'tool_call_id'::text) = 'tc_9'::text)",
                }
            ],
        }
        found = find_predicate_mismatch_events_scan(plan)
        assert len(found) == 1
