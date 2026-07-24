"""Tests for the retirement registry + descriptor (#1573).

This is the declarative core of the pattern-retirement lifecycle (epic #1572):
the ``Retirement`` dataclass and the seeded registry. These tests pin the
acceptance criteria — the two seeded descriptors exist, each registers all SEVEN
persisted tool surfaces (including the ``connectors.tools_schema`` surface the
ad-hoc retirement missed), and the dataclass enforces action/successor
consistency.
"""

from __future__ import annotations

import pytest

from aios.retirements import Retirement, Surface
from aios.retirements import registry as reg

# The seven persisted tool surfaces, as (table, jsonb_col) pairs.
EXPECTED_TOOL_SURFACES = {
    ("agents", "tools"),
    ("agent_versions", "tools"),
    ("workflows", "tools"),
    ("workflow_versions", "tools"),
    ("wf_runs", "tools"),
    ("sessions", "tools"),
    # The SEVENTH surface — the connector tools_schema the ad-hoc retirement missed.
    ("connectors", "tools_schema"),
}


def test_registry_seeds_two_descriptors() -> None:
    assert len(reg.REGISTRY) == 2
    assert reg.LEGACY_BUILTIN_RENAMES in reg.REGISTRY
    assert reg.RETIRED_GOAL_OUTCOME_BUILTINS in reg.REGISTRY


def test_every_descriptor_registers_all_seven_surfaces() -> None:
    for retirement in reg.REGISTRY:
        pairs = {(s.table, s.jsonb_col) for s in retirement.surfaces}
        assert pairs == EXPECTED_TOOL_SURFACES, (
            f"{retirement.domain} retirement is missing surfaces: {EXPECTED_TOOL_SURFACES - pairs}"
        )
        # Exactly seven, no dupes.
        assert len(retirement.surfaces) == 7


def test_seventh_surface_is_the_connectors_tools_schema() -> None:
    # The silent hole the original ad-hoc retirement missed.
    for retirement in reg.REGISTRY:
        connector_surfaces = [s for s in retirement.surfaces if s.table == "connectors"]
        assert len(connector_surfaces) == 1
        assert connector_surfaces[0].jsonb_col == "tools_schema"


def test_sessions_surface_is_nullable_others_are_not() -> None:
    for retirement in reg.REGISTRY:
        by_table = {s.table: s for s in retirement.surfaces}
        assert by_table["sessions"].nullable is True
        for table in (
            "agents",
            "agent_versions",
            "workflows",
            "workflow_versions",
            "wf_runs",
            "connectors",
        ):
            assert by_table[table].nullable is False


def test_shared_predicate_substitutes_jsonb_col_and_binds_token() -> None:
    for retirement in reg.REGISTRY:
        for surface in retirement.surfaces:
            # The column is substituted into the predicate; :token stays bound.
            assert surface.jsonb_col in surface.predicate_sql
            assert "<jsonb_col>" not in surface.predicate_sql
            assert ":token" in surface.predicate_sql
            assert "jsonb_array_elements" in surface.predicate_sql


def test_legacy_builtin_renames_descriptor() -> None:
    r = reg.LEGACY_BUILTIN_RENAMES
    assert r.action == "rename"
    assert r.domain == reg.TOOL_SURFACE_DOMAIN
    assert r.token_map() == {
        "invoke": "call_session",
        "invoke_agent": "call_agent",
        "invoke_workflow": "call_workflow",
        "create_run": "call_workflow",
        "await_run": "call_workflow",
        "cancel_run": "stop_task",
    }
    assert r.contract_rev == "0155"


def test_retired_goal_outcome_descriptor_is_drop_with_no_successor() -> None:
    r = reg.RETIRED_GOAL_OUTCOME_BUILTINS
    assert r.action == "drop"
    assert set(r.tokens) == {"complete_goal", "fail_goal"}
    assert all(succ is None for succ in r.token_map().values())


def test_rename_requires_successor_for_every_token() -> None:
    with pytest.raises(ValueError):
        Retirement(
            domain="tool_surface",
            action="rename",
            token="x",
            successor=None,
            surfaces=reg.TOOL_SURFACES,
            introduced_rev="0001",
        )


def test_drop_must_not_have_a_successor() -> None:
    with pytest.raises(ValueError):
        Retirement(
            domain="tool_surface",
            action="drop",
            token="x",
            successor="y",
            surfaces=reg.TOOL_SURFACES,
            introduced_rev="0001",
        )


def test_retirement_requires_at_least_one_surface() -> None:
    with pytest.raises(ValueError):
        Retirement(
            domain="tool_surface",
            action="drop",
            token="x",
            surfaces=(),
            introduced_rev="0001",
        )


def test_retirement_requires_a_token_or_mappings() -> None:
    with pytest.raises(ValueError):
        Retirement(
            domain="tool_surface",
            action="drop",
            surfaces=reg.TOOL_SURFACES,
            introduced_rev="0001",
        )


def test_surface_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        Surface(table="", jsonb_col="tools", predicate_sql="x")
    with pytest.raises(ValueError):
        Surface(table="agents", jsonb_col="", predicate_sql="x")
    with pytest.raises(ValueError):
        Surface(table="agents", jsonb_col="tools", predicate_sql="")


def test_registry_is_importable_as_single_source() -> None:
    # Sanity: the module-level REGISTRY tuple is the consulted artifact.
    assert isinstance(reg.REGISTRY, tuple)
    assert all(isinstance(r, Retirement) for r in reg.REGISTRY)
