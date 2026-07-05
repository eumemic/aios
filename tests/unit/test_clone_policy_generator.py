"""Unit tests for the #1676 clone-policy projection generator.

Pure (no DB): these pin the generator's arm→SQL mapping and the structural
invariants of the policy tables.  The *schema*-coverage gate lives in the
integration tier (``test_clone_policy_completeness.py``); here we prove the
generator emits the right projection given a policy.
"""

from __future__ import annotations

import pytest

from aios.db.queries.clone_policy import (
    CLONE_POLICIES,
    EVENTS_POLICY,
    SESSIONS_POLICY,
    TRIGGERS_POLICY,
    Arm,
    build_projection,
)


def test_reset_default_columns_are_omitted() -> None:
    """RESET_DEFAULT is the only arm that drops its column from the projection."""
    policy = {
        "id": Arm.MINT_ID,
        "keep": Arm.COPY,
        "gone": Arm.RESET_DEFAULT,
    }
    proj = build_projection(
        policy, source_alias="s", new_id_expr="i.id", session_id_param="$2"
    )
    assert proj.columns == ("id", "keep")
    assert proj.select_exprs == ("i.id", "s.keep")
    assert "gone" not in proj.insert_columns_sql


def test_copy_uses_alias_when_present_else_bare() -> None:
    policy = {"c": Arm.COPY}
    aliased = build_projection(
        policy, source_alias="s", new_id_expr="i.id", session_id_param="$2"
    )
    bare = build_projection(
        policy, source_alias="", new_id_expr="$1", session_id_param="$1"
    )
    assert aliased.select_exprs == ("s.c",)
    assert bare.select_exprs == ("c",)


def test_remap_session_and_new_value_and_mint_ingest() -> None:
    policy = {
        "id": Arm.MINT_ID,
        "session_id": Arm.REMAP_SESSION,
        "workspace_volume_path": Arm.NEW_VALUE,
        "ingest_token_hash": Arm.MINT_INGEST_TOKEN,
    }
    proj = build_projection(
        policy,
        source_alias="s",
        new_id_expr="i.id",
        session_id_param="$2",
        new_value_exprs={
            "workspace_volume_path": "$3",
            "ingest_token_hash": "CASE WHEN s.source='external_event' "
            "THEN i.h ELSE NULL END",
        },
    )
    assert proj.select_exprs == (
        "i.id",
        "$2",
        "$3",
        "CASE WHEN s.source='external_event' THEN i.h ELSE NULL END",
    )


def test_new_value_without_expression_raises() -> None:
    with pytest.raises(KeyError):
        build_projection(
            {"wp": Arm.NEW_VALUE},
            source_alias="s",
            new_id_expr="i.id",
            session_id_param="$2",
        )


def test_column_order_follows_policy_insertion_order() -> None:
    policy = {"b": Arm.COPY, "a": Arm.COPY, "c": Arm.COPY}
    proj = build_projection(
        policy, source_alias="s", new_id_expr="i.id", session_id_param="$2"
    )
    assert proj.columns == ("b", "a", "c")


def test_every_policy_column_has_a_valid_arm() -> None:
    """Structural: no column is left unclassified (no ``None`` / stray value)."""
    for name, policy in CLONE_POLICIES.items():
        for col, arm in policy.items():
            assert isinstance(arm, Arm), f"{name}.{col} has non-Arm {arm!r}"


def test_events_policy_copies_the_0127_class_mass_columns() -> None:
    """The #1676 LIVE drift fix: the 0127 counters must be COPY, not omitted."""
    for col in (
        "cumulative_messages",
        "cumulative_text_mass",
        "cumulative_tool_result_mass",
        "cumulative_thinking_mass",
        "cumulative_tool_use_mass",
    ):
        assert EVENTS_POLICY[col] is Arm.COPY


def test_sessions_authority_arms() -> None:
    """The reviewed authority decisions: surface/model COPY, run-lineage RESET."""
    for col in ("tools", "mcp_servers", "http_servers", "surface_frozen", "model",
                "litellm_extra"):
        assert SESSIONS_POLICY[col] is Arm.COPY, col
    for col in ("parent_run_id", "origin"):
        assert SESSIONS_POLICY[col] is Arm.RESET_DEFAULT, col


def test_triggers_ingest_token_is_source_conditional() -> None:
    assert TRIGGERS_POLICY["ingest_token_hash"] is Arm.MINT_INGEST_TOKEN
    for col in ("running_since", "last_fire_at", "last_fire_status",
                "consecutive_failures"):
        assert TRIGGERS_POLICY[col] is Arm.RESET_DEFAULT, col
