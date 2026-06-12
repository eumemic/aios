"""Fixture builder for the workflow replay-semantics fingerprint test."""

from __future__ import annotations

from typing import Any

from aios.workflows.child_id import child_session_id
from aios.workflows.determinism import (
    HOST_SEMANTICS_EPOCH,
    CallKeyer,
    canonical_json,
    canonical_schema_json,
    storable_text,
)
from aios.workflows.wf_script_host import _IMPORTABLE_MODULES, _SAFE_BUILTIN_NAMES

# Deadline/frame/workspace-survivability constants are deliberately excluded: they can
# make a wake live or die, but they do not alter replay equality or call_key derivation.
# Sandbox filesystem resume policy is likewise outside today's keyer; a future
# resume-health decision may choose to bump the epoch if that axis becomes replay-visible.


def _keys_for(prefix: str, specs: list[tuple[str, Any]]) -> list[str]:
    keyer = CallKeyer()
    return [prefix + keyer.next(capability_id, spec) for capability_id, spec in specs]


def build_snapshot() -> dict[str, Any]:
    repeated_spec = ("agent", {"agent_id": "ag_1", "input": {"same": True}})
    base_specs: list[tuple[str, Any]] = [
        ("gate", {"b": 1.0, "a": ("tuple", {"z": [3.25, 2.0]})}),
        ("tool", {"tool_name": "web", "input": {"nested": {"y": 2, "x": 1.0}}}),
        repeated_spec,
        repeated_spec,
        ("agent", {"agent_id": "ag_2", "input": storable_text("nul\x00surrogate\ud800")}),
    ]
    branch_specs = [
        ("agent", {"agent_id": "ag_branch", "input": {"items": (1.0, 1.5)}}),
        ("gate", {"branch": "two"}),
    ]
    child_inputs = [
        ("run_01J00000000000000000000000", "sha:" + "a" * 64 + "#0"),
        ("run_01J00000000000000000000000", "0.1/sha:" + "b" * 64 + "#2"),
    ]
    return {
        "epoch": HOST_SEMANTICS_EPOCH,
        "canonical_json": {
            "int_float_tuple_order": canonical_json({"z": (1.0, 1.5), "a": {"b": 2.0, "a": True}}),
            "unicode": canonical_json({"text": "snowman ☃ and slash /"}),
            "sanitized_text": storable_text("nul\x00surrogate\ud800"),
        },
        "canonical_schema_json": canonical_schema_json(
            {
                "type": "number",
                "minimum": 1.0,
                "multipleOf": 0.25,
                "description": "float literals stay verbatim",
            }
        ),
        "call_keys": {
            "base": _keys_for("", base_specs),
            "parallel_branch": _keys_for("0.1/", branch_specs),
        },
        "surface": {
            "namespace": [
                "AgentError",
                "AgentNoReturnError",
                "agent",
                "budget",
                "gate",
                "log",
                "parallel",
                "phase",
                "pipeline",
                "tool",
            ],
            "safe_builtins": sorted(_SAFE_BUILTIN_NAMES),
            "importable_modules": sorted(_IMPORTABLE_MODULES),
        },
        "child_session_ids": [
            {
                "run_id": run_id,
                "call_key": call_key,
                "session_id": child_session_id(run_id, call_key),
            }
            for run_id, call_key in child_inputs
        ],
    }
