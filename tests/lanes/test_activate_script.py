"""Tests for src/aios/lanes/activate_script.py — verify the script text is well-formed."""

from __future__ import annotations

import ast
import textwrap

from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT


class TestLaneActivateScript:
    """Structural checks on the embedded workflow script constant."""

    def test_script_is_nonempty_string(self) -> None:
        assert isinstance(LANE_ACTIVATE_SCRIPT, str)
        assert len(LANE_ACTIVATE_SCRIPT) > 500

    def test_script_parses_as_valid_python(self) -> None:
        """The script text must be valid Python (parseable by ast)."""
        tree = ast.parse(LANE_ACTIVATE_SCRIPT)
        assert isinstance(tree, ast.Module)

    def test_script_defines_main(self) -> None:
        """The script must define an async main(input) entry point."""
        tree = ast.parse(LANE_ACTIVATE_SCRIPT)
        top_level_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and isinstance(node, ast.AsyncFunctionDef)
            and node.name == "main"
        ]
        assert "main" in top_level_names, "script must define 'async def main(input)'"

    def test_script_contains_telemetry_repo(self) -> None:
        """Post-apply verification requires TELEMETRY_REPO constant."""
        assert "eumemic/eumemic-company" in LANE_ACTIVATE_SCRIPT

    def test_script_contains_telemetry_path(self) -> None:
        assert "ops/telemetry/resource_telemetry.json" in LANE_ACTIVATE_SCRIPT

    def test_script_contains_lock_path_template(self) -> None:
        assert "app/infra/lanes/{lane}.lock.json" in LANE_ACTIVATE_SCRIPT

    def test_script_uses_optimistic_concurrency(self) -> None:
        """The script should pass version on updates for optimistic concurrency."""
        assert '"version"' in LANE_ACTIVATE_SCRIPT or "'version'" in LANE_ACTIVATE_SCRIPT

    def test_script_never_deletes(self) -> None:
        """The script must not contain DELETE method calls (never-delete invariant)."""
        assert '"DELETE"' not in LANE_ACTIVATE_SCRIPT
        assert "'DELETE'" not in LANE_ACTIVATE_SCRIPT

    def test_script_returns_typed_result(self) -> None:
        """The script should return outcome/lane/merge_sha/spec_hash/deltas."""
        for key in ["outcome", "lane", "merge_sha", "spec_hash", "deltas", "verification"]:
            assert f'"{key}"' in LANE_ACTIVATE_SCRIPT or f"'{key}'" in LANE_ACTIVATE_SCRIPT

    def test_script_has_phase_calls(self) -> None:
        """The script should use phase() for structured progress reporting."""
        assert "phase(" in LANE_ACTIVATE_SCRIPT

    def test_script_has_six_phases(self) -> None:
        """Should have phases: read-lock, ensure-workflow, ensure-agent, ensure-session, ensure-trigger, verify."""
        for phase_name in ["read-lock", "ensure-workflow", "ensure-agent", "ensure-session", "ensure-trigger", "verify"]:
            assert phase_name in LANE_ACTIVATE_SCRIPT, f"missing phase: {phase_name}"
