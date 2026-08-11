"""Tests for lane activation models and script structure.

Tests the pure-data layer (``aios.lanes.models``) and the structural
properties of the ``lane_activate`` workflow script — that it compiles,
declares ``async def main(input)``, references the required constants,
and follows the value-domain I/O contract.
"""

from __future__ import annotations

from typing import Any

import pytest

# ── LaneLock parsing tests ──────────────────────────────────────────────────


def _sample_lock() -> dict[str, Any]:
    """A minimal valid lock-file dict."""
    return {
        "_provenance": {
            "builder_version": "lane-expand/3",
            "resolved_agent_ids": {"launcher": "agent-id-123"},
            "spec_hash": "abc123def456",
        },
        "workflow": {
            "name": "__lane__test-workflow",
            "script": "async def main(input):\n    return {'ok': True}\n",
            "description": "Test workflow",
            "tools": [{"type": "http_request", "enabled": True}],
            "http_servers": [{"name": "github", "base_url": "https://api.github.com"}],
        },
        "cron_trigger": {
            "name": "__lane__test-trigger",
            "trigger_name": "lane-test-trigger",
            "action": {
                "workflow_id": "__lane__test-workflow",
                "input_template": {"repo": "eumemic/aios"},
                "vault_ids": ["vault-1"],
                "workflow_version": None,
            },
            "source": {
                "schedule": "0,30 * * * *",
                "timezone": "UTC",
            },
            "enabled": True,
        },
        "launcher_agent": {
            "name": "__lane__test-launcher",
            "model": "anthropic/claude-sonnet-4-20250514",
            "description": "Test launcher agent",
            "tools": [],
            "http_servers": [],
        },
        "launcher_session": {
            "agent_id": "__lane__test-launcher",
            "environment_id": "dev-env",
            "title": "__lane__test-launcher",
            "archive_when_idle": False,
            "vault_ids": ["vault-1"],
        },
    }


class TestLaneLockParsing:
    """Tests for LaneLock.from_dict()."""

    def test_round_trip_minimal(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.provenance.builder_version == "lane-expand/3"
        assert lock.provenance.spec_hash == "abc123def456"
        assert lock.workflow.name == "__lane__test-workflow"
        assert "async def main" in lock.workflow.script

    def test_provenance_fields(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.provenance.resolved_agent_ids == {"launcher": "agent-id-123"}

    def test_workflow_fields(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.workflow.description == "Test workflow"
        assert len(lock.workflow.tools) == 1
        assert lock.workflow.tools[0]["type"] == "http_request"
        assert len(lock.workflow.http_servers) == 1

    def test_cron_trigger_fields(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.cron_trigger.name == "__lane__test-trigger"
        assert lock.cron_trigger.trigger_name == "lane-test-trigger"
        assert lock.cron_trigger.action.workflow_id == "__lane__test-workflow"
        assert lock.cron_trigger.action.input_template == {"repo": "eumemic/aios"}
        assert lock.cron_trigger.action.vault_ids == ["vault-1"]
        assert lock.cron_trigger.action.workflow_version is None
        assert lock.cron_trigger.source.schedule == "0,30 * * * *"
        assert lock.cron_trigger.source.timezone == "UTC"
        assert lock.cron_trigger.enabled is True

    def test_launcher_agent_fields(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.launcher_agent.name == "__lane__test-launcher"
        assert lock.launcher_agent.model == "anthropic/claude-sonnet-4-20250514"
        assert lock.launcher_agent.description == "Test launcher agent"

    def test_launcher_session_fields(self):
        from aios.lanes.models import LaneLock

        lock = LaneLock.from_dict(_sample_lock())
        assert lock.launcher_session.agent_id == "__lane__test-launcher"
        assert lock.launcher_session.environment_id == "dev-env"
        assert lock.launcher_session.title == "__lane__test-launcher"
        assert lock.launcher_session.archive_when_idle is False
        assert lock.launcher_session.vault_ids == ["vault-1"]

    def test_missing_optional_provenance_fields(self):
        from aios.lanes.models import LaneLock

        data = _sample_lock()
        data["_provenance"] = {"builder_version": "v1"}
        lock = LaneLock.from_dict(data)
        assert lock.provenance.resolved_agent_ids == {}
        assert lock.provenance.spec_hash == ""

    def test_missing_provenance_section(self):
        from aios.lanes.models import LaneLock

        data = _sample_lock()
        del data["_provenance"]
        lock = LaneLock.from_dict(data)
        assert lock.provenance.builder_version == ""

    def test_missing_required_field_raises(self):
        from aios.lanes.models import LaneLock

        data = _sample_lock()
        del data["workflow"]["name"]
        with pytest.raises(KeyError):
            LaneLock.from_dict(data)

    def test_optional_workflow_description_none(self):
        from aios.lanes.models import LaneLock

        data = _sample_lock()
        data["workflow"]["description"] = None
        lock = LaneLock.from_dict(data)
        assert lock.workflow.description is None

    def test_empty_tools_and_servers(self):
        from aios.lanes.models import LaneLock

        data = _sample_lock()
        data["workflow"]["tools"] = []
        data["workflow"]["http_servers"] = []
        lock = LaneLock.from_dict(data)
        assert lock.workflow.tools == []
        assert lock.workflow.http_servers == []


# ── ActivationResult tests ──────────────────────────────────────────────────


class TestActivationResult:
    """Tests for ActivationResult and ObjectDelta."""

    def test_to_dict_activated(self):
        from aios.lanes.models import ActivationOutcome, ActivationResult, ObjectDelta

        result = ActivationResult(
            outcome=ActivationOutcome.ACTIVATED,
            lane="test-lane",
            merge_sha="abc123",
            spec_hash="def456",
            deltas=[
                ObjectDelta(
                    object_kind="workflow",
                    object_name="wf-1",
                    action="created",
                    object_id="wf-id-1",
                    new_version=1,
                ),
                ObjectDelta(
                    object_kind="agent",
                    object_name="agent-1",
                    action="unchanged",
                    object_id="agent-id-1",
                    old_version=3,
                ),
            ],
            verification={"telemetry_repo_in_script": True},
        )
        d = result.to_dict()
        assert d["outcome"] == "activated"
        assert d["lane"] == "test-lane"
        assert d["merge_sha"] == "abc123"
        assert d["spec_hash"] == "def456"
        assert len(d["deltas"]) == 2
        assert d["deltas"][0]["action"] == "created"
        assert d["deltas"][0]["new_version"] == 1
        assert d["deltas"][1]["action"] == "unchanged"
        assert d["verification"]["telemetry_repo_in_script"] is True
        assert d["error"] is None

    def test_to_dict_no_op(self):
        from aios.lanes.models import ActivationOutcome, ActivationResult

        result = ActivationResult(
            outcome=ActivationOutcome.NO_OP,
            lane="test-lane",
            merge_sha="abc123",
            spec_hash="def456",
        )
        d = result.to_dict()
        assert d["outcome"] == "no_op"
        assert d["deltas"] == []

    def test_to_dict_failed(self):
        from aios.lanes.models import ActivationOutcome, ActivationResult

        result = ActivationResult(
            outcome=ActivationOutcome.FAILED,
            lane="test-lane",
            merge_sha="abc123",
            spec_hash="",
            error="lock file not found",
        )
        d = result.to_dict()
        assert d["outcome"] == "failed"
        assert d["error"] == "lock file not found"

    def test_outcome_enum_values(self):
        from aios.lanes.models import ActivationOutcome

        assert ActivationOutcome.ACTIVATED.value == "activated"
        assert ActivationOutcome.NO_OP.value == "no_op"
        assert ActivationOutcome.FAILED.value == "failed"


# ── Workflow script structural tests ─────────────────────────────────────────


class TestLaneActivateScript:
    """Structural tests for the lane_activate workflow script.

    These verify the script's structure and contract without running it
    in the full wf_script_host — they compile it, inspect the namespace,
    and check that it declares the required entry point and constants.
    """

    def test_script_compiles(self):
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        code = compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec")
        assert code is not None

    def test_script_declares_main(self):
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        assert "main" in namespace
        assert callable(namespace["main"])

    def test_script_declares_constants(self):
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        assert namespace["GITHUB_SERVER"] == "github"
        assert namespace["AIOS_SERVER"] == "aios"
        assert namespace["TELEMETRY_REPO"] == "eumemic/eumemic-company"
        assert namespace["TELEMETRY_PATH"] == "ops/telemetry/resource_telemetry.json"

    def test_script_lock_path_template(self):
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        namespace: dict[str, Any] = {}
        exec(compile(LANE_ACTIVATE_SCRIPT, "<lane_activate>", "exec"), namespace)
        template = namespace["LOCK_PATH_TEMPLATE"]
        assert "{lane}" in template
        assert template.format(lane="test") == "app/infra/lanes/test.lock.json"

    def test_script_references_tool(self):
        """The script must use tool() for http_request calls."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        assert 'tool("http_request"' in LANE_ACTIVATE_SCRIPT

    def test_script_uses_phases(self):
        """The script must use phase() for structured progress reporting."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        assert 'phase("read-lock")' in LANE_ACTIVATE_SCRIPT
        assert 'phase("ensure-workflow")' in LANE_ACTIVATE_SCRIPT
        assert 'phase("ensure-agent")' in LANE_ACTIVATE_SCRIPT
        assert 'phase("ensure-session")' in LANE_ACTIVATE_SCRIPT
        assert 'phase("ensure-trigger")' in LANE_ACTIVATE_SCRIPT
        assert 'phase("verify")' in LANE_ACTIVATE_SCRIPT

    def test_script_never_deletes(self):
        """The script must never issue DELETE requests (never-delete invariant)."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        # The script should only use GET, POST, PUT — never DELETE
        assert '"DELETE"' not in LANE_ACTIVATE_SCRIPT
        assert "'DELETE'" not in LANE_ACTIVATE_SCRIPT

    def test_script_returns_typed_result(self):
        """The script's main() return paths must include outcome field."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        # All return paths should include "outcome"
        assert '"outcome"' in LANE_ACTIVATE_SCRIPT
        assert '"activated"' in LANE_ACTIVATE_SCRIPT
        assert '"no_op"' in LANE_ACTIVATE_SCRIPT
        assert '"failed"' in LANE_ACTIVATE_SCRIPT

    def test_script_uses_optimistic_concurrency(self):
        """The script must pass version for optimistic concurrency on updates."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        # The update bodies must include "version" for optimistic concurrency
        assert '"version": live_version' in LANE_ACTIVATE_SCRIPT

    def test_script_handles_missing_input(self):
        """The script must handle missing lane/merge_sha gracefully."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        assert "missing required input" in LANE_ACTIVATE_SCRIPT


# ── Integration contract tests ──────────────────────────────────────────────


class TestScriptWfHostContract:
    """Verify the script meets the wf_script_host contract.

    The wf_script_host requires:
    1. ``async def main(input)`` at module level
    2. Only uses capabilities from ``author_namespace()``
    3. Returns a value (not raise) for the run output
    """

    def test_main_is_async_generator_protocol(self):
        """main(input) must be async (returns a coroutine)."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        assert "async def main(input)" in LANE_ACTIVATE_SCRIPT

    def test_script_only_uses_allowed_builtins(self):
        """The script should only use json, base64 (safe builtins) plus tool/log/phase."""
        from aios.lanes.activate_script import LANE_ACTIVATE_SCRIPT

        # Should not import anything that requires filesystem or network directly
        assert "import os" not in LANE_ACTIVATE_SCRIPT
        assert "import subprocess" not in LANE_ACTIVATE_SCRIPT
        assert "import socket" not in LANE_ACTIVATE_SCRIPT
        assert "import requests" not in LANE_ACTIVATE_SCRIPT
        assert "import httpx" not in LANE_ACTIVATE_SCRIPT
