"""Tests for src/aios/lanes/models.py — pure data-model tests, no I/O."""

from __future__ import annotations

from typing import Any

import pytest

from aios.lanes.models import (
    ActivationOutcome,
    ActivationResult,
    LaneLock,
    ObjectDelta,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_LOCK_DICT: dict[str, Any] = {
    "_provenance": {
        "builder_version": "0.1.0",
        "resolved_agent_ids": {"my-agent": "ag_123"},
        "spec_hash": "abc123",
    },
    "workflow": {
        "name": "__lane__test-workflow",
        "script": "print('hello')",
        "description": "test wf",
        "tools": [],
        "http_servers": [],
    },
    "cron_trigger": {
        "name": "__lane__test-trigger",
        "trigger_name": "test-trigger",
        "action": {
            "workflow_id": "wf_placeholder",
            "input_template": {"lane": "test", "merge_sha": "deadbeef"},
            "vault_ids": ["vault_1"],
            "workflow_version": None,
        },
        "source": {"schedule": "0 */6 * * *", "timezone": "America/Los_Angeles"},
        "enabled": True,
    },
    "launcher_agent": {
        "name": "__lane__test-agent",
        "model": "claude-sonnet-4-20250514",
        "description": "test agent",
        "tools": [{"name": "bash"}],
        "http_servers": [],
    },
    "launcher_session": {
        "agent_id": "__lane__test-agent",
        "environment_id": "env_abc",
        "title": "Test session",
        "archive_when_idle": False,
        "vault_ids": ["vault_1"],
    },
}


# ── LaneLock.from_dict ───────────────────────────────────────────────────────


class TestLaneLockFromDict:
    """Round-trip parse of a lock-file dict into typed models."""

    def test_parses_minimal_lock(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        assert isinstance(lock, LaneLock)
        assert lock.provenance.builder_version == "0.1.0"
        assert lock.provenance.spec_hash == "abc123"
        assert lock.provenance.resolved_agent_ids == {"my-agent": "ag_123"}

    def test_workflow_fields(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        assert lock.workflow.name == "__lane__test-workflow"
        assert lock.workflow.script == "print('hello')"
        assert lock.workflow.description == "test wf"
        assert lock.workflow.tools == []
        assert lock.workflow.http_servers == []

    def test_cron_trigger_fields(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        ct = lock.cron_trigger
        assert ct.name == "__lane__test-trigger"
        assert ct.trigger_name == "test-trigger"
        assert ct.enabled is True
        assert ct.source.schedule == "0 */6 * * *"
        assert ct.source.timezone == "America/Los_Angeles"
        assert ct.action.workflow_id == "wf_placeholder"
        assert ct.action.input_template == {"lane": "test", "merge_sha": "deadbeef"}
        assert ct.action.vault_ids == ["vault_1"]

    def test_launcher_agent_fields(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        la = lock.launcher_agent
        assert la.name == "__lane__test-agent"
        assert la.model == "claude-sonnet-4-20250514"
        assert la.tools == [{"name": "bash"}]

    def test_launcher_session_fields(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        ls = lock.launcher_session
        assert ls.agent_id == "__lane__test-agent"
        assert ls.environment_id == "env_abc"
        assert ls.title == "Test session"
        assert ls.archive_when_idle is False
        assert ls.vault_ids == ["vault_1"]

    def test_missing_provenance_defaults(self) -> None:
        data = {k: v for k, v in MINIMAL_LOCK_DICT.items() if k != "_provenance"}
        lock = LaneLock.from_dict(data)

        assert lock.provenance.builder_version == ""
        assert lock.provenance.spec_hash == ""
        assert lock.provenance.resolved_agent_ids == {}

    def test_frozen_dataclasses(self) -> None:
        lock = LaneLock.from_dict(MINIMAL_LOCK_DICT)

        with pytest.raises(AttributeError):
            lock.workflow.name = "changed"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            lock.cron_trigger.enabled = False  # type: ignore[misc]


# ── ActivationResult ──────────────────────────────────────────────────────────


class TestActivationResult:
    """ActivationResult.to_dict serialisation."""

    def test_to_dict_round_trip(self) -> None:
        delta = ObjectDelta(
            object_kind="workflow",
            object_name="wf-1",
            action="created",
            object_id="wf_abc",
            new_version=1,
        )
        result = ActivationResult(
            outcome=ActivationOutcome.ACTIVATED,
            lane="test",
            merge_sha="deadbeef1234",
            spec_hash="abc123",
            deltas=[delta],
        )
        d = result.to_dict()

        assert d["outcome"] == "activated"
        assert d["lane"] == "test"
        assert d["merge_sha"] == "deadbeef1234"
        assert len(d["deltas"]) == 1
        assert d["deltas"][0]["action"] == "created"
        assert d["error"] is None

    def test_failed_result(self) -> None:
        result = ActivationResult(
            outcome=ActivationOutcome.FAILED,
            lane="x",
            merge_sha="",
            spec_hash="",
            error="something broke",
        )
        d = result.to_dict()

        assert d["outcome"] == "failed"
        assert d["error"] == "something broke"
        assert d["deltas"] == []

    def test_no_op_result(self) -> None:
        result = ActivationResult(
            outcome=ActivationOutcome.NO_OP,
            lane="y",
            merge_sha="aaa",
            spec_hash="bbb",
        )
        assert result.to_dict()["outcome"] == "no_op"


# ── ActivationOutcome enum ────────────────────────────────────────────────────


class TestActivationOutcome:
    def test_values(self) -> None:
        assert ActivationOutcome.ACTIVATED.value == "activated"
        assert ActivationOutcome.NO_OP.value == "no_op"
        assert ActivationOutcome.FAILED.value == "failed"

    def test_str_mixin(self) -> None:
        # StrEnum members' str() gives the value; verify via .value
        assert str(ActivationOutcome.ACTIVATED) == "activated"
        assert ActivationOutcome.ACTIVATED.value == "activated"
