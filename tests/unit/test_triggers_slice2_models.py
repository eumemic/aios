"""Pydantic + composer validation for the slice-2 triggers models (#819).

Pure in-memory: no Postgres, no Docker. The jsonb round-trip half of the
write-bound story (a numeric-expanded template must still READ back) is the
e2e read-acceptance regression; here we pin the write-side 422s, the Replace
semantics, the envelope composition, and that the read adapters stay
structure-only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aios.harness.trigger_runner import compose_workflow_run_input
from aios.models.triggers import (
    MAX_INPUT_TEMPLATE_BYTES,
    TRIGGER_ACTION_ADAPTER,
    TRIGGER_SOURCE_ADAPTER,
    RunCompletionSource,
    TriggerCreate,
    TriggerRunEcho,
    TriggerUpdate,
    WorkflowAction,
    compute_initial_next_fire,
)
from aios.models.workflows import WfRun

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)


def _create(**overrides: object) -> TriggerCreate:
    base: dict[str, object] = {
        "name": "reactor",
        "source": {"kind": "run_completion", "workflow_id": "wf_watched"},
        "action": {"kind": "workflow", "workflow_id": "wf_target"},
    }
    base.update(overrides)
    return TriggerCreate.model_validate(base)


def _wf_run(**overrides: Any) -> WfRun:
    base: dict[str, Any] = {
        "id": "wfr_done",
        "workflow_id": "wf_watched",
        "account_id": "acc_t",
        "environment_id": "env_t",
        "script": "async def main(input): ...",
        "script_sha": "x" * 64,
        "host_semantics_epoch": 1,
        "status": "completed",
        "output": {"rows": 3},
        "last_event_seq": 2,
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return WfRun.model_validate(base)


class TestRunCompletionSource:
    def test_statuses_default_all_three_materialized(self) -> None:
        """Omitted ``statuses`` defaults to all terminal statuses AND the
        default is materialized into the stored spec (the CHECK requires the
        key; the matcher carries no defaults knowledge)."""
        spec = _create()
        assert isinstance(spec.source, RunCompletionSource)
        assert spec.source.statuses == ["completed", "errored", "cancelled"]
        dumped = spec.source.model_dump(mode="json", exclude={"kind"})
        assert dumped == {
            "workflow_id": "wf_watched",
            "statuses": ["completed", "errored", "cancelled"],
        }

    def test_statuses_narrowed_and_validated(self) -> None:
        spec = _create(
            source={"kind": "run_completion", "workflow_id": "wf_w", "statuses": ["errored"]}
        )
        assert isinstance(spec.source, RunCompletionSource)
        assert spec.source.statuses == ["errored"]
        with pytest.raises(ValidationError):
            _create(source={"kind": "run_completion", "workflow_id": "wf_w", "statuses": []})
        with pytest.raises(ValidationError):
            _create(
                source={"kind": "run_completion", "workflow_id": "wf_w", "statuses": ["running"]}
            )

    def test_update_replace_requires_statuses(self) -> None:
        """§2.2 Replace rule: a partial run_completion source on UPDATE 422s
        instead of silently resetting a narrowed filter back to all-three."""
        with pytest.raises(ValidationError, match="statuses"):
            TriggerUpdate.model_validate(
                {"source": {"kind": "run_completion", "workflow_id": "wf_w"}}
            )
        ok = TriggerUpdate.model_validate(
            {
                "source": {
                    "kind": "run_completion",
                    "workflow_id": "wf_w",
                    "statuses": ["errored"],
                }
            }
        )
        assert ok.source is not None

    def test_read_adapter_accepts_persisted_shape(self) -> None:
        src = TRIGGER_SOURCE_ADAPTER.validate_python(
            {"kind": "run_completion", "workflow_id": "wf_w", "statuses": ["completed"]}
        )
        assert isinstance(src, RunCompletionSource)


class TestWorkflowAction:
    def test_create_defaults_materialized(self) -> None:
        """Float pin, null template, empty vaults — all materialized into the
        stored jsonb so the CHECK can require every key."""
        spec = _create()
        assert isinstance(spec.action, WorkflowAction)
        assert spec.action.model_dump(mode="json") == {
            "kind": "workflow",
            "workflow_id": "wf_target",
            "workflow_version": None,
            "input_template": None,
            "vault_ids": [],
        }

    def test_no_environment_id_field(self) -> None:
        """environment_id is a column, never a wire field — anything on the
        union member is agent-reachable through the trigger_create tool."""
        with pytest.raises(ValidationError):
            _create(action={"kind": "workflow", "workflow_id": "wf_t", "environment_id": "env_x"})

    def test_update_replace_requires_all_optional_fields(self) -> None:
        """A partial workflow action on UPDATE 422s instead of silently
        flipping a pin to float / nulling the template / dropping vaults."""
        with pytest.raises(ValidationError):
            TriggerUpdate.model_validate({"action": {"kind": "workflow", "workflow_id": "wf_t"}})
        ok = TriggerUpdate.model_validate(
            {
                "action": {
                    "kind": "workflow",
                    "workflow_id": "wf_t",
                    "workflow_version": None,
                    "input_template": None,
                    "vault_ids": [],
                }
            }
        )
        assert ok.action is not None

    def test_pin_must_be_positive_int(self) -> None:
        with pytest.raises(ValidationError):
            _create(action={"kind": "workflow", "workflow_id": "wf_t", "workflow_version": 0})


class TestInputTemplateWriteBound:
    def test_oversize_template_422s_on_write_models(self) -> None:
        big = {"blob": "x" * MAX_INPUT_TEMPLATE_BYTES}
        with pytest.raises(ValidationError, match="input_template serializes"):
            _create(action={"kind": "workflow", "workflow_id": "wf_t", "input_template": big})
        with pytest.raises(ValidationError, match="input_template serializes"):
            TriggerUpdate.model_validate(
                {
                    "action": {
                        "kind": "workflow",
                        "workflow_id": "wf_t",
                        "workflow_version": None,
                        "input_template": big,
                        "vault_ids": [],
                    }
                }
            )

    def test_read_adapter_is_unbounded(self) -> None:
        """The bound is WRITE-PATH ONLY: byte bounds are not jsonb-round-trip
        stable (numeric expansion), and the action adapter runs inside the
        scheduler's claim transaction — a read-side bound would let one row
        halt every trigger on the deployment."""
        oversize: dict[str, Any] = {
            "kind": "workflow",
            "workflow_id": "wf_t",
            "workflow_version": None,
            "input_template": {"blob": "x" * (MAX_INPUT_TEMPLATE_BYTES * 2)},
            "vault_ids": [],
        }
        action = TRIGGER_ACTION_ADAPTER.validate_python(oversize)
        assert isinstance(action, WorkflowAction)

    def test_nan_and_infinity_rejected_at_write(self) -> None:
        """allow_nan=False measures exactly what jsonb will accept — a 422
        here instead of a 500 at INSERT."""
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                _create(
                    action={
                        "kind": "workflow",
                        "workflow_id": "wf_t",
                        "input_template": {"v": bad},
                    }
                )


class TestComputeInitialNextFireRunCompletion:
    def test_none_for_run_completion(self) -> None:
        src = RunCompletionSource(workflow_id="wf_w")
        assert compute_initial_next_fire(src, NOW) is None


class TestComposeWorkflowRunInput:
    def test_timer_fire_has_no_run_key(self) -> None:
        composed = compose_workflow_run_input(
            trigger_id="trig_1",
            trigger_name="nightly",
            source="cron",
            fired_at=NOW,
            input_template={"report": "daily"},
        )
        assert composed == {
            "trigger": {
                "id": "trig_1",
                "name": "nightly",
                "source": "cron",
                "fired_at": NOW.isoformat(),
            },
            "input": {"report": "daily"},
        }

    def test_completed_watch_carries_output_by_value(self) -> None:
        composed = compose_workflow_run_input(
            trigger_id="trig_1",
            trigger_name="reactor",
            source="run_completion",
            fired_at=NOW,
            input_template=None,
            completed_run=_wf_run(),
            completed_error=None,
        )
        assert composed["trigger"]["run"] == {
            "id": "wfr_done",
            "workflow_id": "wf_watched",
            "status": "completed",
            "output": {"rows": 3},
            "error": None,
        }
        assert composed["input"] is None

    def test_errored_watch_carries_error_kind_and_null_output(self) -> None:
        composed = compose_workflow_run_input(
            trigger_id="trig_1",
            trigger_name="reactor",
            source="run_completion",
            fired_at=NOW,
            input_template={"notify": "#ops"},
            completed_run=_wf_run(status="errored", output=None),
            completed_error={"kind": "script_error"},
        )
        run = composed["trigger"]["run"]
        assert run["status"] == "errored"
        assert run["output"] is None
        assert run["error"] == {"kind": "script_error"}

    def test_template_with_trigger_key_nests_harmlessly(self) -> None:
        """The author's shape is never parsed or mutated — a template carrying
        its own 'trigger' key just nests under 'input' (no clobber, no alias)."""
        template = {"trigger": "my own key", "x": 1}
        composed = compose_workflow_run_input(
            trigger_id="trig_1",
            trigger_name="t",
            source="one_shot",
            fired_at=NOW,
            input_template=template,
        )
        assert composed["input"] == template
        assert composed["trigger"]["source"] == "one_shot"
        # Pure dict composition — JSON-serializable end to end.
        json.dumps(composed)


class TestTriggerRunEcho:
    def test_open_status_string_on_read(self) -> None:
        """Rows written by future writers must always read back — status is an
        open str, not a Literal."""
        echo = TriggerRunEcho.model_validate(
            {
                "id": "trun_1",
                "trigger_id": "trig_1",
                "trigger_context": "run_completion",
                "event": {"run_id": "wfr_1", "workflow_id": "wf_w", "status": "completed"},
                "status": "some_future_status",
                "result_id": None,
                "error_summary": None,
                "created_at": NOW,
                "started_at": None,
                "finished_at": None,
            }
        )
        assert echo.status == "some_future_status"
        assert echo.event is not None and echo.event["run_id"] == "wfr_1"
