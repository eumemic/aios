"""The operator trigger request models (#2473) admit only a timer source and a
budgeted workflow action, so the API rejects anything else with a 422."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aios.models.triggers import OperatorTriggerCreate, OperatorTriggerUpdate

_ACTION = {"kind": "workflow", "workflow_id": "wf_x", "budget_usd": 1.0}


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "weekly",
        "source": {"kind": "cron", "schedule": "0 9 * * 1"},
        "action": _ACTION,
        "environment_id": "env_x",
    }
    body.update(over)
    return body


def test_a_cron_or_one_shot_workflow_trigger_is_accepted() -> None:
    OperatorTriggerCreate.model_validate(_body())
    OperatorTriggerCreate.model_validate(
        _body(source={"kind": "one_shot", "fire_at": "2030-01-01T00:00:00Z"})
    )


@pytest.mark.parametrize(
    "over",
    [
        pytest.param(
            {"source": {"kind": "run_completion", "workflow_id": "wf_y"}}, id="run_completion"
        ),
        pytest.param({"source": {"kind": "external_event"}}, id="external_event"),
        pytest.param({"action": {"kind": "workflow", "workflow_id": "wf_x"}}, id="no_budget"),
        pytest.param({"action": {**_ACTION, "budget_usd": 0}}, id="zero_budget"),
        pytest.param({"action": {"kind": "wake_owner", "content": "go"}}, id="wake_owner"),
        pytest.param(
            {"action": {"kind": "sandbox_command", "command": "true"}}, id="sandbox_command"
        ),
        pytest.param({"environment_id": None}, id="no_environment"),
        pytest.param({"name": "ingest"}, id="reserved_name"),
        pytest.param({"source": {"kind": "cron", "schedule": "not a cron"}}, id="bad_cron"),
    ],
)
def test_anything_else_is_rejected(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OperatorTriggerCreate.model_validate(_body(**over))


def test_an_update_keeps_the_operator_shapes() -> None:
    with pytest.raises(ValidationError):
        OperatorTriggerUpdate.model_validate({"source": {"kind": "external_event"}})
    with pytest.raises(ValidationError):
        OperatorTriggerUpdate.model_validate(
            {
                "action": {
                    "kind": "workflow",
                    "workflow_id": "wf_x",
                    "workflow_version": None,
                    "version": None,
                    "input_template": None,
                    "vault_ids": [],
                    "max_outstanding_runs": None,
                    "budget_usd": None,
                }
            }
        )
