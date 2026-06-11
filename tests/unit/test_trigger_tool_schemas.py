"""Schema↔model drift guards for the trigger tools' slice-2 oneOf branches.

The handlers validate with the shared Pydantic models, so the JSON schema is
documentation the model sees — these tests pin that it tells the truth about
the Replace semantics (update side requires what create defaults) and never
exposes ``environment_id`` (a column, deliberately not a wire field — anything
on the tool surface is agent-reachable).
"""

from __future__ import annotations

from typing import Any

from aios.tools.trigger_create import (
    TRIGGER_CREATE_PARAMETERS_SCHEMA,
)
from aios.tools.trigger_update import (
    TRIGGER_UPDATE_PARAMETERS_SCHEMA,
)


def _branch(schema: dict[str, Any], field: str, kind: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = schema["properties"][field]["oneOf"]
    for candidate in candidates:
        if candidate["properties"]["kind"]["const"] == kind:
            return candidate
    raise AssertionError(f"no {kind!r} branch on {field}")


class TestCreateSchemaBranches:
    def test_run_completion_branch(self) -> None:
        branch = _branch(TRIGGER_CREATE_PARAMETERS_SCHEMA, "source", "run_completion")
        assert branch["required"] == ["kind", "workflow_id"]
        statuses = branch["properties"]["statuses"]
        assert statuses["items"]["enum"] == ["completed", "errored", "cancelled"]
        assert statuses["default"] == ["completed", "errored", "cancelled"]
        assert branch["additionalProperties"] is False

    def test_workflow_branch(self) -> None:
        branch = _branch(TRIGGER_CREATE_PARAMETERS_SCHEMA, "action", "workflow")
        assert branch["required"] == ["kind", "workflow_id"]
        assert branch["properties"]["workflow_version"]["type"] == ["integer", "null"]
        assert "environment_id" not in branch["properties"]
        assert branch["additionalProperties"] is False
        # input_template is deliberately schemaless (any JSON type, incl. null).
        assert "type" not in branch["properties"]["input_template"]

    def test_existing_branches_untouched(self) -> None:
        assert [
            b["properties"]["kind"]["const"]
            for b in TRIGGER_CREATE_PARAMETERS_SCHEMA["properties"]["source"]["oneOf"]
        ] == ["cron", "one_shot", "run_completion"]
        assert [
            b["properties"]["kind"]["const"]
            for b in TRIGGER_CREATE_PARAMETERS_SCHEMA["properties"]["action"]["oneOf"]
        ] == ["sandbox_command", "wake_owner", "workflow"]


class TestUpdateSchemaReplaceSemantics:
    def test_run_completion_requires_statuses(self) -> None:
        branch = _branch(TRIGGER_UPDATE_PARAMETERS_SCHEMA, "source", "run_completion")
        assert branch["required"] == ["kind", "workflow_id", "statuses"]
        assert "default" not in branch["properties"]["statuses"]

    def test_workflow_requires_all_optional_at_create_fields(self) -> None:
        branch = _branch(TRIGGER_UPDATE_PARAMETERS_SCHEMA, "action", "workflow")
        assert branch["required"] == [
            "kind",
            "workflow_id",
            "workflow_version",
            "input_template",
            "vault_ids",
        ]
        assert "environment_id" not in branch["properties"]
