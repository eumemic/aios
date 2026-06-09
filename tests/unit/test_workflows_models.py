"""Pydantic validation for the workflow request models (Block 3 surface).

Pure in-memory: no Postgres, no Docker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.workflows import GateResume, WfRunCreate, WorkflowCreate


class TestWorkflowCreate:
    def test_minimal(self) -> None:
        wf = WorkflowCreate.model_validate(
            {"name": "echo", "script": "async def main(i): return i"}
        )
        assert wf.name == "echo"
        assert wf.input_schema is None and wf.output_schema is None

    def test_with_schemas_round_trips(self) -> None:
        payload = {
            "name": "w",
            "script": "async def main(i): return 1",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "integer"},
        }
        assert WorkflowCreate.model_validate(payload).model_dump() == {
            **payload,
            "input_schema": {"type": "object"},
            "output_schema": {"type": "integer"},
            "description": None,
        }

    def test_requires_name_and_script(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowCreate.model_validate({"name": "w"})
        with pytest.raises(ValidationError):
            WorkflowCreate.model_validate({"script": "x"})


class TestWfRunCreate:
    def test_minimal(self) -> None:
        run = WfRunCreate.model_validate({"workflow_id": "wf_1", "environment_id": "env_1"})
        assert run.workflow_id == "wf_1" and run.environment_id == "env_1"
        assert run.input is None

    def test_requires_workflow_and_environment(self) -> None:
        with pytest.raises(ValidationError):
            WfRunCreate.model_validate({"workflow_id": "wf_1"})
        with pytest.raises(ValidationError):
            WfRunCreate.model_validate({"environment_id": "env_1"})


class TestGateResume:
    def test_minimal(self) -> None:
        g = GateResume.model_validate({"gate_nonce": "abc"})
        assert g.gate_nonce == "abc" and g.result is None

    def test_requires_nonce(self) -> None:
        with pytest.raises(ValidationError):
            GateResume.model_validate({"result": 1})
