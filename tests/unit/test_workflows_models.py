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
            "tools": [],
            "mcp_servers": [],
            "http_servers": [],
        }

    def test_declared_surface_round_trips(self) -> None:
        wf = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "mcp_servers": [{"name": "s", "url": "https://srv.example"}],
                "http_servers": [{"name": "h", "base_url": "https://api.example"}],
            }
        )
        assert wf.mcp_servers[0].url == "https://srv.example"
        assert wf.http_servers[0].base_url == "https://api.example"
        assert wf.tools == []

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
        assert run.vault_ids == []

    def test_vault_ids_round_trip(self) -> None:
        run = WfRunCreate.model_validate(
            {"workflow_id": "wf_1", "environment_id": "env_1", "vault_ids": ["v_1", "v_2"]}
        )
        assert run.vault_ids == ["v_1", "v_2"]

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
