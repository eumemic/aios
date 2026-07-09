"""Pydantic validation for the workflow request models (Block 3 surface).

Pure in-memory: no Postgres, no Docker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import HttpServerSpec
from aios.models.workflows import GateResume, WfRun, WfRunCreate, WorkflowCreate, WorkflowUpdate


def _http_server(name: str, base_url: str) -> dict[str, str]:
    return {"name": name, "base_url": base_url}


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
            "output_model": None,
            "description": None,
            "tools": [],
            "mcp_servers": [],
            "http_servers": [],
        }

    def test_output_model_round_trips(self) -> None:
        # The declared effective model the ``workflow:`` binding carries (#1637).
        wf = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "output_model": "anthropic/claude-opus-4-6",
            }
        )
        assert wf.output_model == "anthropic/claude-opus-4-6"

    def test_output_model_rejects_empty_string(self) -> None:
        # ``min_length=1``: an empty effective model is meaningless, not a clear.
        with pytest.raises(ValidationError):
            WorkflowCreate.model_validate(
                {"name": "w", "script": "async def main(i): return 1", "output_model": ""}
            )

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
        assert isinstance(wf.http_servers[0], HttpServerSpec)
        assert wf.http_servers[0].base_url == "https://api.example"
        assert wf.tools == []

    def test_requires_name_and_script(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowCreate.model_validate({"name": "w"})
        with pytest.raises(ValidationError):
            WorkflowCreate.model_validate({"script": "x"})

    def test_rejects_duplicate_http_server_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            WorkflowCreate.model_validate(
                {
                    "name": "w",
                    "script": "async def main(i): return 1",
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_rejects_duplicate_tool_key(self) -> None:
        """#1758 scope (1): unique-attenuation-identity applies at every ingress
        edge that carries a ``tools`` list, not just ``AgentCreate``."""
        with pytest.raises(ValidationError, match=r"duplicate tool entry 'bash'"):
            WorkflowCreate.model_validate(
                {
                    "name": "w",
                    "script": "async def main(i): return 1",
                    "tools": [{"type": "bash"}, {"type": "bash"}],
                }
            )

    def test_rejects_duplicate_mcp_server_name(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate mcp server name 'gh'"):
            WorkflowCreate.model_validate(
                {
                    "name": "w",
                    "script": "async def main(i): return 1",
                    "mcp_servers": [
                        {"name": "gh", "url": "https://gh1"},
                        {"name": "gh", "url": "https://gh2"},
                    ],
                }
            )

    def test_accepts_duplicate_http_server_names_with_distinct_base_urls(self) -> None:
        workflow = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert [
            server.base_url
            for server in workflow.http_servers
            if isinstance(server, HttpServerSpec)
        ] == [
            "https://one.example.com",
            "https://two.example.com",
        ]

    def test_accepts_names_only_http_servers(self) -> None:
        # #953 names-only sugar: a bare string references a grant the acting agent
        # holds; it carries no base_url at the model layer (resolved in the service).
        wf = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "http_servers": ["davenant"],
            }
        )
        assert wf.http_servers == ["davenant"]

    def test_accepts_mixed_names_and_specs(self) -> None:
        wf = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "http_servers": ["davenant", _http_server("h", "https://api.example")],
            }
        )
        assert wf.http_servers[0] == "davenant"
        assert isinstance(wf.http_servers[1], HttpServerSpec)
        assert wf.http_servers[1].base_url == "https://api.example"

    def test_bare_names_skip_base_url_uniqueness(self) -> None:
        # The cross-item base_url uniqueness check applies to full specs only — two
        # bare names (no base_url yet) never collide at the model layer.
        wf = WorkflowCreate.model_validate(
            {
                "name": "w",
                "script": "async def main(i): return 1",
                "http_servers": ["davenant", "other"],
            }
        )
        assert wf.http_servers == ["davenant", "other"]


class TestWorkflowUpdate:
    def test_version_required_fields_optional(self) -> None:
        upd = WorkflowUpdate.model_validate({"version": 3, "script": "async def main(i): pass"})
        assert upd.version == 3 and upd.script is not None
        assert upd.name is None and upd.tools is None  # omitted = preserved
        with pytest.raises(ValidationError):
            WorkflowUpdate.model_validate({"script": "x"})  # version is mandatory

    def test_rejects_duplicate_http_server_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            WorkflowUpdate.model_validate(
                {
                    "version": 3,
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_accepts_duplicate_http_server_names_with_distinct_base_urls(self) -> None:
        update = WorkflowUpdate.model_validate(
            {
                "version": 3,
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert update.http_servers is not None
        assert [
            server.base_url for server in update.http_servers if isinstance(server, HttpServerSpec)
        ] == [
            "https://one.example.com",
            "https://two.example.com",
        ]

    def test_accepts_omitted_http_servers(self) -> None:
        update = WorkflowUpdate.model_validate({"version": 3})

        assert update.http_servers is None


class TestWfRunCreate:
    def test_minimal(self) -> None:
        run = WfRunCreate.model_validate({"workflow_id": "wf_1", "environment_id": "env_1"})
        assert run.workflow_id == "wf_1" and run.environment_id == "env_1"
        assert run.input is None
        assert run.vault_ids == []
        assert run.budget_usd is None

    def test_budget_usd_validation(self) -> None:
        run = WfRunCreate.model_validate(
            {"workflow_id": "wf_1", "environment_id": "env_1", "budget_usd": 1.25}
        )
        assert run.budget_usd == 1.25
        with pytest.raises(ValidationError):
            WfRunCreate.model_validate(
                {"workflow_id": "wf_1", "environment_id": "env_1", "budget_usd": 0}
            )
        with pytest.raises(ValidationError):
            WfRunCreate.model_validate(
                {"workflow_id": "wf_1", "environment_id": "env_1", "budget_usd": -1}
            )

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


def test_wf_run_budget_usd_round_trip() -> None:
    run = WfRun(
        id="wfr_1",
        workflow_id="wf_1",
        account_id="acc_1",
        environment_id="env_1",
        script="async def main(input): return None",
        script_sha="sha",
        host_semantics_epoch=1,
        status="pending",
        last_event_seq=0,
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        budget_usd=2.5,
    )
    assert run.budget_usd == 2.5
