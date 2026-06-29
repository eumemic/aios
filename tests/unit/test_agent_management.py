"""Unit tests for the agent-acting agent builtins (T1, #1470).

These stub the worker pool + services so they need no live Postgres. They cover the
identity-load-bearing invariants that don't depend on the DB:

* **F1** — a trusted id smuggled into the arguments (``creator_session_id`` /
  ``editor_session_id`` / ``account_id``) is rejected by the tool schema
  (``additionalProperties: false``) before the handler ever runs.
* **F2** — ``account_id`` is derived server-side from the executing session id, and the
  creator/editor identity passed to the service is the harness-supplied ``session_id``,
  never an argument.
* **Registration** — all five builtins register as ``agent_tool`` transport with a
  closed (``additionalProperties: false``) wire schema.
* **Error handling** — a bad-argument pydantic error bails locally as a ``ToolBail``;
  a service ``AiosError`` (e.g. the surface-attenuation ``ForbiddenError``, a duplicate
  ``ConflictError``) propagates unconverted for the dispatch layer to classify.
* **Return shape** — list_agents trims heavy fields; get_agent returns the full body.

Full surface-attenuation / spawn-edge behavior is covered by the integration test
(``tests/integration/test_agent_management_attenuation.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.errors import ConflictError, CryptoDecryptError, ForbiddenError
from aios.models.agents import Agent
from aios.services import agents as agents_service
from aios.tools import agent_management as am
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import openai_tool_entry, registry

_DT = datetime(2026, 1, 1, tzinfo=UTC)

_AGENT_TOOLS = ("create_agent", "update_agent", "archive_agent", "get_agent", "list_agents")


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    """Make ``require_pool`` + ``load_session_account_id`` cheap (no DB)."""
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acc_x")
    )


def _agent(**over: Any) -> Agent:
    base: dict[str, Any] = dict(
        id="agt_1",
        version=1,
        name="a",
        model="test/dummy",
        system="SECRET-SYSTEM",
        tools=[],
        skills=[],
        mcp_servers=[],
        http_servers=[],
        description="d",
        metadata={},
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Agent(**base)


class TestRegistration:
    def test_all_registered_as_agent_tool(self) -> None:
        for name in _AGENT_TOOLS:
            assert registry.get(name).transport == "agent_tool", name

    def test_wire_schema_is_closed(self) -> None:
        # F1 rides additionalProperties:false — a trusted-id key is rejected up front.
        for name in _AGENT_TOOLS:
            params = openai_tool_entry(registry.get(name))["function"]["parameters"]
            assert params.get("additionalProperties") is False, name


class TestModelRoutingWithheld:
    """#823 tripwire: the model-routing override (``litellm_extra`` → ``api_base``) is
    not authorable from inside a session — neither visible in the schema nor accepted on
    input. The spawn-edge clamp stays the independent backstop; this withholds it on the
    authoring plane so no self-authoring agent can MINT a redirect."""

    def test_litellm_extra_absent_from_create_and_update_schemas(self) -> None:
        import json

        for name in ("create_agent", "update_agent"):
            schema = json.dumps(registry.get(name).parameters_schema)
            assert "litellm_extra" not in schema, name

    async def test_create_rejects_litellm_extra_value(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "create_agent",
                {
                    "name": "a",
                    "model": "test/dummy",
                    "litellm_extra": {"api_base": "https://evil.example"},
                },
            )

    async def test_update_rejects_litellm_extra_value(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "update_agent",
                {
                    "agent_id": "agt_1",
                    "version": 1,
                    "litellm_extra": {"api_base": "https://evil.example"},
                },
            )


class TestSchemaRejectsInjectedTrustedIds:
    """F1: the trusted ids are never schema fields, so injection is rejected up front."""

    async def test_create_agent_creator_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "create_agent",
                {"name": "a", "model": "test/dummy", "creator_session_id": "ses_victim"},
            )

    async def test_create_agent_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "create_agent",
                {"name": "a", "model": "test/dummy", "account_id": "acc_victim"},
            )

    async def test_update_agent_editor_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "update_agent",
                {"agent_id": "agt_1", "version": 1, "editor_session_id": "ses_victim"},
            )

    async def test_archive_agent_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "archive_agent", {"agent_id": "agt_1", "account_id": "acc_victim"}
            )

    async def test_get_agent_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "get_agent", {"agent_id": "agt_1", "account_id": "acc_victim"}
            )

    async def test_list_agents_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin("ses_1", "list_agents", {"account_id": "acc_victim"})


class TestTrustedIdsDerivedServerSide:
    """F2: account_id derives from the session; the creator/editor identity is the
    executing session_id, never an argument."""

    async def test_create_passes_session_as_creator_and_derived_account(
        self, monkeypatch: Any
    ) -> None:
        mock_create = AsyncMock(return_value=_agent())
        monkeypatch.setattr("aios.services.agents.create_agent", mock_create)
        await am.create_agent_handler("ses_exec", {"name": "a", "model": "test/dummy"})
        assert mock_create.await_args is not None
        kwargs = mock_create.await_args.kwargs
        assert kwargs["creator_session_id"] == "ses_exec"
        assert kwargs["account_id"] == "acc_x"

    async def test_update_passes_session_as_editor_and_version(self, monkeypatch: Any) -> None:
        mock_update = AsyncMock(return_value=_agent(version=2))
        monkeypatch.setattr("aios.services.agents.update_agent", mock_update)
        await am.update_agent_handler("ses_exec", {"agent_id": "agt_1", "version": 1})
        assert mock_update.await_args is not None
        kwargs = mock_update.await_args.kwargs
        assert kwargs["editor_session_id"] == "ses_exec"
        assert kwargs["expected_version"] == 1
        assert kwargs["account_id"] == "acc_x"
        # update_agent(pool, agent_id, ...): pool is args[0], agent_id is args[1].
        assert mock_update.await_args.args[1] == "agt_1"


class TestErrorPropagation:
    async def test_pydantic_semantic_error_becomes_toolbail(self) -> None:
        # An mcp server named after a builtin passes the JSON schema but fails the
        # McpServerSpec custom validator — tool_input surfaces it as a ToolBail.
        with pytest.raises(ToolBail, match="invalid arguments"):
            await am.create_agent_handler(
                "ses_1",
                {
                    "name": "a",
                    "model": "test/dummy",
                    "mcp_servers": [{"name": "bash", "url": "https://x"}],
                },
            )

    async def test_forbidden_surface_error_propagates(self, monkeypatch: Any) -> None:
        # The create-time surface-attenuation breach surfaces as a ForbiddenError the
        # handler does NOT convert — the dispatch layer turns it into a clean
        # model-visible result.
        monkeypatch.setattr(
            "aios.services.agents.create_agent",
            AsyncMock(side_effect=ForbiddenError("nope", detail={"exceeds": {"tools": ["bash"]}})),
        )
        with pytest.raises(ForbiddenError, match="nope"):
            await am.create_agent_handler("ses_1", {"name": "a", "model": "test/dummy"})

    async def test_duplicate_name_conflict_propagates(self, monkeypatch: Any) -> None:
        # create_agent 409s on a duplicate (account_id, name) — the model branches
        # create-vs-update rather than introducing an agent_upsert.
        monkeypatch.setattr(
            "aios.services.agents.create_agent",
            AsyncMock(side_effect=ConflictError("duplicate name")),
        )
        with pytest.raises(ConflictError):
            await am.create_agent_handler("ses_1", {"name": "a", "model": "test/dummy"})

    async def test_internal_5xx_propagates(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.create_agent",
            AsyncMock(side_effect=CryptoDecryptError("boom")),
        )
        with pytest.raises(CryptoDecryptError):
            await am.create_agent_handler("ses_1", {"name": "a", "model": "test/dummy"})


class TestWorkflowModelBindingPrivilege:
    """#1636: a self-authoring (non-operator) principal — the create/update_agent
    model-tool path always carries a ``creator_session_id`` / ``editor_session_id``
    — may NOT bind a ``workflow:`` model. The guard fires BEFORE any DB call, so it
    exercises the real service with a ``None`` pool.

    The companion spawn-edge arms (per-call ``agent(model=…)`` override + the generic
    agentless child) live in the workflow-step integration suite, where a run row and
    its owning principal exist.
    """

    async def test_create_agent_self_authoring_workflow_model_forbidden(self) -> None:
        # The real service: a creator_session_id is set, so the binding privilege
        # rejects a workflow: model before _enforce_surface_attenuation hits the DB.
        with pytest.raises(ForbiddenError, match="operator-only"):
            await agents_service.create_agent(
                None,  # type: ignore[arg-type]  # guard fires before the pool is touched
                account_id="acc_x",
                name="a",
                model="workflow:wf_1",
                system="s",
                tools=[],
                description=None,
                metadata={},
                window_min=1000,
                window_max=100000,
                creator_session_id="ses_self",
            )

    async def test_update_agent_self_authoring_workflow_model_forbidden(self) -> None:
        with pytest.raises(ForbiddenError, match="operator-only"):
            await agents_service.update_agent(
                None,  # type: ignore[arg-type]
                "agt_1",
                account_id="acc_x",
                expected_version=1,
                model="workflow:wf_1@2",
                editor_session_id="ses_self",
            )

    async def test_create_agent_self_authoring_raw_model_passes_guard(
        self, monkeypatch: Any
    ) -> None:
        # A raw provider model is admissible for a self-authoring principal — the
        # guard is a no-op and the flow proceeds to the surface-attenuation clamp
        # (stubbed here to a no-op) and the insert (stubbed to return an agent).
        monkeypatch.setattr(
            "aios.services.agents._enforce_surface_attenuation", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            "aios.services.agents.skills_service.resolve_skill_refs",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "aios.services.agents.skills_service.serialize_skills_for_snapshot",
            lambda *a, **k: "[]",
        )

        class _FakePool:
            def acquire(self) -> Any:
                raise AssertionError("reached the insert — guard admitted the raw model")

        monkeypatch.setattr(
            "aios.services.agents.queries.insert_agent", AsyncMock(return_value=_agent())
        )
        # The guard must NOT raise; the AssertionError below proves we passed it.
        with pytest.raises(AssertionError, match="reached the insert"):
            await agents_service.create_agent(
                _FakePool(),  # type: ignore[arg-type]
                account_id="acc_x",
                name="a",
                model="test/dummy",
                system="s",
                tools=[],
                description=None,
                metadata={},
                window_min=1000,
                window_max=100000,
                creator_session_id="ses_self",
            )


class TestReturnShape:
    async def test_get_agent_returns_full_body(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.get_agent", AsyncMock(return_value=_agent(version=3))
        )
        out = await am.get_agent_handler("ses_1", {"agent_id": "agt_1"})
        # FULL — the re-read loop before an update_agent retry needs system + version.
        assert out["version"] == 3
        assert out["system"] == "SECRET-SYSTEM"

    async def test_list_agents_trims_heavy_fields(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("aios.services.agents.list_agents", AsyncMock(return_value=[_agent()]))
        out = await am.list_agents_handler("ses_1", {})
        assert len(out["agents"]) == 1
        summary = out["agents"][0]
        assert summary["name"] == "a" and summary["version"] == 1
        for heavy in ("system", "tools", "mcp_servers", "http_servers", "metadata"):
            assert heavy not in summary

    async def test_archive_agent_returns_archived_flag(self, monkeypatch: Any) -> None:
        mock_archive = AsyncMock(return_value=None)
        monkeypatch.setattr("aios.services.agents.archive_agent", mock_archive)
        out = await am.archive_agent_handler("ses_1", {"agent_id": "agt_1"})
        assert out == {"agent_id": "agt_1", "archived": True}
        assert mock_archive.await_args is not None
        assert mock_archive.await_args.kwargs["account_id"] == "acc_x"
