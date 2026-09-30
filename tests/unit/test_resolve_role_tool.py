"""Unit tests for the ``resolve_role`` builtin (#1940).

Role resolution is the v0 addressing primitive: a caller asks for a ROLE and gets
the LIVE agent id, instead of caching an id that goes stale on re-spawn. These
tests stub the worker pool + service so they need no Postgres; the SQL match
semantics (name / ``metadata.role``, archived exclusion, account scoping) are
covered by ``tests/integration/test_resolve_role.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.errors import ConflictError, NotFoundError
from aios.models.agents import Agent, AgentCreate
from aios.services import agents as agents_service
from aios.tools import agent_management as am
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import openai_tool_entry, registry

_DT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acc_x")
    )


def _agent(**over: Any) -> Agent:
    base: dict[str, Any] = dict(
        id="agt_live",
        version=4,
        name="ops-agent",
        model="test/dummy",
        system="SECRET-SYSTEM",
        tools=[],
        skills=[],
        mcp_servers=[],
        http_servers=[],
        description="ops",
        metadata={"role": "ops"},
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Agent(**base)


class TestRegistration:
    def test_registered_as_agent_tool_with_closed_schema(self) -> None:
        tool = registry.get("resolve_role")
        assert tool.transport == "agent_tool"
        params = openai_tool_entry(tool)["function"]["parameters"]
        assert params.get("additionalProperties") is False
        assert params["required"] == ["role"]

    def test_grantable_in_agent_definition(self) -> None:
        agent = AgentCreate.model_validate(
            {"name": "caller", "model": "test/dummy", "tools": [{"type": "resolve_role"}]}
        )
        assert agent.tools[0].type == "resolve_role"


class TestArguments:
    async def test_account_id_injection_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "resolve_role", {"role": "ops", "account_id": "acc_victim"}
            )

    async def test_empty_role_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin("ses_1", "resolve_role", {"role": ""})


class TestResolution:
    async def test_returns_live_agent_id_with_account_derived_server_side(
        self, monkeypatch: Any
    ) -> None:
        mock = AsyncMock(return_value=_agent())
        monkeypatch.setattr("aios.services.agents.resolve_role", mock)
        out = await am.resolve_role_handler("ses_exec", {"role": "ops"})
        assert out["role"] == "ops"
        assert out["agent_id"] == "agt_live"
        assert out["agent"]["id"] == "agt_live"
        assert out["agent"]["name"] == "ops-agent"
        assert out["agent"]["version"] == 4
        # Lean summary, like list_agents — no system prompt / surface bodies.
        for heavy in ("system", "tools", "mcp_servers", "http_servers", "metadata"):
            assert heavy not in out["agent"]
        assert mock.await_args is not None
        assert mock.await_args.args[1] == "ops"
        assert mock.await_args.kwargs["account_id"] == "acc_x"

    async def test_unbound_role_is_loud_not_found(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.resolve_role",
            AsyncMock(side_effect=NotFoundError("no live binding for role 'ops'")),
        )
        with pytest.raises(NotFoundError, match="no live binding for role 'ops'"):
            await am.resolve_role_handler("ses_1", {"role": "ops"})

    async def test_ambiguous_role_is_loud_conflict(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.resolve_role",
            AsyncMock(side_effect=ConflictError("role 'ops' is ambiguous")),
        )
        with pytest.raises(ConflictError, match="ambiguous"):
            await am.resolve_role_handler("ses_1", {"role": "ops"})


class TestServiceErrors:
    """The service turns the candidate set into exactly-one-or-a-loud-error."""

    async def test_zero_candidates_raises_not_found_naming_the_role(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.queries.list_agents_for_role", AsyncMock(return_value=[])
        )
        with pytest.raises(NotFoundError, match="no live binding for role 'ops'") as exc:
            await agents_service.resolve_role(_FakePool(), "ops", account_id="acc_x")
        assert exc.value.detail == {"role": "ops"}

    async def test_multiple_candidates_raises_conflict_listing_them(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.queries.list_agents_for_role",
            AsyncMock(
                return_value=[_agent(id="agt_b", name="ops"), _agent(id="agt_a", name="other")]
            ),
        )
        with pytest.raises(ConflictError, match="ambiguous") as exc:
            await agents_service.resolve_role(_FakePool(), "ops", account_id="acc_x")
        assert exc.value.detail == {"role": "ops", "agent_ids": ["agt_b", "agt_a"]}

    async def test_single_candidate_is_returned(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.agents.queries.list_agents_for_role",
            AsyncMock(return_value=[_agent()]),
        )
        agent = await agents_service.resolve_role(_FakePool(), "ops", account_id="acc_x")
        assert agent.id == "agt_live"


class _Conn:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakePool:
    def acquire(self) -> _Conn:
        return _Conn()
