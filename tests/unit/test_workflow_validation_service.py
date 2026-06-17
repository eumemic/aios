"""Service-path tests for create-time workflow validation (#1284).

Exercises ``aios.services.workflows.create_workflow`` / ``update_workflow`` (the
service path required by acceptance criterion 8) with a stubbed pool + query layer
— no live Postgres. Covers:

  * the validator runs and the workflow is **not** created on a rejection
    (``insert_workflow`` is never called),
  * criterion 5 — a string-literal ``agent(agent_id="A")`` whose required surface is
    not covered by the declared surface is rejected, naming the missing element,
  * the agent-surface check does NOT false-reject when the named agent does not
    resolve (authoring may precede the agent's creation),
  * criteria 1-7 are enforced on **update** as well as create (merged script/tools).

The tool-handler path (``create_workflow_handler``) calls straight through to this
service, so enforcing it here enforces it there too (see ``test_workflow_management``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError, ValidationError
from aios.models.agents import Agent, ToolSpec
from aios.models.workflows import Workflow
from aios.services import agents as agents_service
from aios.services import workflows as service

_DT = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeConn:
    pass


class _FakePool:
    """Minimal async-context ``acquire()`` shim — the service only needs the conn
    handle to thread into the (stubbed) query functions."""

    def acquire(self) -> Any:
        conn = _FakeConn()

        class _Ctx:
            async def __aenter__(self_inner) -> _FakeConn:
                return conn

            async def __aexit__(self_inner, *exc: Any) -> bool:
                return False

        return _Ctx()


def _workflow(**over: Any) -> Workflow:
    base: dict[str, Any] = dict(
        id="wf_1",
        account_id="acc_x",
        name="w",
        version=1,
        script="async def main(input):\n    return input\n",
        tools=[],
        mcp_servers=[],
        http_servers=[],
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Workflow(**base)


def _agent(tools: list[ToolSpec]) -> Agent:
    return Agent(
        id="ag_1",
        version=1,
        name="researcher",
        model="m",
        system="s",
        tools=tools,
        mcp_servers=[],
        http_servers=[],
        description=None,
        metadata={},
        window_min=1,
        window_max=2,
        created_at=_DT,
        updated_at=_DT,
    )


@pytest.fixture
def insert_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture ``insert_workflow`` calls; a rejected create must record none."""
    calls: list[dict[str, Any]] = []

    async def _insert(conn: Any, **kw: Any) -> Workflow:
        calls.append(kw)
        return _workflow(**{k: v for k, v in kw.items() if k in {"name", "script", "tools"}})

    monkeypatch.setattr(wf_queries, "insert_workflow", _insert)
    return calls


@pytest.fixture
def update_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _update(conn: Any, workflow_id: str, **kw: Any) -> Workflow:
        calls.append({"workflow_id": workflow_id, **kw})
        return _workflow(version=2)

    monkeypatch.setattr(wf_queries, "update_workflow", _update)
    return calls


_GOOD = "async def main(input):\n    return input\n"


class TestCreatePathValidation:
    async def test_valid_operator_create_succeeds(self, insert_calls: list[Any]) -> None:
        wf = await service.create_workflow(
            _FakePool(),
            account_id="acc_x",
            name="w",
            script="async def main(input):\n    return await tool('bash', {})\n",
            tools=[ToolSpec(type="bash")],
        )
        assert wf.name == "w"
        assert len(insert_calls) == 1

    async def test_syntax_error_not_created(self, insert_calls: list[Any]) -> None:
        with pytest.raises(ValidationError):
            await service.create_workflow(
                _FakePool(),
                account_id="acc_x",
                name="w",
                script="async def main(input):\n    x = (\n",
            )
        assert insert_calls == []  # workflow not created

    async def test_missing_main_not_created(self, insert_calls: list[Any]) -> None:
        with pytest.raises(ValidationError, match="main"):
            await service.create_workflow(
                _FakePool(),
                account_id="acc_x",
                name="w",
                script="x = 1\n",
            )
        assert insert_calls == []

    async def test_under_declared_tool_not_created(self, insert_calls: list[Any]) -> None:
        with pytest.raises(ValidationError, match="http_request"):
            await service.create_workflow(
                _FakePool(),
                account_id="acc_x",
                name="w",
                script="async def main(input):\n    return await tool('http_request', {})\n",
                tools=[],
            )
        assert insert_calls == []


class TestAgentSurface:
    """Criterion 5 — string-literal ``agent(agent_id="A")`` surface coverage."""

    async def test_under_declared_agent_surface_rejected(
        self, insert_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The named agent needs 'http_request' but the workflow declares no tools →
        # the #794 clamp would silently strip it. Rejected, naming the missing element.
        async def _get_agent(pool: Any, agent_id: str, *, account_id: str) -> Agent:
            assert agent_id == "researcher"
            return _agent([ToolSpec(type="http_request")])

        monkeypatch.setattr(agents_service, "get_agent", _get_agent)
        with pytest.raises(ValidationError, match="researcher") as ei:
            await service.create_workflow(
                _FakePool(),
                account_id="acc_x",
                name="w",
                script="async def main(input):\n    return await agent({'x': 1}, agent_id='researcher')\n",
                tools=[],
            )
        assert "http_request" in str(ei.value)
        assert ei.value.detail["reason"] == "undeclared_agent_surface"
        assert insert_calls == []

    async def test_covered_agent_surface_accepted(
        self, insert_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _get_agent(pool: Any, agent_id: str, *, account_id: str) -> Agent:
            return _agent([ToolSpec(type="http_request")])

        monkeypatch.setattr(agents_service, "get_agent", _get_agent)
        wf = await service.create_workflow(
            _FakePool(),
            account_id="acc_x",
            name="w",
            script="async def main(input):\n    return await agent({'x': 1}, agent_id='researcher')\n",
            tools=[ToolSpec(type="http_request")],
        )
        assert wf is not None
        assert len(insert_calls) == 1

    async def test_unresolved_agent_does_not_false_reject(
        self, insert_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Agent not yet created → not a create-time rejection (authoring may precede it).
        async def _get_agent(pool: Any, agent_id: str, *, account_id: str) -> Agent:
            raise NotFoundError("no such agent")

        monkeypatch.setattr(agents_service, "get_agent", _get_agent)
        wf = await service.create_workflow(
            _FakePool(),
            account_id="acc_x",
            name="w",
            script="async def main(input):\n    return await agent({'x': 1}, agent_id='later')\n",
            tools=[],
        )
        assert wf is not None
        assert len(insert_calls) == 1


class TestUpdatePathValidation:
    """Criteria 1-7 apply on update as well as create (merged script/tools)."""

    async def test_update_new_script_syntax_error_rejected(
        self, update_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tools omitted → the merged-validation fetches current; stub it.
        async def _get_wf(pool: Any, wid: str, *, account_id: str) -> Workflow:
            return _workflow(tools=[])

        monkeypatch.setattr(service, "get_workflow", _get_wf)
        with pytest.raises(ValidationError):
            await service.update_workflow(
                _FakePool(),
                "wf_1",
                account_id="acc_x",
                expected_version=1,
                script="async def main(input):\n    x = (\n",
            )
        assert update_calls == []

    async def test_update_new_script_under_declared_against_existing_tools(
        self, update_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tools omitted → merged tools = current.tools (empty) → uncovered 'http_request'.
        async def _get_wf(pool: Any, wid: str, *, account_id: str) -> Workflow:
            return _workflow(tools=[])

        monkeypatch.setattr(service, "get_workflow", _get_wf)
        with pytest.raises(ValidationError, match="http_request"):
            await service.update_workflow(
                _FakePool(),
                "wf_1",
                account_id="acc_x",
                expected_version=1,
                script="async def main(input):\n    return await tool('http_request', {})\n",
            )
        assert update_calls == []

    async def test_update_script_omitted_validates_stored_script(
        self, update_calls: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # script omitted → merged script = current.script (which is valid) → accepted,
        # even though only the description changed.
        async def _get_wf(pool: Any, wid: str, *, account_id: str) -> Workflow:
            return _workflow(script=_GOOD, tools=[])

        monkeypatch.setattr(service, "get_workflow", _get_wf)
        wf = await service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            description="new desc",
        )
        assert wf.version == 2
        assert len(update_calls) == 1

    async def test_update_valid_new_script_and_tools_succeeds(
        self, update_calls: list[Any]
    ) -> None:
        wf = await service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script="async def main(input):\n    return await tool('bash', {})\n",
            tools=[ToolSpec(type="bash")],
        )
        assert wf.version == 2
        assert len(update_calls) == 1
