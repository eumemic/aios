"""Unit tests for create-time workflow script validation (#1286).

Two layers:

* The **pure validator** (:func:`aios.workflows.script_validation.validate_workflow_script`)
  — compile + ``async def main(input)`` assertion + AST-derived tool/agent surface union
  check, with a stub agent-surface resolver. This is the literal-only, validate-declared
  contract (acceptance criteria 1-7).
* The **service path** (:func:`aios.services.workflows.create_workflow` /
  ``update_workflow``) — proves the validator is wired into BOTH create and update with a
  stubbed pool/queries (criterion 8), so an under-declared or mis-shaped script is rejected
  *before* anything is written, and a valid one still creates exactly as before.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.errors import NotFoundError, ValidationError
from aios.models.agents import ToolSpec
from aios.models.workflows import Workflow
from aios.services import workflows as wf_service
from aios.workflows.script_validation import (
    extract_literal_agent_ids,
    validate_workflow_script,
)

_DT = datetime(2026, 1, 1, tzinfo=UTC)

_VALID_MAIN = "async def main(input):\n    return input\n"


def _resolver(mapping: dict[str, frozenset[str]]):
    def resolve(agent_id: str) -> frozenset[str] | None:
        return mapping.get(agent_id)

    return resolve


# ─── the pure validator ──────────────────────────────────────────────────────


class TestCompileAndShape:
    def test_valid_script_passes(self) -> None:
        # Criterion 6 (happy path): a correct top-level async main is accepted.
        validate_workflow_script(_VALID_MAIN, tools=[])

    def test_syntax_error_rejected_with_location(self) -> None:
        # Criterion 1: a non-compiling script is rejected with a compile/syntax message
        # surfacing the line where available.
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script("async def main(input):\n    return (1\n", tools=[])
        assert "compile" in ei.value.message
        assert "line" in ei.value.message

    def test_missing_main_rejected(self) -> None:
        # Criterion 2: compiles but no top-level main.
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script("x = 1\n", tools=[])
        assert "main" in ei.value.message

    def test_main_nested_not_toplevel_rejected(self) -> None:
        # A main defined inside another function is NOT a top-level main.
        src = "def wrapper():\n    async def main(input):\n        return input\n"
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script(src, tools=[])
        assert "main" in ei.value.message

    def test_plain_def_main_rejected(self) -> None:
        # Criterion 3: top-level main that is not async.
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script("def main(input):\n    return input\n", tools=[])
        assert "async" in ei.value.message

    @pytest.mark.parametrize(
        "sig",
        [
            "async def main():\n    return 1\n",
            "async def main(a, b):\n    return 1\n",
            "async def main(*args):\n    return 1\n",
            "async def main(**kw):\n    return 1\n",
            "async def main(*, input):\n    return 1\n",
        ],
    )
    def test_mis_signatured_main_rejected(self, sig: str) -> None:
        # Criterion 3: async main whose arity is not exactly one positional parameter.
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script(sig, tools=[])
        assert "input" in ei.value.message

    def test_posonly_input_accepted(self) -> None:
        # A positional-only ``input`` still accepts the host's ``main(input_value)`` call.
        validate_workflow_script("async def main(input, /):\n    return input\n", tools=[])

    def test_single_positional_param_any_name_accepted(self) -> None:
        # The host calls main positionally, so the parameter NAME is not load-bearing —
        # ``async def main(i)`` (a real existing-test shape) is admissible.
        validate_workflow_script("async def main(i):\n    return i\n", tools=[])

    def test_default_valued_single_param_accepted(self) -> None:
        validate_workflow_script("async def main(input=None):\n    return input\n", tools=[])


class TestToolSurface:
    def test_under_declared_tool_rejected_naming_it(self) -> None:
        # Criterion 4: a literal tool("X") not in the declared surface → rejected, X named.
        src = _VALID_MAIN.rstrip() + '\n\nasync def helper():\n    await tool("bash", {})\n'
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script(src, tools=[])
        assert "bash" in ei.value.message
        assert ei.value.detail.get("missing_tools") == ["bash"]

    def test_covered_tool_accepted(self) -> None:
        src = 'async def main(input):\n    return await tool("bash", {})\n'
        validate_workflow_script(src, tools=[ToolSpec(type="bash")])

    def test_covered_by_custom_name_accepted(self) -> None:
        src = 'async def main(input):\n    return await tool("mytool", {})\n'
        validate_workflow_script(
            src,
            tools=[
                ToolSpec(
                    type="custom",
                    name="mytool",
                    description="d",
                    input_schema={"type": "object"},
                )
            ],
        )

    def test_multiple_missing_tools_all_named(self) -> None:
        src = (
            'async def main(input):\n    await tool("bash", {})\n    await tool("web_search", {})\n'
        )
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script(src, tools=[ToolSpec(type="bash")])
        assert "web_search" in ei.value.message
        assert ei.value.detail.get("missing_tools") == ["web_search"]

    def test_un_ast_able_tool_name_excluded(self) -> None:
        # Criterion 7: a computed/variable tool name is not extracted → not a violation.
        src = 'async def main(input):\n    name = input["tool"]\n    return await tool(name, {})\n'
        validate_workflow_script(src, tools=[])

    def test_attribute_call_not_treated_as_builtin(self) -> None:
        # ``something.tool("x")`` is not the injected builtin and must be ignored.
        src = 'async def main(input):\n    return obj.tool("bash", {})\n'
        validate_workflow_script(src, tools=[])


class TestAgentSurface:
    def test_under_declared_agent_surface_rejected(self) -> None:
        # Criterion 5: agent(agent_id="A") whose surface is not covered → rejected, named.
        src = 'async def main(input):\n    return await agent({}, agent_id="a1")\n'
        with pytest.raises(ValidationError) as ei:
            validate_workflow_script(
                src, tools=[], resolve_agent_tools=_resolver({"a1": frozenset({"bash"})})
            )
        assert "bash" in ei.value.message
        assert ei.value.detail.get("missing_agent_surface") == ["bash"]

    def test_covered_agent_surface_accepted(self) -> None:
        src = 'async def main(input):\n    return await agent({}, agent_id="a1")\n'
        validate_workflow_script(
            src,
            tools=[ToolSpec(type="bash")],
            resolve_agent_tools=_resolver({"a1": frozenset({"bash"})}),
        )

    def test_unresolvable_agent_excluded(self) -> None:
        # A literal agent_id that does not resolve (absent/cross-account) is excluded
        # from the union — never a false rejection.
        src = 'async def main(input):\n    return await agent({}, agent_id="ghost")\n'
        validate_workflow_script(src, tools=[], resolve_agent_tools=_resolver({}))

    def test_un_ast_able_agent_id_excluded(self) -> None:
        # Criterion 7: agent_id from input/a variable is not extracted → accepted.
        src = 'async def main(input):\n    return await agent({}, agent_id=input["a"])\n'
        called: list[str] = []

        def resolve(agent_id: str) -> frozenset[str] | None:
            called.append(agent_id)
            return frozenset({"bash"})

        validate_workflow_script(src, tools=[], resolve_agent_tools=resolve)
        assert called == []  # nothing literal to resolve

    def test_no_resolver_skips_agent_check(self) -> None:
        # Without a resolver the agent dimension is simply not enforced (still accepts).
        src = 'async def main(input):\n    return await agent({}, agent_id="a1")\n'
        validate_workflow_script(src, tools=[])


class TestExtractLiteralAgentIds:
    def test_extracts_literals_only(self) -> None:
        src = (
            "async def main(input):\n"
            '    await agent({}, agent_id="a1")\n'
            '    await agent({}, agent_id=input["x"])\n'
            '    await agent({}, agent_id="a2")\n'
        )
        assert extract_literal_agent_ids(src) == {"a1", "a2"}

    def test_unparsable_yields_empty(self) -> None:
        assert extract_literal_agent_ids("async def main(input):\n    return (1\n") == set()


# ─── service path: validation wired into create + update ─────────────────────


class _FakeAcquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


def _workflow(**over: Any) -> Workflow:
    base: dict[str, Any] = dict(
        id="wf_1",
        account_id="acc_x",
        name="w",
        version=1,
        script=_VALID_MAIN,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Workflow(**base)


@pytest.fixture
def fake_pool() -> _FakePool:
    return _FakePool(conn=object())


class TestCreateServicePath:
    """Operator path (no creator_session_id) — skips attenuation, so the validator is
    the only gate before the insert. Proves criteria 1-7 are enforced on create."""

    async def test_valid_script_creates(self, monkeypatch: Any, fake_pool: _FakePool) -> None:
        insert = AsyncMock(return_value=_workflow())
        monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
        wf = await wf_service.create_workflow(
            fake_pool, account_id="acc_x", name="w", script=_VALID_MAIN
        )
        assert wf.id == "wf_1"
        assert insert.await_count == 1

    @pytest.mark.parametrize(
        "script",
        [
            "async def main(input):\n    return (1\n",  # syntax
            "x = 1\n",  # missing main
            "def main(input):\n    return input\n",  # not async
            "async def main():\n    return 1\n",  # bad sig
        ],
    )
    async def test_bad_script_rejected_and_not_inserted(
        self, monkeypatch: Any, fake_pool: _FakePool, script: str
    ) -> None:
        insert = AsyncMock(return_value=_workflow())
        monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
        with pytest.raises(ValidationError):
            await wf_service.create_workflow(fake_pool, account_id="acc_x", name="w", script=script)
        assert insert.await_count == 0  # not created

    async def test_under_declared_tool_rejected_not_inserted(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        insert = AsyncMock(return_value=_workflow())
        monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
        script = 'async def main(input):\n    return await tool("bash", {})\n'
        with pytest.raises(ValidationError) as ei:
            await wf_service.create_workflow(
                fake_pool, account_id="acc_x", name="w", script=script, tools=[]
            )
        assert "bash" in ei.value.message
        assert insert.await_count == 0

    async def test_literal_agent_surface_resolved_and_enforced(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        insert = AsyncMock(return_value=_workflow())
        monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
        # The named agent declares ``bash``; the workflow under-declares (no tools).
        # The service reads only ``agent.tools``, so a lightweight stub suffices.
        agent = SimpleNamespace(tools=[ToolSpec(type="bash")])
        monkeypatch.setattr("aios.services.agents.get_agent", AsyncMock(return_value=agent))
        script = 'async def main(input):\n    return await agent({}, agent_id="a1")\n'
        with pytest.raises(ValidationError) as ei:
            await wf_service.create_workflow(
                fake_pool, account_id="acc_x", name="w", script=script, tools=[]
            )
        assert "bash" in ei.value.message
        assert insert.await_count == 0

    async def test_unresolvable_agent_does_not_block_create(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        insert = AsyncMock(return_value=_workflow())
        monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
        monkeypatch.setattr(
            "aios.services.agents.get_agent",
            AsyncMock(side_effect=NotFoundError("no such agent")),
        )
        script = 'async def main(input):\n    return await agent({}, agent_id="ghost")\n'
        wf = await wf_service.create_workflow(
            fake_pool, account_id="acc_x", name="w", script=script, tools=[]
        )
        assert wf.id == "wf_1"
        assert insert.await_count == 1


class TestUpdateServicePath:
    """Operator path update — criterion 8: the validator runs on update too, and over the
    EFFECTIVE (merged) script + tools."""

    async def test_valid_update_succeeds(self, monkeypatch: Any, fake_pool: _FakePool) -> None:
        update = AsyncMock(return_value=_workflow(version=2))
        monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
        monkeypatch.setattr(
            "aios.services.workflows.get_workflow", AsyncMock(return_value=_workflow())
        )
        wf = await wf_service.update_workflow(
            fake_pool,
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=_VALID_MAIN,
        )
        assert wf.version == 2
        assert update.await_count == 1

    async def test_bad_script_update_rejected_not_written(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        update = AsyncMock(return_value=_workflow(version=2))
        monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
        monkeypatch.setattr(
            "aios.services.workflows.get_workflow", AsyncMock(return_value=_workflow())
        )
        with pytest.raises(ValidationError):
            await wf_service.update_workflow(
                fake_pool,
                "wf_1",
                account_id="acc_x",
                expected_version=1,
                script="def main(input):\n    return input\n",
            )
        assert update.await_count == 0

    async def test_update_tools_only_validates_against_current_script(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        # Updating ONLY tools must re-validate the EFFECTIVE pair: the stored script
        # calls tool("bash"); narrowing tools to [] (dropping bash) must be rejected.
        stored = _workflow(
            script='async def main(input):\n    return await tool("bash", {})\n',
            tools=[ToolSpec(type="bash")],
        )
        monkeypatch.setattr("aios.services.workflows.get_workflow", AsyncMock(return_value=stored))
        update = AsyncMock(return_value=_workflow(version=2))
        monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
        with pytest.raises(ValidationError) as ei:
            await wf_service.update_workflow(
                fake_pool,
                "wf_1",
                account_id="acc_x",
                expected_version=1,
                tools=[],  # drops bash → effective script no longer covered
            )
        assert "bash" in ei.value.message
        assert update.await_count == 0

    async def test_update_script_only_validates_against_current_tools(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        # Updating ONLY the script must validate it against the STORED tools.
        stored = _workflow(tools=[ToolSpec(type="bash")])
        monkeypatch.setattr("aios.services.workflows.get_workflow", AsyncMock(return_value=stored))
        update = AsyncMock(return_value=_workflow(version=2))
        monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
        new_script = 'async def main(input):\n    return await tool("bash", {})\n'
        wf = await wf_service.update_workflow(
            fake_pool,
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=new_script,
        )
        assert wf.version == 2
        assert update.await_count == 1

    async def test_update_touching_neither_script_nor_tools_skips_validation(
        self, monkeypatch: Any, fake_pool: _FakePool
    ) -> None:
        # A description-only update never reads/validates the script (no fetch needed on
        # the operator path) and still writes.
        get_wf = AsyncMock()
        monkeypatch.setattr("aios.services.workflows.get_workflow", get_wf)
        update = AsyncMock(return_value=_workflow(version=2))
        monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
        wf = await wf_service.update_workflow(
            fake_pool,
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            description="new desc",
        )
        assert wf.version == 2
        assert get_wf.await_count == 0
        assert update.await_count == 1
