"""Service-path enforcement of create-time script validation (#1285, criterion 8).

These exercise ``aios.services.workflows.create_workflow`` / ``update_workflow`` on the
operator path (no acting session → no surface attenuation, no agent DB read), with the
query layer mocked, to prove:

* validation runs and rejects an invalid script **before any DB write** (the workflow
  is NOT created / updated), and
* a valid, fully-covered script flows through to the insert/update query unchanged
  (the happy path is unaffected).

The tool-handler path (``create_workflow_handler`` / ``update_workflow_handler``) calls
straight through to these service functions, so enforcing here enforces both paths.
``update_workflow`` is covered too (criteria 1-7 apply on update).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.models.agents import ToolSpec
from aios.models.workflows import Workflow
from aios.services import workflows as wf_service
from aios.workflows.script_validation import WorkflowScriptValidationError

_DT = datetime(2026, 1, 1, tzinfo=UTC)

_BAD_SYNTAX = "async def main(input):\n    x = (\n"
_NO_MAIN = "x = 1\n"
_NOT_ASYNC = "def main(input):\n    return input\n"
_BAD_SIG = "async def main():\n    return 1\n"
_UNDER_TOOL = 'async def main(input):\n    return await tool("http_request", {})\n'
_VALID = 'async def main(input):\n    return await tool("bash", {})\n'


class _FakePool:
    """A pool whose ``acquire()`` yields a sentinel conn; tracks whether it was used."""

    def __init__(self) -> None:
        self.acquired = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        self.acquired = True
        yield object()


def _workflow(**over: Any) -> Workflow:
    base: dict[str, Any] = dict(
        id="wf_1",
        account_id="acc_x",
        name="w",
        version=1,
        script=_VALID,
        tools=[ToolSpec(type="bash")],
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Workflow(**base)


# ─── create path ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("script", "needle"),
    [
        (_BAD_SYNTAX, "compile"),
        (_NO_MAIN, "main"),
        (_NOT_ASYNC, "async"),
        (_BAD_SIG, "input"),
    ],
)
async def test_create_rejects_invalid_script_before_db(
    monkeypatch: Any, script: str, needle: str
) -> None:
    insert = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
    pool = _FakePool()
    with pytest.raises(WorkflowScriptValidationError, match=needle):
        await wf_service.create_workflow(pool, account_id="acc_x", name="w", script=script)
    insert.assert_not_called()
    assert pool.acquired is False  # not created — no DB connection taken


async def test_create_rejects_under_declared_tool_naming_it(monkeypatch: Any) -> None:
    insert = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
    pool = _FakePool()
    with pytest.raises(WorkflowScriptValidationError, match="http_request"):
        await wf_service.create_workflow(
            pool, account_id="acc_x", name="w", script=_UNDER_TOOL, tools=[ToolSpec(type="bash")]
        )
    insert.assert_not_called()


async def test_create_accepts_valid_covered_script(monkeypatch: Any) -> None:
    created = _workflow()
    insert = AsyncMock(return_value=created)
    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
    out = await wf_service.create_workflow(
        _FakePool(), account_id="acc_x", name="w", script=_VALID, tools=[ToolSpec(type="bash")]
    )
    assert out is created
    insert.assert_awaited_once()


async def test_create_accepts_unastable_tool_name(monkeypatch: Any) -> None:
    insert = AsyncMock(return_value=_workflow())
    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", insert)
    script = 'async def main(input):\n    n = input["t"]\n    return await tool(n, {})\n'
    await wf_service.create_workflow(_FakePool(), account_id="acc_x", name="w", script=script)
    insert.assert_awaited_once()


# ─── update path ─────────────────────────────────────────────────────────────


async def test_update_rejects_invalid_script_before_db(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.get_workflow", AsyncMock(return_value=_workflow())
    )
    update = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    pool = _FakePool()
    with pytest.raises(WorkflowScriptValidationError, match="compile"):
        await wf_service.update_workflow(
            pool,
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=_BAD_SYNTAX,
            tools=[],
        )
    update.assert_not_called()
    assert pool.acquired is False  # not updated — no write connection taken


async def test_update_rejects_under_declared_tool(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.get_workflow", AsyncMock(return_value=_workflow())
    )
    update = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    with pytest.raises(WorkflowScriptValidationError, match="http_request"):
        await wf_service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=_UNDER_TOOL,
            tools=[ToolSpec(type="bash")],
        )
    update.assert_not_called()


async def test_update_validates_new_script_against_preserved_tools(monkeypatch: Any) -> None:
    # script changes, tools omitted (preserved) → validation must read current.tools.
    # The current workflow declares only ``bash``; the new script calls ``http_request``.
    current = _workflow(tools=[ToolSpec(type="bash")])
    monkeypatch.setattr(
        "aios.services.workflows.get_workflow", AsyncMock(return_value=current)
    )
    update = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    with pytest.raises(WorkflowScriptValidationError, match="http_request"):
        await wf_service.update_workflow(
            _FakePool(), "wf_1", account_id="acc_x", expected_version=1, script=_UNDER_TOOL
        )
    update.assert_not_called()


async def test_update_validates_preserved_script_against_new_tools(monkeypatch: Any) -> None:
    # script omitted (preserved current ``tool("bash")`` body), tools narrowed to [] →
    # the preserved script's literal ``tool("bash")`` is now under-declared → rejected.
    current = _workflow(script=_VALID, tools=[ToolSpec(type="bash")])
    monkeypatch.setattr(
        "aios.services.workflows.get_workflow", AsyncMock(return_value=current)
    )
    update = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    with pytest.raises(WorkflowScriptValidationError, match="bash"):
        await wf_service.update_workflow(
            _FakePool(), "wf_1", account_id="acc_x", expected_version=1, tools=[]
        )
    update.assert_not_called()


async def test_update_stale_token_409_precedes_script_validation(monkeypatch: Any) -> None:
    # A stale optimistic token keeps its 409 even when the new script is invalid —
    # the precondition is checked before validation (no 422 masking the 409).
    from aios.errors import ConflictError

    monkeypatch.setattr(
        "aios.services.workflows.get_workflow",
        AsyncMock(return_value=_workflow(version=5)),
    )
    update = AsyncMock()
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    with pytest.raises(ConflictError, match="version mismatch"):
        await wf_service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=_BAD_SYNTAX,
        )
    update.assert_not_called()


async def test_update_accepts_valid_covered_script(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.get_workflow", AsyncMock(return_value=_workflow())
    )
    updated = _workflow(version=2)
    update = AsyncMock(return_value=updated)
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", update)
    out = await wf_service.update_workflow(
        _FakePool(),
        "wf_1",
        account_id="acc_x",
        expected_version=1,
        script=_VALID,
        tools=[ToolSpec(type="bash")],
    )
    assert out is updated
    update.assert_awaited_once()
