"""Service-path wiring for create-time validation (#1284, criterion 8).

Proves that BOTH the create path and the update path of
``aios.services.workflows`` enforce :func:`validate_workflow_script` — the single
chokepoint the tool-handler builtin (``create_workflow_handler`` /
``update_workflow_handler``), the HTTP router, and any other caller all flow
through. No DB: the query layer is stubbed, and on a rejecting script the service
must raise *before* ever touching it.

(The pure validator's full criteria 1-7 coverage lives in
``test_workflow_script_validation.py``; here we only assert the service calls it
and that a valid script still reaches the insert/update query unchanged — the
happy path is unaffected.)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from aios.models.agents import ToolSpec
from aios.models.workflows import Workflow
from aios.services import workflows as service
from aios.workflows.script_validation import WorkflowScriptValidationError

_DT = datetime(2026, 1, 1, tzinfo=UTC)

_BAD_SCRIPT = "async def main(input):\n    return (\n"  # syntax error
_UNDECLARED = 'async def main(input):\n    return await tool("bash", {})\n'
_GOOD_SCRIPT = "async def main(input):\n    return input\n"


class _FakePool:
    """A pool whose ``acquire()`` yields a sentinel conn — the stubbed query
    functions ignore it, so no DB is touched."""

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield object()


def _workflow(**over: Any) -> Workflow:
    base: dict[str, Any] = dict(
        id="wf_1",
        account_id="acc_x",
        name="w",
        version=1,
        script=_GOOD_SCRIPT,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Workflow(**base)


# ── create ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_rejects_bad_script_before_db(monkeypatch: Any) -> None:
    insert_called = False

    async def _insert(*a: Any, **k: Any) -> Workflow:
        nonlocal insert_called
        insert_called = True
        return _workflow()

    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", _insert)
    with pytest.raises(WorkflowScriptValidationError):
        await service.create_workflow(_FakePool(), account_id="acc_x", name="w", script=_BAD_SCRIPT)
    assert insert_called is False  # rejected before the insert


@pytest.mark.asyncio
async def test_create_rejects_undeclared_tool(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.db.queries.workflows.insert_workflow",
        lambda *a, **k: pytest.fail("insert should not run"),
    )
    with pytest.raises(WorkflowScriptValidationError) as exc:
        await service.create_workflow(
            _FakePool(), account_id="acc_x", name="w", script=_UNDECLARED, tools=[]
        )
    assert "bash" in exc.value.message


@pytest.mark.asyncio
async def test_create_valid_reaches_insert(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def _insert(conn: Any, **k: Any) -> Workflow:
        seen.update(k)
        return _workflow(script=k["script"])

    monkeypatch.setattr("aios.db.queries.workflows.insert_workflow", _insert)
    wf = await service.create_workflow(
        _FakePool(),
        account_id="acc_x",
        name="w",
        script=_UNDECLARED,
        tools=[ToolSpec(type="bash")],
    )
    assert seen["script"] == _UNDECLARED  # happy path unchanged: reached the insert
    assert wf.script == _UNDECLARED


# ── update ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_rejects_bad_script(monkeypatch: Any) -> None:
    # Operator path (no actor): the bad script is rejected before the update query.
    async def _get(conn: Any, workflow_id: str, *, account_id: str) -> Workflow:
        return _workflow(version=1)

    monkeypatch.setattr("aios.db.queries.workflows.get_workflow", _get)
    monkeypatch.setattr(
        "aios.db.queries.workflows.update_workflow",
        lambda *a, **k: pytest.fail("update should not run"),
    )
    with pytest.raises(WorkflowScriptValidationError):
        await service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            script=_BAD_SCRIPT,
            tools=[],
            mcp_servers=[],
        )


@pytest.mark.asyncio
async def test_update_validates_effective_surface_with_preserved_script(
    monkeypatch: Any,
) -> None:
    # script omitted (preserved) but tools narrowed to [] — the preserved script
    # still calls tool("bash"), so the effective shape is now under-declared.
    async def _get(conn: Any, workflow_id: str, *, account_id: str) -> Workflow:
        return _workflow(script=_UNDECLARED, tools=[ToolSpec(type="bash")])

    monkeypatch.setattr("aios.db.queries.workflows.get_workflow", _get)
    monkeypatch.setattr(
        "aios.db.queries.workflows.update_workflow",
        lambda *a, **k: pytest.fail("update should not run"),
    )
    with pytest.raises(WorkflowScriptValidationError) as exc:
        await service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,
            tools=[],  # narrow away bash while the script still calls it
        )
    assert "bash" in exc.value.message


@pytest.mark.asyncio
async def test_update_stale_version_409_precedes_script_validation(monkeypatch: Any) -> None:
    # A stale optimistic token (409) keeps precedence over a 422 script-validation error
    # even when the submitted script is invalid — the token-probe contract existing
    # callers rely on (an invalid throwaway script must not mask the conflict).
    from aios.errors import ConflictError

    async def _get(conn: Any, workflow_id: str, *, account_id: str) -> Workflow:
        return _workflow(version=5)

    monkeypatch.setattr("aios.db.queries.workflows.get_workflow", _get)
    monkeypatch.setattr(
        "aios.db.queries.workflows.update_workflow",
        lambda *a, **k: pytest.fail("update should not run"),
    )
    with pytest.raises(ConflictError):
        await service.update_workflow(
            _FakePool(),
            "wf_1",
            account_id="acc_x",
            expected_version=1,  # stale
            script=_BAD_SCRIPT,  # invalid, but the 409 must win
        )


@pytest.mark.asyncio
async def test_update_valid_reaches_update_query(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def _get(conn: Any, workflow_id: str, *, account_id: str) -> Workflow:
        return _workflow(script=_GOOD_SCRIPT, tools=[])

    async def _update(conn: Any, workflow_id: str, **k: Any) -> Workflow:
        seen.update(k)
        return _workflow(version=2, script=k.get("script") or _GOOD_SCRIPT)

    monkeypatch.setattr("aios.db.queries.workflows.get_workflow", _get)
    monkeypatch.setattr("aios.db.queries.workflows.update_workflow", _update)
    wf = await service.update_workflow(
        _FakePool(),
        "wf_1",
        account_id="acc_x",
        expected_version=1,
        script=_UNDECLARED,
        tools=[ToolSpec(type="bash")],
    )
    assert wf.version == 2  # happy path unchanged: reached the update query
