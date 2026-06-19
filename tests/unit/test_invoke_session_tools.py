"""Unit tests for the session-caller ``invoke*`` builtins (#1127).

These stub the worker pool + services so they need no live Postgres. They cover:

* **identity invariant** — a trusted ``caller``/``account_id`` smuggled into the
  arguments is rejected by the tool schema (``additionalProperties: false``) before
  the handler runs.
* **park + resolve** — the handler writes the edge via ``service.invoke`` with
  ``caller={kind:session, id:<this session>}``, parks via ``await_session`` /
  ``await_run``, and shapes ``{ok | error}``.
* **output_schema** — a non-conforming answer is reported fail-loud as an error.
* **porcelain wiring** — ``invoke_agent`` (create+invoke) and ``invoke_workflow``
  (create_run+await) call the right services with the caller's env / lineage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.invocations import InvocationHandle
from aios.models.sessions import SessionAwaitResponse
from aios.models.workflows import WfRunWaitResponse
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult

_CALLER = "ses_caller"
_ACCOUNT = "acc_x"


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )
    monkeypatch.setattr(
        "aios.services.sessions.get_session_basic",
        AsyncMock(return_value=SimpleNamespace(environment_id="env_1", parent_run_id=None)),
    )
    # await_completion-backed parks resolve db_url from settings; make it cheap.
    monkeypatch.setattr(
        "aios.config.get_settings", lambda: SimpleNamespace(db_url="postgresql://x")
    )


def _handle(servicer_id: str = "ses_target", request_id: str = "req_1") -> InvocationHandle:
    return InvocationHandle(servicer_kind="session", servicer_id=servicer_id, request_id=request_id)


async def test_invoke_arg_schema_forbids_caller(monkeypatch: Any) -> None:
    """A smuggled ``caller`` key is rejected by the schema before the handler runs."""
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _CALLER,
            "invoke",
            {"session_id": "ses_target", "caller": {"kind": "session", "id": "evil"}},
        )


async def test_invoke_parks_and_returns_ok(monkeypatch: Any) -> None:
    inv_mock = AsyncMock(return_value=_handle())
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.sessions.await_session",
        AsyncMock(
            return_value=SessionAwaitResponse(
                done=True, last_reacted_seq=5, result={"v": 1}, is_error=False
            )
        ),
    )
    out = await invoke_builtin(_CALLER, "invoke", {"session_id": "ses_target", "input": "hi"})
    assert out == {"ok": {"v": 1}}
    # caller names THIS session, target_kind=session, target is the model-supplied id.
    assert inv_mock.await_args is not None
    kwargs = inv_mock.await_args.kwargs
    assert kwargs["caller"] == {"kind": "session", "id": _CALLER}
    assert kwargs["target_kind"] == "session"
    assert kwargs["target"] == "ses_target"


async def test_invoke_returns_error_outcome(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.sessions.await_session",
        AsyncMock(
            return_value=SessionAwaitResponse(
                done=True, last_reacted_seq=5, result=None, is_error=True, error={"kind": "boom"}
            )
        ),
    )
    out = await invoke_builtin(_CALLER, "invoke", {"session_id": "ses_target"})
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert out.content == {"error": {"kind": "boom"}}


async def test_invoke_output_schema_violation(monkeypatch: Any) -> None:
    """A non-conforming answer is reported fail-loud (output_schema_violation)."""
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.sessions.await_session",
        AsyncMock(
            return_value=SessionAwaitResponse(
                done=True, last_reacted_seq=5, result="not-an-object", is_error=False
            )
        ),
    )
    out = await invoke_builtin(
        _CALLER,
        "invoke",
        {"session_id": "ses_target", "output_schema": {"type": "object"}},
    )
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str) and "output_schema_violation" in out.content


async def test_invoke_output_schema_conforms(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.sessions.await_session",
        AsyncMock(
            return_value=SessionAwaitResponse(
                done=True, last_reacted_seq=5, result={"a": 1}, is_error=False
            )
        ),
    )
    out = await invoke_builtin(
        _CALLER,
        "invoke",
        {"session_id": "ses_target", "output_schema": {"type": "object"}},
    )
    assert out == {"ok": {"a": 1}}


async def test_invoke_repolls_until_done(monkeypatch: Any) -> None:
    """A not-yet-done await is re-polled until it resolves."""
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    await_mock = AsyncMock(
        side_effect=[
            SessionAwaitResponse(done=False, last_reacted_seq=1),
            SessionAwaitResponse(done=True, last_reacted_seq=2, result="ok", is_error=False),
        ]
    )
    monkeypatch.setattr("aios.services.sessions.await_session", await_mock)
    out = await invoke_builtin(_CALLER, "invoke", {"session_id": "ses_target"})
    assert out == {"ok": "ok"}
    assert await_mock.await_count == 2


async def test_invoke_agent_create_then_invoke(monkeypatch: Any) -> None:
    inv_mock = AsyncMock(return_value=_handle(servicer_id="ses_child"))
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.sessions.await_session",
        AsyncMock(
            return_value=SessionAwaitResponse(
                done=True, last_reacted_seq=1, result="r", is_error=False
            )
        ),
    )
    out = await invoke_builtin(_CALLER, "invoke_agent", {"agent_id": "agt_1", "input": "go"})
    assert out == {"ok": "r"}
    assert inv_mock.await_args is not None
    kwargs = inv_mock.await_args.kwargs
    assert kwargs["target_kind"] == "agent"
    assert kwargs["target"] == "agt_1"
    assert kwargs["environment_id"] == "env_1"  # inherits caller's env
    assert kwargs["caller"] == {"kind": "session", "id": _CALLER}


async def test_invoke_workflow_create_run_then_await(monkeypatch: Any) -> None:
    run_mock = AsyncMock(return_value=SimpleNamespace(id="run_1"))
    monkeypatch.setattr("aios.services.workflows.create_run", run_mock)
    monkeypatch.setattr(
        "aios.services.workflows.await_run",
        AsyncMock(
            return_value=WfRunWaitResponse(
                run_status="completed", done=True, output={"k": "v"}, is_error=False
            )
        ),
    )
    out = await invoke_builtin(
        _CALLER, "invoke_workflow", {"workflow_id": "wf_1", "input": {"x": 1}}
    )
    assert out == {"ok": {"k": "v"}}
    assert run_mock.await_args is not None
    kwargs = run_mock.await_args.kwargs
    assert kwargs["workflow_id"] == "wf_1"
    assert kwargs["environment_id"] == "env_1"
    assert kwargs["launcher_session_id"] == _CALLER


async def test_invoke_workflow_error_outcome(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.create_run", AsyncMock(return_value=SimpleNamespace(id="run_1"))
    )
    monkeypatch.setattr(
        "aios.services.workflows.await_run",
        AsyncMock(
            return_value=WfRunWaitResponse(
                run_status="errored", done=True, output=None, is_error=True, error={"kind": "x"}
            )
        ),
    )
    out = await invoke_builtin(_CALLER, "invoke_workflow", {"workflow_id": "wf_1"})
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert out.content == {"error": {"kind": "x"}}


async def test_invoke_workflow_output_schema_violation(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.create_run", AsyncMock(return_value=SimpleNamespace(id="run_1"))
    )
    monkeypatch.setattr(
        "aios.services.workflows.await_run",
        AsyncMock(
            return_value=WfRunWaitResponse(
                run_status="completed", done=True, output="bad", is_error=False
            )
        ),
    )
    out = await invoke_builtin(
        _CALLER, "invoke_workflow", {"workflow_id": "wf_1", "output_schema": {"type": "object"}}
    )
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str) and "output_schema_violation" in out.content
