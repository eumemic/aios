"""Unit tests for the model-facing task verbs — ``stop_task`` + ``list_tasks`` (#1428).

DB-free: the registry shape (model-only transport, no smuggled args) and the ``stop_task``
handler's branch logic + security wiring are pinned by patching the query/service seams. The
end-to-end DB effects (cancel-marker / cancel-signal seeding, the open-roster read) live in
``tests/integration/test_model_task_tools.py``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from aios.tools import tasks as task_tools
from aios.tools.registry import ToolResult

# ─── registration shape ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["stop_task", "list_tasks"])
def test_registered_as_model_only_tool(name: str) -> None:
    """Both verbs register model-only (``agent_tool``) with a closed (``extra="forbid"``)
    schema, so the CLI broker refuses them and a smuggled trusted id is rejected up front."""
    import aios.tools  # noqa: F401 - trigger built-in registration side effects
    from aios.tools.registry import registry

    definition = registry.get(name)
    assert definition.transport == "agent_tool"
    assert definition.parameters_schema.get("additionalProperties") is False


# ─── stop_task handler branches (patched seams) ──────────────────────────────


class _FakeAcquireCM:
    async def __aenter__(self) -> Any:
        return mock.MagicMock()

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _fake_pool() -> Any:
    pool = mock.MagicMock()
    pool.acquire = mock.Mock(return_value=_FakeAcquireCM())
    return pool


def _async_ret(value: Any) -> Any:
    async def _f(*_a: object, **_k: object) -> Any:
        return value

    return _f


def _wire_common(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the pool + account-load seams; return the fake pool for call-arg assertions."""
    fake_pool = _fake_pool()
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: fake_pool)
    monkeypatch.setattr("aios.services.sessions.load_session_account_id", _async_ret("acc_1"))
    return fake_pool


async def test_stop_task_threads_canceller_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An open run task → cancel_invocation is seeded with canceller_session_id = the executing
    session (the launcher guard, construction-held) and the full edge handle."""
    fake_pool = _wire_common(monkeypatch)
    monkeypatch.setattr(
        "aios.db.queries.find_parked_servicer", _async_ret(("run", "wfr_1", None, None))
    )
    monkeypatch.setattr("aios.db.queries.workflows.derive_run_response", _async_ret(None))
    cancel_spy = mock.AsyncMock()
    monkeypatch.setattr("aios.services.invocations.cancel_invocation", cancel_spy)

    out = await task_tools.stop_task_handler("ses_caller", {"tool_call_id": "tc_1"})

    assert out == {"ok": "stop requested"}
    cancel_spy.assert_awaited_once_with(
        fake_pool,
        servicer_kind="run",
        servicer_id="wfr_1",
        request_id=None,
        account_id="acc_1",
        canceller_session_id="ses_caller",
    )


async def test_stop_task_none_handle_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A foreign/absent tool_call_id (find_parked_servicer → None) → a model-visible error and
    NO cancel seeded."""
    _wire_common(monkeypatch)
    monkeypatch.setattr("aios.db.queries.find_parked_servicer", _async_ret(None))
    cancel_spy = mock.AsyncMock()
    monkeypatch.setattr("aios.services.invocations.cancel_invocation", cancel_spy)

    out = await task_tools.stop_task_handler("ses_caller", {"tool_call_id": "tc_missing"})

    assert isinstance(out, ToolResult)
    assert out.is_error is True
    assert isinstance(out.content, str) and "no open task" in out.content
    cancel_spy.assert_not_awaited()


async def test_stop_task_already_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session task that already answered (derive_response non-None) → 'already resolved',
    NO cancel seeded."""
    _wire_common(monkeypatch)
    monkeypatch.setattr(
        "aios.db.queries.find_parked_servicer",
        _async_ret(("session", "ses_srv", "req_1", None)),
    )
    monkeypatch.setattr(
        "aios.db.queries.derive_response",
        _async_ret({"result": {"v": 1}, "is_error": False, "error": None}),
    )
    cancel_spy = mock.AsyncMock()
    monkeypatch.setattr("aios.services.invocations.cancel_invocation", cancel_spy)

    out = await task_tools.stop_task_handler("ses_caller", {"tool_call_id": "tc_1"})

    assert out == {"ok": "already resolved"}
    cancel_spy.assert_not_awaited()
