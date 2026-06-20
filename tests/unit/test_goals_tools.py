"""Unit tests for the self-goal builtins ``set_goal`` / ``cancel_goal`` (#1414).

These stub the worker pool + services so they need no live Postgres. They cover:

* **identity invariant** — a smuggled ``caller``/``session_id``/``target`` in the
  arguments is rejected by the schema (``extra=forbid``) before the handler runs.
* **no-park write** — ``set_goal`` writes the self-edge via ``service.set_goal``
  (which uses the no-park ``Ask(ExistingSession)`` arm) and returns ``{goal_id}``;
  it never parks.
* **deterministic id** — the ``request_id`` is a pure function of
  ``(session_id, tool_call_id)``; two different tool_call_ids → two distinct goals,
  the same tool_call_id → the same id (idempotency key).
* **cap** — past the open-goal cap ``set_goal`` returns a rate-limited error.
* **SECURITY** — ``cancel_goal`` rejects a non-self obligation BEFORE any write,
  and stamps ``{kind:cancelled, by:self}`` on a genuine self-goal.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.services import sessions as sessions_service
from aios.tools.goals import cancel_goal_response
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult

_CALLER = "ses_caller"
_ACCOUNT = "acc_x"


class _FakePool:
    """Minimal ``pool.acquire()`` async-context-manager stub for cancel_goal_response
    (which only opens a conn to read ``get_request_caller``, itself monkeypatched)."""

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield object()


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )


# ─── set_goal ────────────────────────────────────────────────────────────────


async def test_set_goal_schema_forbids_caller() -> None:
    """A smuggled trusted field is rejected by the schema before the handler runs."""
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _CALLER,
            "set_goal",
            {"goal": "do the thing", "caller": {"kind": "session", "id": "evil"}},
            "tc_1",
        )


async def test_set_goal_schema_forbids_session_id_and_target() -> None:
    for smuggled in ({"session_id": "ses_x"}, {"target": "ses_x"}):
        with pytest.raises(ToolBail):
            await invoke_builtin(_CALLER, "set_goal", {"goal": "g", **smuggled}, "tc_1")


async def test_set_goal_writes_self_edge_no_park(monkeypatch: Any) -> None:
    """``set_goal`` calls the service ``set_goal`` writer and returns ``{goal_id}``.

    The service fn writes the edge via the no-park ``Ask(ExistingSession)`` arm; the
    handler never invokes a park primitive — the open edge re-wakes via the guard
    nudge, not a parked tool-call (which would self-deadlock the quiescence guard).
    """
    set_goal_mock = AsyncMock(return_value="req_goal_1")
    monkeypatch.setattr("aios.services.sessions.set_goal", set_goal_mock)
    # If the handler tried to park, this would be invoked — assert it never is.
    park_mock = AsyncMock()
    monkeypatch.setattr("aios.tools.invoke_session._park_on_session", park_mock, raising=False)

    out = await invoke_builtin(_CALLER, "set_goal", {"goal": "ship it"}, "tc_42")
    assert out == {"goal_id": "req_goal_1"}
    park_mock.assert_not_called()

    kwargs = set_goal_mock.await_args.kwargs
    assert kwargs["goal"] == "ship it"
    # The request_id is the deterministic id for THIS (session, tool_call_id).
    assert kwargs["request_id"] == sessions_service.goal_request_id(_CALLER, "tc_42")


async def test_set_goal_deterministic_id_distinguishes_calls() -> None:
    """Two different tool_call_ids → distinct goals; the same one → the same id."""
    a = sessions_service.goal_request_id(_CALLER, "tc_a")
    b = sessions_service.goal_request_id(_CALLER, "tc_b")
    assert a != b  # two deliberate set_goal(same text) calls → two distinct goals
    assert a == sessions_service.goal_request_id(_CALLER, "tc_a")  # stable re-dispatch key
    assert a.startswith("req_")


async def test_set_goal_passes_output_schema(monkeypatch: Any) -> None:
    set_goal_mock = AsyncMock(return_value="req_goal_2")
    monkeypatch.setattr("aios.services.sessions.set_goal", set_goal_mock)
    await invoke_builtin(
        _CALLER,
        "set_goal",
        {"goal": "g", "output_schema": {"type": "object"}},
        "tc_9",
    )
    assert set_goal_mock.await_args.kwargs["output_schema"] == {"type": "object"}


async def test_set_goal_cap_returns_error(monkeypatch: Any) -> None:
    """Past the open-goal cap the handler surfaces the rate-limited error."""
    from aios.errors import RateLimitedError

    monkeypatch.setattr(
        "aios.services.sessions.set_goal",
        AsyncMock(side_effect=RateLimitedError("open-goal cap reached (10/10)")),
    )
    out = await invoke_builtin(_CALLER, "set_goal", {"goal": "g"}, "tc_cap")
    assert isinstance(out, ToolResult) and out.is_error
    assert "cap" in str(out.content)


# ─── cancel_goal ─────────────────────────────────────────────────────────────


async def test_cancel_goal_schema_forbids_extra() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_CALLER, "cancel_goal", {"goal_id": "req_1", "by": "operator"})


async def test_cancel_goal_self_goal_stamps_cancelled(monkeypatch: Any) -> None:
    """A genuine self-goal is retracted with ``{kind:cancelled, by:self}``."""
    pool = _FakePool()
    # caller edge IS a self-goal of this session.
    get_caller = AsyncMock(return_value={"kind": "session", "id": _CALLER})
    monkeypatch.setattr("aios.db.queries.get_request_caller", get_caller)
    respond = AsyncMock(return_value="responded")
    monkeypatch.setattr("aios.tools.workflow_completion.respond_to_request", respond)

    status = await cancel_goal_response(
        pool, _CALLER, account_id=_ACCOUNT, goal_id="req_g", by="self"
    )
    assert status == "cancelled"
    kwargs = respond.await_args.kwargs
    assert kwargs["is_error"] is True
    assert kwargs["error"] == {"kind": "cancelled", "by": "self"}
    assert kwargs["request_id"] == "req_g"


async def test_cancel_goal_rejects_peer_invoke_before_write(monkeypatch: Any) -> None:
    """SECURITY: a peer-invoke obligation (caller is a DIFFERENT session) is rejected
    BEFORE any response is written — never stamped ``{by:self}``."""
    pool = _FakePool()
    get_caller = AsyncMock(return_value={"kind": "session", "id": "ses_OTHER"})
    monkeypatch.setattr("aios.db.queries.get_request_caller", get_caller)
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.workflow_completion.respond_to_request", respond)

    status = await cancel_goal_response(
        pool, _CALLER, account_id=_ACCOUNT, goal_id="req_peer", by="self"
    )
    assert status == "not_self_goal"
    respond.assert_not_called()


async def test_cancel_goal_rejects_workflow_child_before_write(monkeypatch: Any) -> None:
    """SECURITY: a workflow-child obligation (caller.kind == run) is rejected."""
    pool = _FakePool()
    monkeypatch.setattr(
        "aios.db.queries.get_request_caller",
        AsyncMock(return_value={"kind": "run", "id": "run_1"}),
    )
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.workflow_completion.respond_to_request", respond)

    status = await cancel_goal_response(
        pool, _CALLER, account_id=_ACCOUNT, goal_id="req_child", by="self"
    )
    assert status == "not_self_goal"
    respond.assert_not_called()


async def test_cancel_goal_unknown_request_rejected(monkeypatch: Any) -> None:
    """An id with no edge (None caller) is rejected before any write."""
    pool = _FakePool()
    monkeypatch.setattr("aios.db.queries.get_request_caller", AsyncMock(return_value=None))
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.workflow_completion.respond_to_request", respond)

    status = await cancel_goal_response(
        pool, _CALLER, account_id=_ACCOUNT, goal_id="req_nope", by="self"
    )
    assert status == "not_self_goal"
    respond.assert_not_called()


async def test_cancel_goal_operator_by_label(monkeypatch: Any) -> None:
    """The operator path stamps ``by:operator`` on a verified self-goal."""
    pool = _FakePool()
    monkeypatch.setattr(
        "aios.db.queries.get_request_caller",
        AsyncMock(return_value={"kind": "session", "id": _CALLER}),
    )
    respond = AsyncMock(return_value="responded")
    monkeypatch.setattr("aios.tools.workflow_completion.respond_to_request", respond)

    status = await cancel_goal_response(
        pool, _CALLER, account_id=_ACCOUNT, goal_id="req_g", by="operator"
    )
    assert status == "cancelled"
    assert respond.await_args.kwargs["error"] == {"kind": "cancelled", "by": "operator"}
