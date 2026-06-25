"""Unit tests for the explicit goal-management builtins (#1508, #1512).

These stub the worker pool + services so they need no live Postgres. They cover:

* **identity invariant** — a trusted ``caller``/``account_id`` smuggled into the
  arguments is rejected by the tool schema (``additionalProperties: false``) before
  the handler runs.
* **create_goal opens a self-goal** — the handler writes the self-referential
  awaited edge via ``service.invoke`` with ``target_kind="session"``,
  ``target=<this session>``, ``caller={kind:session, id:<this session>}`` (the
  #1414 self-goal path), carrying the REQUIRED ``output_schema`` (the completion
  contract), and returns the opened edge's ``request_id`` as ``goal_id``.
* **output_schema is mandatory** — a ``create_goal`` without an ``output_schema``
  is rejected by the tool schema before the handler runs (#1512: no schemaless goal).
* **admission cap** — at/over the per-session open-goal cap ``create_goal`` returns
  a clear error and opens no edge.
* **list_goals** filters the open-obligation set to self-caller goals only.
* **complete_goal / fail_goal** close an open self-goal via ``respond_to_request``
  (the return / error arms), and reject a goal_id that isn't an open self-goal.
* **complete_goal validates result** — the ``result`` is checked against the goal's
  persisted ``output_schema`` (reusing ``_validate_output`` / the
  ``output_schema_violation`` path from ``invoke_session.py``); a non-conforming
  result is rejected and no response is written.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.sessions import Obligation
from aios.models.tasks import TaskHandle
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult

_SELF = "ses_self"
_ACCOUNT = "acc_x"

# A representative output_schema — the completion contract a goal pins up front.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"shipped": {"type": "boolean"}},
    "required": ["shipped"],
}


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )
    monkeypatch.setattr(
        "aios.tools.goal_management.get_settings",
        lambda: SimpleNamespace(session_open_goals_max=10),
    )


def _self_goal(rid: str, *, summary: str = "ship the thing", age_s: int = 0) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind="session",
        caller_id=_SELF,
        opened_at=datetime.now(UTC),
        summary=summary,
    )


def _foreign(rid: str, *, kind: str = "api") -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind=kind,
        caller_id="someone_else",
        opened_at=datetime.now(UTC),
        summary="a caller-assigned task",
    )


def _stub_open_obligations(monkeypatch: Any, obligations: list[Obligation]) -> None:
    """Stub ``queries.get_open_obligations`` (called under an acquired conn)."""
    monkeypatch.setattr("aios.db.queries.get_open_obligations", AsyncMock(return_value=obligations))

    class _Conn:
        async def __aenter__(self) -> Any:
            return object()

        async def __aexit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(
        "aios.harness.runtime.require_pool", lambda: SimpleNamespace(acquire=lambda: _Conn())
    )


def _stub_goal_schema(monkeypatch: Any, schema: dict[str, Any] | None) -> None:
    """Stub the goal's persisted ``output_schema`` read (``get_request_output_schema``)
    that ``complete_goal`` resolves to validate the result against."""
    monkeypatch.setattr("aios.db.queries.get_request_output_schema", AsyncMock(return_value=schema))


# ─── create_goal ─────────────────────────────────────────────────────────────


async def test_create_goal_arg_schema_forbids_caller() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _SELF,
            "create_goal",
            {"goal": "x", "output_schema": _SCHEMA, "caller": {"kind": "session", "id": "evil"}},
        )


async def test_create_goal_requires_nonempty_goal() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_SELF, "create_goal", {"goal": "", "output_schema": _SCHEMA})


async def test_create_goal_requires_output_schema() -> None:
    """#1512: output_schema is mandatory — a schemaless goal is rejected up front."""
    with pytest.raises(ToolBail):
        await invoke_builtin(_SELF, "create_goal", {"goal": "ship it"})


async def test_create_goal_opens_self_goal_edge(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [])  # no goals open yet → under cap
    inv = AsyncMock(
        return_value=TaskHandle(servicer_kind="session", servicer_id=_SELF, request_id="req_goal")
    )
    monkeypatch.setattr("aios.services.sessions.invoke", inv)

    out = await invoke_builtin(_SELF, "create_goal", {"goal": "ship it", "output_schema": _SCHEMA})

    assert out == {"goal_id": "req_goal", "goal": "ship it", "status": "open"}
    assert inv.await_args is not None
    kwargs = inv.await_args.kwargs
    # The self-goal path: target IS this session, caller names this session.
    assert kwargs["target_kind"] == "session"
    assert kwargs["target"] == _SELF
    assert kwargs["caller"] == {"kind": "session", "id": _SELF}
    assert kwargs["input"] == {"goal": "ship it"}
    # The completion contract is persisted on the edge, the same way call_* carry it.
    assert kwargs["output_schema"] == _SCHEMA


async def test_create_goal_cap_enforced(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.tools.goal_management.get_settings", lambda: SimpleNamespace(session_open_goals_max=2)
    )
    _stub_open_obligations(monkeypatch, [_self_goal("g1"), _self_goal("g2")])
    inv = AsyncMock()
    monkeypatch.setattr("aios.services.sessions.invoke", inv)

    out = await invoke_builtin(
        _SELF, "create_goal", {"goal": "one too many", "output_schema": _SCHEMA}
    )

    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str)
    assert "cap" in out.content.lower()
    # No edge opened on a cap rejection.
    inv.assert_not_awaited()


async def test_create_goal_cap_ignores_foreign_obligations(monkeypatch: Any) -> None:
    """Caller-assigned (api/run/peer) obligations don't count toward the self-goal cap."""
    monkeypatch.setattr(
        "aios.tools.goal_management.get_settings", lambda: SimpleNamespace(session_open_goals_max=1)
    )
    _stub_open_obligations(monkeypatch, [_foreign("a1", kind="api"), _foreign("r1", kind="run")])
    inv = AsyncMock(
        return_value=TaskHandle(servicer_kind="session", servicer_id=_SELF, request_id="req_g")
    )
    monkeypatch.setattr("aios.services.sessions.invoke", inv)

    out = await invoke_builtin(
        _SELF, "create_goal", {"goal": "still allowed", "output_schema": _SCHEMA}
    )

    assert isinstance(out, dict)
    assert out["goal_id"] == "req_g"
    inv.assert_awaited_once()


# ─── list_goals ──────────────────────────────────────────────────────────────


async def test_list_goals_filters_to_self_goals(monkeypatch: Any) -> None:
    _stub_open_obligations(
        monkeypatch,
        [
            _self_goal("g1", summary="goal one"),
            _foreign("a1"),
            _self_goal("g2", summary="goal two"),
        ],
    )
    out = await invoke_builtin(_SELF, "list_goals", {})
    assert isinstance(out, dict)
    ids = [g["goal_id"] for g in out["goals"]]
    assert ids == ["g1", "g2"]
    assert out["goals"][0]["goal"] == "goal one"
    assert "age" in out["goals"][0]


async def test_list_goals_empty(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [_foreign("a1")])
    out = await invoke_builtin(_SELF, "list_goals", {})
    assert out == {"goals": []}


# ─── complete_goal / fail_goal ───────────────────────────────────────────────


async def test_complete_goal_closes_via_respond(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [_self_goal("g1")])
    _stub_goal_schema(monkeypatch, _SCHEMA)
    respond = AsyncMock(return_value="responded")
    monkeypatch.setattr("aios.tools.goal_management.respond_to_request", respond)

    out = await invoke_builtin(
        _SELF, "complete_goal", {"goal_id": "g1", "result": {"shipped": True}}
    )

    assert out == {"goal_id": "g1", "status": "completed"}
    assert respond.await_args is not None
    kwargs = respond.await_args.kwargs
    assert kwargs["request_id"] == "g1"
    assert kwargs["is_error"] is False
    # The conforming result is recorded verbatim as the closing response value.
    assert kwargs["result"] == {"shipped": True}


async def test_complete_goal_rejects_nonconforming_result(monkeypatch: Any) -> None:
    """#1512: the result is validated against the goal's output_schema; a
    non-conforming result is rejected with output_schema_violation and no response
    is written (same semantics as the call_* / return output_schema path)."""
    _stub_open_obligations(monkeypatch, [_self_goal("g1")])
    _stub_goal_schema(monkeypatch, _SCHEMA)
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.goal_management.respond_to_request", respond)

    out = await invoke_builtin(
        _SELF, "complete_goal", {"goal_id": "g1", "result": {"shipped": "nope"}}
    )

    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str)
    assert "output_schema_violation" in out.content
    # No response written on a schema mismatch — the goal stays open.
    respond.assert_not_awaited()


async def test_fail_goal_closes_with_error(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [_self_goal("g1")])
    respond = AsyncMock(return_value="responded")
    monkeypatch.setattr("aios.tools.goal_management.respond_to_request", respond)

    out = await invoke_builtin(_SELF, "fail_goal", {"goal_id": "g1", "reason": "infeasible"})

    assert out == {"goal_id": "g1", "status": "failed"}
    assert respond.await_args is not None
    kwargs = respond.await_args.kwargs
    assert kwargs["is_error"] is True
    assert kwargs["error"] == {"message": "infeasible"}


async def test_fail_goal_requires_reason() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_SELF, "fail_goal", {"goal_id": "g1"})


async def test_complete_goal_rejects_unknown_goal_id(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [_self_goal("g1")])
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.goal_management.respond_to_request", respond)

    out = await invoke_builtin(_SELF, "complete_goal", {"goal_id": "nope", "result": {}})

    assert isinstance(out, ToolResult)
    assert out.is_error
    respond.assert_not_awaited()


async def test_complete_goal_rejects_foreign_obligation(monkeypatch: Any) -> None:
    """A caller-assigned (api/run/peer) obligation is NOT closeable via the goal surface
    — that must go through return/error."""
    _stub_open_obligations(monkeypatch, [_foreign("a1")])
    respond = AsyncMock()
    monkeypatch.setattr("aios.tools.goal_management.respond_to_request", respond)

    out = await invoke_builtin(_SELF, "complete_goal", {"goal_id": "a1", "result": {}})

    assert isinstance(out, ToolResult)
    assert out.is_error
    respond.assert_not_awaited()
