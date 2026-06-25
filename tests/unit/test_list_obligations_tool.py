"""Unit tests for the ``list_obligations`` model tool (#1522).

The INCOMING obligations view — the source-agnostic replacement for ``list_goals``
(which only listed self-caller goals). Stubs the worker pool + services so they
need no live Postgres. Covered:

* **identity invariant** — a smuggled ``caller``/``account_id`` argument is
  rejected by the tool schema (``additionalProperties: false``) before the handler
  runs (the trusted identity is the harness-supplied executing session).
* **source-agnostic listing** — a self-goal AND a caller-assigned (api/run/peer)
  obligation BOTH appear, each with the correct ``origin`` and ``output_schema``
  (the #1522 acceptance criterion (i)).
* **registration** — registered model-only (``transport="agent_tool"``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.sessions import Obligation
from aios.tools.invoke import ToolBail, invoke_builtin

_SELF = "ses_self"
_ACCOUNT = "acc_x"

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"shipped": {"type": "boolean"}},
    "required": ["shipped"],
}


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )


def _self_goal(rid: str) -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind="session",
        caller_id=_SELF,
        opened_at=datetime.now(UTC),
        summary="ship the thing",
        output_schema=_SCHEMA,
    )


def _foreign(rid: str, *, kind: str = "api") -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind=kind,
        caller_id="someone_else",
        opened_at=datetime.now(UTC),
        summary="a caller-assigned task",
        output_schema=None,
    )


def _stub_open_obligations(monkeypatch: Any, obligations: list[Obligation]) -> None:
    monkeypatch.setattr("aios.db.queries.get_open_obligations", AsyncMock(return_value=obligations))

    class _Conn:
        async def __aenter__(self) -> Any:
            return object()

        async def __aexit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(
        "aios.harness.runtime.require_pool", lambda: SimpleNamespace(acquire=lambda: _Conn())
    )


async def test_arg_schema_forbids_smuggled_caller() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_SELF, "list_obligations", {"caller": {"kind": "session", "id": "x"}})


async def test_source_agnostic_self_and_caller_assigned(monkeypatch: Any) -> None:
    """#1522 acceptance (i): a self-goal AND a caller-assigned obligation both
    appear, with correct ``origin`` and ``output_schema``."""
    _stub_open_obligations(monkeypatch, [_self_goal("g1"), _foreign("a1", kind="run")])
    out = await invoke_builtin(_SELF, "list_obligations", {})
    assert isinstance(out, dict)
    rows = out["obligations"]
    assert [r["request_id"] for r in rows] == ["g1", "a1"]

    by_id = {r["request_id"]: r for r in rows}
    # self-goal: origin=self, carries its acceptance contract
    assert by_id["g1"]["origin"] == "self"
    assert by_id["g1"]["caller_kind"] == "session"
    assert isinstance(by_id["g1"]["output_schema"], str)
    assert "shipped" in by_id["g1"]["output_schema"]
    # caller-assigned run task: origin=run, schemaless here
    assert by_id["a1"]["origin"] == "run"
    assert by_id["a1"]["caller_kind"] == "run"
    assert by_id["a1"]["output_schema"] is None


async def test_empty(monkeypatch: Any) -> None:
    _stub_open_obligations(monkeypatch, [])
    out = await invoke_builtin(_SELF, "list_obligations", {})
    assert out == {"obligations": []}


def test_registered_model_only() -> None:
    from aios.tools.registry import registry

    assert registry.has("list_obligations")
    spec = registry.get("list_obligations")
    assert spec.transport == "agent_tool"
