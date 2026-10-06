"""``get_request`` (#2475): one ref rebuilt inline, with the by-ref params rule.

The grant check and the rebuild are stubbed; the integration tests cover both.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.errors import NotFoundError
from aios.services.requests import Missing, Rebuilt
from aios.workflows import run_replay

_REF = {"session_id": "ses_1", "request_id": "evt_1"}
_PARAMS = {"api_base": "https://proxy.internal/v1", "temperature": 0.3}


class _Pool:
    def acquire(self) -> Any:
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *exc: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: _Pool())


def _rebuilt(fidelity: str = "exact") -> Rebuilt:
    return Rebuilt(
        request={"messages": [{"role": "user", "content": "q"}], "tools": None, "params": _PARAMS},
        fidelity=fidelity,  # type: ignore[arg-type]
        record={"model": "openrouter/prod", "capability_model": "openrouter/prod"},
    )


async def _get(
    rebuilt: Any, *, model: str | None = None, granted: bool = True
) -> tuple[dict[str, Any], Any]:
    args: dict[str, Any] = {"request_ref": _REF}
    if model is not None:
        args["model"] = model
    rebuild = (
        AsyncMock(side_effect=rebuilt)
        if isinstance(rebuilt, Exception)
        else AsyncMock(return_value=rebuilt)
    )
    with (
        patch.object(run_replay, "request_ref_granted", AsyncMock(return_value=granted)),
        patch.object(run_replay, "rebuild_request", rebuild),
    ):
        result = await run_replay.invoke_replay_tool(
            run=SimpleNamespace(id="wfr_1", account_id="acc_1"),  # type: ignore[arg-type]
            tool_name="get_request",
            args=args,
        )
    return result, rebuild


async def test_the_captured_model_keeps_the_captured_params() -> None:
    result, rebuild = await _get(_rebuilt())
    assert result == {
        "messages": [{"role": "user", "content": "q"}],
        "tools": None,
        "params": _PARAMS,
        "fidelity": "exact",
    }
    assert rebuild.await_args.kwargs["target_model"] is None


async def test_another_model_gets_no_captured_params() -> None:
    result, rebuild = await _get(_rebuilt("rerendered"), model="openrouter/judge")
    assert result["params"] is None
    assert result["fidelity"] == "rerendered"
    assert rebuild.await_args.kwargs["target_model"] == "openrouter/judge"


@pytest.mark.parametrize(
    "rebuilt",
    [Missing(what="blob", record={}), NotFoundError("gone"), RuntimeError("decode")],
)
async def test_an_unavailable_request_is_an_error_value(rebuilt: Any) -> None:
    result, _ = await _get(rebuilt)
    assert result["error_kind"] == "request_unavailable"


async def test_an_ungranted_ref_is_refused_before_any_rebuild() -> None:
    result, rebuild = await _get(_rebuilt(), granted=False)
    assert result["error_kind"] == "request_ref_not_granted"
    rebuild.assert_not_awaited()


async def test_a_workflow_model_is_refused() -> None:
    result, rebuild = await _get(_rebuilt(), model="workflow:wf_1")
    assert "not a workflow" in result["error"]
    rebuild.assert_not_awaited()


async def test_bad_arguments_are_an_error_value() -> None:
    result = await run_replay.invoke_replay_tool(
        run=SimpleNamespace(id="wfr_1", account_id="acc_1"),  # type: ignore[arg-type]
        tool_name="get_request",
        args={"request_ref": {"session_id": "s"}},
    )
    assert "bad arguments" in result["error"]
