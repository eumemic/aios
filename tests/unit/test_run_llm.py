"""run_llm: the worker-side resolver for a workflow run's ``call_llm()`` (#1633).

Pure in-memory — the run is a stand-in carrying only the fields the resolver
reads (``account_id``, ``default_child_model``), and ``call_litellm`` is mocked
so no provider call leaves the process. These cover the four runtime guards
(``workflow:`` rejection, the api_base clamp, model resolution, the
provider-auth conflict guard), the raw-turn projection, the "errors are
values" contract, and the cost-meter charge.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.config import Settings
from aios.harness.completion import LlmResponse, ModelCallDeadlineError
from aios.models.model_providers import ProviderAuth
from aios.workflows import run_llm
from aios.workflows.run_llm import _to_microusd, invoke_call_llm
from aios.workflows.wf_script_host import call_llm


@pytest.fixture(autouse=True)
def _stub_provider_auth_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard 3 (provider-auth conflict) needs a worker context (pool/crypto_box)
    and hits the DB. Stub it to a clean pass — no resolved row, no conflict —
    so tests exercising the OTHER guards don't need to know about it. Tests
    for Guard 3 itself override ``resolve_provider_auth``/
    ``check_provider_auth_conflict`` directly.
    """
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr(
        "aios.services.model_providers.resolve_provider_auth", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "aios.services.model_providers.check_provider_auth_conflict", AsyncMock(return_value=None)
    )


def _run(*, default_child_model: str | None = "gpt-4o-mini") -> Any:
    return SimpleNamespace(id="wfr_1", account_id="acc_t", default_child_model=default_child_model)


def _spec(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": None,
        "params": None,
        "session_id": None,
    }
    base.update(over)
    return base


def _response(*, content: str = "hi", cost: float | None = 0.002) -> LlmResponse:
    return LlmResponse.from_message(
        {"role": "assistant", "content": content},
        usage={"input_tokens": 10, "output_tokens": 5},
        cost=cost,
        finish_reason="stop",
    )


# ─── the author shim ──────────────────────────────────────────────────────────


def test_call_llm_shim_emits_capability() -> None:
    cap = call_llm({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    # Mirrors tool(): the credential-free script only emits the frontier.
    assert cap._capability_id == "call_llm"
    assert cap._spec["model"] == "m"
    assert cap._spec["messages"] == [{"role": "user", "content": "x"}]


def test_call_llm_shim_rejects_missing_messages() -> None:
    # A malformed request is a deterministic (replay-identical) author error.
    import pytest

    with pytest.raises(ValueError, match="messages"):
        call_llm({"model": "m"})


# ─── the worker resolver: the raw turn ────────────────────────────────────────


async def test_returns_raw_assistant_turn() -> None:
    resp = _response()
    with patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=resp)) as m:
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    # The RAW turn: content + (unexecuted) tool_calls + finish_reason + usage + cost.
    assert result == {
        "content": "hi",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "cost": 0.002,
        "message": {"role": "assistant", "content": "hi"},
    }
    assert cost == 2000  # 0.002 USD → 2000 micro-USD (charged at the inference site)
    # The model string is a binding concern passed alongside the request payload.
    assert m.await_args is not None
    assert m.await_args.kwargs["model"] == "gpt-4o-mini"


async def test_unexecuted_tool_calls_passthrough() -> None:
    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    resp = LlmResponse.from_message(
        {"role": "assistant", "content": "", "tool_calls": tc},
        usage={"input_tokens": 1, "output_tokens": 1},
        cost=None,
        finish_reason="tool_calls",
    )
    with patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=resp)):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    # call_llm returns the requested calls UNEXECUTED — the script decides what to do.
    assert result["tool_calls"] == tc
    assert result["finish_reason"] == "tool_calls"
    assert cost == 0  # provider reported no cost → charge 0


# ─── guard 1: workflow: rejection (leaf-only) ─────────────────────────────────


async def test_workflow_model_target_rejected() -> None:
    with patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m:
        result, cost = await invoke_call_llm(run=_run(), spec=_spec(model="workflow:wf_x"))
    assert "error" in result and "workflow:" in result["error"]
    assert cost == 0
    m.assert_not_awaited()  # the inference never ran


async def test_no_model_anywhere_is_recoverable_error() -> None:
    with patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m:
        result, cost = await invoke_call_llm(
            run=_run(default_child_model=None), spec=_spec(model=None)
        )
    assert "error" in result and "model" in result["error"]
    assert cost == 0
    m.assert_not_awaited()


async def test_model_defaults_to_run_default_child_model() -> None:
    with patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=_response())) as m:
        await invoke_call_llm(run=_run(default_child_model="claude-x"), spec=_spec(model=None))
    assert m.await_args is not None
    assert m.await_args.kwargs["model"] == "claude-x"


# ─── guard 2: the model-identity (api_base) clamp ─────────────────────────────


async def test_untrusted_api_base_rejected() -> None:
    settings = Settings(trusted_inference_api_bases=[])
    with (
        patch("aios.services.attenuation.get_settings", return_value=settings),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m,
    ):
        result, cost = await invoke_call_llm(
            run=_run(), spec=_spec(params={"api_base": "https://evil.example"})
        )
    assert "error" in result and "untrusted" in result["error"]
    assert cost == 0
    m.assert_not_awaited()


async def test_trusted_api_base_admitted() -> None:
    settings = Settings(trusted_inference_api_bases=["https://ok.example"])
    with (
        patch("aios.services.attenuation.get_settings", return_value=settings),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=_response())) as m,
    ):
        result, _ = await invoke_call_llm(
            run=_run(), spec=_spec(params={"api_base": "https://ok.example"})
        )
    assert "error" not in result
    # params (carrying the trusted api_base) round-trips into the LlmRequest.
    assert m.await_args is not None
    assert m.await_args.args[0].params == {"api_base": "https://ok.example"}


# ─── guard 3: provider-auth conflict ──────────────────────────────────────────


async def test_provider_auth_conflict_rejected() -> None:
    with (
        patch(
            "aios.services.model_providers.check_provider_auth_conflict",
            AsyncMock(return_value="conflict message"),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m,
    ):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    assert result == {"error": "call_llm refused: conflict message"}
    assert cost == 0
    m.assert_not_awaited()  # the inference never ran


async def test_resolved_auth_forwarded_to_call_litellm() -> None:
    auth = ProviderAuth(api_key="sk-resolved", api_base=None, owner_account_id="acc_t")
    with (
        patch(
            "aios.services.model_providers.resolve_provider_auth",
            AsyncMock(return_value=auth),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=_response())) as m,
    ):
        result, _ = await invoke_call_llm(run=_run(), spec=_spec())
    assert "error" not in result
    assert m.await_args is not None
    assert m.await_args.kwargs["auth"] is auth


# ─── errors are values ────────────────────────────────────────────────────────


async def test_provider_error_is_recoverable_value() -> None:
    with patch(
        "aios.workflows.run_llm.call_litellm",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    assert "error" in result and "boom" in result["error"]
    assert cost == 0  # a failed call bought nothing


async def test_deadline_error_charges_partial_estimate() -> None:
    exc = ModelCallDeadlineError(
        "deadline",
        usage={"input_tokens": 100, "output_tokens": 50},
        cost_usd=None,
        chunks_seen=0,
    )
    with (
        patch("aios.workflows.run_llm.call_litellm", AsyncMock(side_effect=exc)),
        patch("aios.workflows.run_llm.estimate_cost_usd", return_value=0.001),
    ):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    assert "error" in result and "timed out" in result["error"]
    # A timeout still spent provider time — charge the estimate so budget can't be dodged.
    assert cost == 1000


# ─── cost-meter unit ──────────────────────────────────────────────────────────


def test_to_microusd() -> None:
    assert _to_microusd(0.002) == 2000
    assert _to_microusd(None) == 0
    assert _to_microusd(0) == 0
    assert _to_microusd(-1) == 0


def test_has_inflight_false_when_unknown() -> None:
    assert run_llm.has_inflight("wfr_x", "sha:k#0") is False
