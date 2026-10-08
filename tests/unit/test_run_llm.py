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
from aios.harness.completion import LlmResponse, ModelCallDeadlineError, estimate_cost_usd
from aios.models.model_providers import ProviderAuth
from aios.services.requests import Missing, Rebuilt
from aios.workflows import run_llm
from aios.workflows.run_llm import _to_microusd, invoke_call_llm
from aios.workflows.wf_script_host import call_llm


@pytest.fixture(autouse=True)
def _legacy_inference_policy(legacy_env: None) -> None:
    """This suite intentionally reaches model behavior beyond credential admission."""


@pytest.fixture(autouse=True)
def _stub_provider_auth_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard 3 (provider-auth conflict) needs a worker context (pool/crypto_box)
    and hits the DB. Stub it to a clean pass — no resolved row, no conflict —
    so tests exercising the OTHER guards don't need to know about it. Tests
    for Guard 3 itself override ``resolve_provider_auth_or_conflict`` directly.
    """
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr(
        "aios.services.model_providers.resolve_provider_auth_or_conflict",
        AsyncMock(return_value=(None, None)),
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


async def test_unconfigured_provider_fails_closed() -> None:
    with (
        patch(
            "aios.services.model_providers.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(None, None)),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as model_call,
        patch(
            "aios.workflows.run_llm.get_settings",
            return_value=SimpleNamespace(inference_credential_policy="account_only"),
        ),
    ):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())

    assert result["error_kind"] == "model_provider_not_configured"
    assert cost == 0
    model_call.assert_not_awaited()


async def test_provider_auth_conflict_rejected() -> None:
    with (
        patch(
            "aios.services.model_providers.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(None, "conflict message")),
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
            "aios.services.model_providers.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(auth, None)),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=_response())) as m,
    ):
        result, _ = await invoke_call_llm(run=_run(), spec=_spec())
    assert "error" not in result
    assert m.await_args is not None
    assert m.await_args.kwargs["auth"] is auth


async def test_provider_auth_resolution_raise_is_recoverable_value() -> None:
    """Guard 3 does I/O + crypto, so unlike guards 1-2 it CAN raise (a corrupt
    ciphertext row → CryptoDecryptError). invoke_call_llm's 'never raises'
    contract requires that to become a recoverable {"error": ...} value —
    an uncaught raise escapes _run_call_llm_task (no outer except) with no
    result signal, and the sweep re-dispatches forever (silent wedge)."""
    from aios.errors import CryptoDecryptError

    with (
        patch(
            "aios.services.model_providers.resolve_provider_auth_or_conflict",
            AsyncMock(side_effect=CryptoDecryptError("corrupt row")),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m,
    ):
        result, cost = await invoke_call_llm(run=_run(), spec=_spec())
    assert "error" in result and "resolution failed" in result["error"]
    assert cost == 0
    m.assert_not_awaited()  # the inference never ran


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


# ─── by reference (#2474) ─────────────────────────────────────────────────────

_REF = {"session_id": "ses_1", "request_id": "evt_1"}
_CAPTURED_PARAMS = {"api_base": "https://proxy.internal/v1", "temperature": 0.3}


def _rebuilt(*, captured_model: str = "openrouter/captured") -> Rebuilt:
    return Rebuilt(
        request={
            "messages": [{"role": "user", "content": "from the log"}],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "params": _CAPTURED_PARAMS,
        },
        fidelity="exact",
        record={"model": captured_model},
    )


def _ref_run(principal: str) -> Any:
    return SimpleNamespace(
        id="wfr_1", account_id="acc_t", default_child_model="gpt-4o-mini", principal=principal
    )


async def _call_by_ref(
    run: Any, rebuilt: Any, *, model: str | None
) -> tuple[dict[str, Any], int, Any]:
    with (
        patch("aios.workflows.run_llm.rebuild_request", AsyncMock(return_value=rebuilt)) as rb,
        patch("aios.workflows.run_llm.call_litellm", AsyncMock(return_value=_response())) as m,
    ):
        result, cost = await invoke_call_llm(
            run=run, spec={"kind": "ref", "request_ref": _REF, "model": model}
        )
    assert rb.await_args is not None
    assert rb.await_args.kwargs["target_model"] == (model or run.default_child_model)
    return result, cost, m


async def test_by_ref_with_the_captured_model_keeps_its_params_and_endpoint() -> None:
    """The captured api_base passed #823 for its launcher, so it's admitted for the
    same model even though it is on no allowlist."""
    result, cost, m = await _call_by_ref(
        _ref_run("session"), _rebuilt(), model="openrouter/captured"
    )
    assert result["fidelity"] == "exact"  # reported at use, since a sample doesn't rebuild
    request = m.await_args.args[0]
    assert request.messages == [{"role": "user", "content": "from the log"}]
    assert request.tools == [{"type": "function", "function": {"name": "t"}}]
    assert request.params == _CAPTURED_PARAMS
    assert request.session_id == "ses_1"  # a run acting for the session shares its key
    assert cost == 2000


async def test_by_ref_with_another_model_sends_no_captured_params() -> None:
    """Another model must not reach the captured endpoint with its own key."""
    _, _, m = await _call_by_ref(_ref_run("operator"), _rebuilt(), model="openrouter/judge")
    request = m.await_args.args[0]
    assert request.params is None
    assert request.session_id == "wfr_1"  # an operator run gets its own cache key


async def test_by_ref_of_a_workflow_capture_keeps_params_but_not_their_endpoint() -> None:
    """A request captured for a ``workflow:`` binding handed its params to the bound
    run, so another model keeps them; nothing vouches for the endpoint, so an
    ``api_base`` in them must be allowlisted, as inline."""
    workflow_capture = _rebuilt(captured_model="workflow:wf_cand@3")
    result, cost, m = await _call_by_ref(
        _ref_run("operator"), workflow_capture, model="openrouter/baseline"
    )
    assert "untrusted endpoint" in result["error"]
    assert cost == 0
    m.assert_not_awaited()

    no_endpoint = Rebuilt(
        request={**workflow_capture.request, "params": {"temperature": 0.3}},
        fidelity="exact",
        record=workflow_capture.record,
    )
    _, _, m = await _call_by_ref(_ref_run("operator"), no_endpoint, model="openrouter/baseline")
    assert m.await_args.args[0].params == {"temperature": 0.3}


async def test_by_ref_defaults_to_the_runs_default_child_model() -> None:
    _, _, m = await _call_by_ref(_ref_run("operator"), _rebuilt(), model=None)
    assert m.await_args.kwargs["model"] == "gpt-4o-mini"


async def test_by_ref_an_unavailable_request_is_an_error_value() -> None:
    result, cost, m = await _call_by_ref(
        _ref_run("operator"), Missing(what="blob", record={}), model="openrouter/judge"
    )
    assert result["error_kind"] == "request_unavailable"
    assert cost == 0
    m.assert_not_awaited()


async def test_by_ref_a_failing_rebuild_is_an_error_value_not_a_raise() -> None:
    """``invoke_call_llm`` never raises: an escape would leave no result signal and the
    sweep would re-dispatch the call forever."""
    with (
        patch(
            "aios.workflows.run_llm.rebuild_request",
            AsyncMock(side_effect=KeyError("slate")),
        ),
        patch("aios.workflows.run_llm.call_litellm", AsyncMock()) as m,
    ):
        result, cost = await invoke_call_llm(
            run=_ref_run("operator"),
            spec={"kind": "ref", "request_ref": _REF, "model": "openrouter/judge"},
        )
    assert "rebuilding the request failed" in result["error"]
    assert cost == 0
    m.assert_not_awaited()


async def test_by_ref_rejects_a_workflow_model_before_rebuilding() -> None:
    with patch("aios.workflows.run_llm.rebuild_request", AsyncMock()) as rb:
        result, _ = await invoke_call_llm(
            run=_ref_run("operator"),
            spec={"kind": "ref", "request_ref": _REF, "model": "workflow:wf_x"},
        )
    assert "workflow:" in result["error"]
    rb.assert_not_awaited()


def test_call_llm_shim_by_ref() -> None:
    cap = call_llm(request_ref=dict(_REF), model="m")
    assert cap._spec == {"kind": "ref", "request_ref": _REF, "model": "m"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request": {"messages": []}, "request_ref": _REF},
        {"request_ref": {"session_id": "s"}},
        {"request_ref": {**_REF, "extra": "x"}},
        {"request_ref": {"session_id": "s", "request_id": 1}},
        {"request": {"messages": []}, "model": "m"},
    ],
)
def test_call_llm_shim_rejects_a_malformed_ref_call(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        call_llm(**kwargs)


def test_inline_call_llm_spec_is_unchanged() -> None:
    """An inline call's spec, and so its call key, is what it was before refs."""
    cap = call_llm({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert set(cap._spec) == {"model", "messages", "tools", "params", "session_id"}


# ─── sub_runs() uncached cost (#2476) ─────────────────────────────────────────


def test_price_uncached_prices_every_usage_entry_without_the_cache() -> None:
    facts = {
        "nodes": [
            {
                "usage": [
                    {
                        "model": "anthropic/claude-sonnet-4-5",
                        "input_tokens": 1000,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 900,
                        "cache_creation_input_tokens": 0,
                        "cost_microusd": 500,
                    },
                    {"model": None, "input_tokens": 5, "output_tokens": 5},
                ]
            }
        ],
        "truncated": False,
    }
    with patch("aios.workflows.run_llm.estimate_cost_usd", return_value=0.0042) as est:
        priced = run_llm.price_uncached(facts)
    # Cache counters are left out, so every prompt token is priced as input.
    est.assert_called_once_with(
        "anthropic/claude-sonnet-4-5", {"input_tokens": 1000, "output_tokens": 10}
    )
    usage = priced["nodes"][0]["usage"]
    assert usage[0]["uncached_cost_microusd"] == 4200
    assert usage[0]["cost_microusd"] == 500
    assert usage[1]["uncached_cost_microusd"] is None
    assert isinstance(priced["litellm_version"], str)


def test_price_uncached_is_none_for_a_model_outside_the_cost_map() -> None:
    facts = {
        "nodes": [{"usage": [{"model": "nope/unknown", "input_tokens": 1, "output_tokens": 1}]}]
    }
    assert run_llm.price_uncached(facts)["nodes"][0]["usage"][0]["uncached_cost_microusd"] is None


def test_price_uncached_against_the_real_cost_map_ignores_the_cache_discount() -> None:
    # Unpatched: pins the premise that litellm counts cache tokens inside the prompt
    # total, so pricing input_tokens with no cache detail is the uncached price.
    model = "anthropic/claude-sonnet-4-5"
    usage = {
        "model": model,
        "input_tokens": 1000,
        "output_tokens": 10,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
    }
    priced = run_llm.price_uncached({"nodes": [{"usage": [usage]}]})
    uncached = priced["nodes"][0]["usage"][0]["uncached_cost_microusd"]
    cached = estimate_cost_usd(
        model, {"input_tokens": 1000, "output_tokens": 10, "cache_read_input_tokens": 900}
    )
    assert cached is not None
    assert uncached is not None
    assert uncached > _to_microusd(cached)


def test_price_uncached_looks_up_an_unknown_model_once() -> None:
    usage = [{"model": "nope/unknown", "input_tokens": i, "output_tokens": 1} for i in range(3)]
    with patch("aios.workflows.run_llm.estimate_cost_usd", return_value=None) as est:
        priced = run_llm.price_uncached({"nodes": [{"usage": usage}]})
    est.assert_called_once()
    assert all(u["uncached_cost_microusd"] is None for u in priced["nodes"][0]["usage"])
