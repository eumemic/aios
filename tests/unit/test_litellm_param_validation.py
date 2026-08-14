from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import litellm
import pytest
from litellm.exceptions import UnsupportedParamsError

from aios.harness import completion
from aios.services.litellm_params import unsupported_openai_params

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_stale_capability_map_is_reported_at_config_save() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"reasoning_effort": "high"}) == [
        "reasoning_effort"
    ]


def test_provider_specific_kwargs_are_not_mislabeled() -> None:
    assert unsupported_openai_params("xai/grok-4.6", {"vendor_bogus": True}) == []


# ─ The unsupported-param path: assert the OUTCOME, not the switch ──────────
#
# ``claude-opus-4-8`` accepts only ``temperature=1``. Asking for 0.8 is the
# concrete case of "a sampling parameter the model rejects". Under
# ``drop_params`` LiteLLM discards it and completes anyway, so the harness
# believes it administered a treatment that never reached the wire — the bug
# in #1674. The tests below drive the real ``call_litellm`` path and assert on
# what the call DID (raised vs. quietly returned), not on the kwargs we built.
#
# ``mock_response`` short-circuits the transport INSIDE LiteLLM, after its
# parameter-validation pipeline. Nothing here is stubbed with a fake that
# could agree with us by construction: the exception is LiteLLM's own.

_REJECTING_MODEL = "anthropic/claude-opus-4-8"
_REJECTED_PARAM = {"temperature": 0.8}


@pytest.fixture
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Settings:
        inference_credential_policy = "account_only"
        model_call_deadline_s = 300.0
        context_admission_mode = "observe"

    monkeypatch.setattr(completion, "get_settings", lambda: _Settings())


@pytest.fixture(autouse=True)
def _restore_litellm_global() -> object:
    """``litellm.drop_params`` is process-global; never leak a mutation."""
    previous = litellm.drop_params
    yield
    litellm.drop_params = previous


async def test_agent_supplied_drop_params_cannot_silence_a_rejected_param(
    _settings: None,
) -> None:
    """An agent asking for ``drop_params`` must not buy itself a silent drop.

    This is the mutation-sensitive test. With ``effective_extra["drop_params"]
    = False`` removed from ``_build_litellm_kwargs``, the agent's ``True``
    reaches LiteLLM, the call SUCCEEDS, and ``temperature`` is discarded en
    route — exactly the silent divergence #1674 is about.
    """
    request = completion.LlmRequest(
        messages=[{"role": "user", "content": "hi"}],
        params={**_REJECTED_PARAM, "drop_params": True, "mock_response": "ok"},
    )

    with pytest.raises(UnsupportedParamsError, match="does not support temperature"):
        await completion.call_litellm(request, model=_REJECTING_MODEL)


async def test_rejected_sampling_param_fails_loud_by_default(_settings: None) -> None:
    """The ordinary path (agent supplies no ``drop_params``) also fails loud."""
    request = completion.LlmRequest(
        messages=[{"role": "user", "content": "hi"}],
        params={**_REJECTED_PARAM, "mock_response": "ok"},
    )

    with pytest.raises(UnsupportedParamsError, match="does not support temperature"):
        await completion.call_litellm(request, model=_REJECTING_MODEL)


async def test_supported_sampling_param_still_reaches_the_wire(_settings: None) -> None:
    """Fail-loud must not mean fail-always: an accepted value still completes.

    Without this, both tests above would pass on a harness that simply broke
    every call.
    """
    request = completion.LlmRequest(
        messages=[{"role": "user", "content": "hi"}],
        params={"temperature": 1, "mock_response": "ok"},
    )

    response = await completion.call_litellm(request, model=_REJECTING_MODEL)

    assert response.content == "ok"


def test_module_import_pins_the_litellm_drop_params_global() -> None:
    """Importing the harness must force ``litellm.drop_params`` False.

    Load-bearing and NOT covered by the tests above: LiteLLM's global WINS
    over a per-call ``drop_params=False``. Measured against litellm 1.96.2::

        global=False, per-call=False -> raises      (what we want)
        global=True,  per-call=False -> SILENT DROP (per-call flag loses)

    So if that global were ever True — a future LiteLLM default, or any other
    importer setting it — ``_build_litellm_kwargs`` would NOT save us.

    Asserting ``litellm.drop_params is False`` in-process is decoration: the
    import already happened at collection, so it passes with the module-level
    line deleted. This runs a fresh interpreter that sets the global True
    FIRST, then imports the harness, and checks the import clamped it back.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; litellm.drop_params = True;"
            " import aios.harness.completion;"
            " print(litellm.drop_params)",
        ],
        capture_output=True,
        text=True,
        env={*os.environ, "PYTHONPATH": str(_SRC)},
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False", probe.stdout


def test_build_kwargs_overrides_agent_supplied_drop_params() -> None:
    """Cheap unit check on our side of the boundary.

    Retained as a fast localizer — it says WHERE the override happens — but it
    is no longer the only evidence; the behavioural tests above are.
    """
    kwargs = completion._build_litellm_kwargs(
        model=_REJECTING_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        auth=None,
        extra={**_REJECTED_PARAM, "drop_params": True},
        session_id=None,
        stream=False,
    )

    assert kwargs["drop_params"] is False
