"""Pin the ``resolve_provider_auth_or_conflict`` → caller seam (issue #1816).

Every caller (harness/loop.py, workflows/run_llm.py, auto_review, voice
transcription) receives ``auth`` from this one producer and the admission
gates read ``auth.has_credentials``. This suite walks every return path of the
real producer and asserts the returned value is either ``None`` or a real
:class:`ProviderAuth` whose ``has_credentials`` is the expected bool — so a
path that ever returns some other shape (or the attribute is renamed) fails
here, not as a silently wedged run in an integration test whose stub drifted
(the PR #2609 ``(object(), None)`` stub in test_eval_gate.py).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.crypto.vault import KEY_BYTES, CryptoBox
from aios.db.queries import ResolvedModelProvider
from aios.models.model_providers import ProviderAuth
from aios.services import model_providers as service
from tests.unit.conftest import fake_pool_yielding_conn


@pytest.fixture(autouse=True)
def _unit_provider_auth_ungated() -> None:
    """Shadow the conftest autouse stub: this file exercises the real producer."""
    return None


@pytest.fixture(autouse=True)
def _legacy_inference_policy(legacy_env: None) -> None:
    """Legacy policy: no root-inheritance denial unless a case opts in."""


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


def _row(crypto_box: CryptoBox, plaintext: str, **kw: Any) -> ResolvedModelProvider:
    blob = crypto_box.derive_account_subkey("acc_x").encrypt(plaintext)
    return ResolvedModelProvider(owner_account_id="acc_x", api_base=None, blob=blob, **kw)


def _assert_contract(auth: object, expected: bool | None) -> None:
    if expected is None:
        assert auth is None
        return
    assert isinstance(auth, ProviderAuth), type(auth)
    assert isinstance(auth.has_credentials, bool)
    assert auth.has_credentials is expected


async def _call(pool: Any, crypto_box: CryptoBox, model: str = "anthropic/claude-x") -> Any:
    auth, conflict = await service.resolve_provider_auth_or_conflict(
        pool, crypto_box, account_id="acc_x", model=model, litellm_extra=None
    )
    assert conflict is None
    return auth


async def test_unresolvable_model_yields_none(crypto_box: CryptoBox) -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not touch the pool"))
    _assert_contract(await _call(pool, crypto_box, model="totally-bogus-model-xyz"), None)


async def test_no_row_yields_none(crypto_box: CryptoBox) -> None:
    pool = fake_pool_yielding_conn(MagicMock())
    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=None),
    ):
        _assert_contract(await _call(pool, crypto_box), None)


async def test_root_owned_row_denied_to_descendant_yields_none(
    crypto_box: CryptoBox, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aios.config import get_settings

    monkeypatch.setattr(get_settings(), "inference_credential_policy", "account_only")
    blob = crypto_box.derive_account_subkey("acc_root").encrypt(json.dumps({"api_key": "k"}))
    row = ResolvedModelProvider(owner_account_id="acc_root", api_base=None, blob=blob)
    pool = fake_pool_yielding_conn(MagicMock())
    with (
        patch(
            "aios.services.model_providers.queries.resolve_model_provider",
            AsyncMock(return_value=row),
        ),
        patch(
            "aios.services.model_providers.queries.get_account",
            AsyncMock(return_value=MagicMock(parent_account_id=None)),
        ),
    ):
        _assert_contract(await _call(pool, crypto_box), None)


@pytest.mark.parametrize(
    ("plaintext", "expected"),
    [
        pytest.param("sk-bare-0140", True, id="legacy-bare-key"),
        pytest.param(json.dumps({"api_key": "k"}), True, id="json-api-key"),
        pytest.param(
            json.dumps({"aws_access_key_id": "A", "aws_secret_access_key": "s"}),
            True,
            id="json-non-api-key-family",
        ),
        pytest.param(json.dumps({}), False, id="defaults-only-empty-blob"),
        pytest.param(json.dumps(["not", "a", "dict"]), True, id="json-non-dict-as-bare-key"),
    ],
)
async def test_row_paths_yield_provider_auth_with_has_credentials(
    crypto_box: CryptoBox, plaintext: str, expected: bool
) -> None:
    pool = fake_pool_yielding_conn(MagicMock())
    with patch(
        "aios.services.model_providers.queries.resolve_model_provider",
        AsyncMock(return_value=_row(crypto_box, plaintext, litellm_defaults={})),
    ):
        _assert_contract(await _call(pool, crypto_box), expected)


@pytest.mark.parametrize("policy", ["account_only"])
async def test_run_llm_gate_reads_has_credentials_off_a_real_auth(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """The consumer side of the seam: run_llm's admission gate refuses a
    defaults-only ProviderAuth exactly as if no row matched."""
    from aios.config import get_settings
    from aios.workflows import run_llm

    monkeypatch.setattr(get_settings(), "inference_credential_policy", policy)
    defaults_only = ProviderAuth(api_key="", api_base=None, owner_account_id="acc_x")
    assert defaults_only.has_credentials is False
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    with patch(
        "aios.workflows.run_llm.model_providers_service.resolve_provider_auth_or_conflict",
        AsyncMock(return_value=(defaults_only, None)),
    ):
        run = MagicMock(id="run_1", account_id="acc_x")
        out, cost = await run_llm.invoke_call_llm(
            run=run,
            spec={"model": "anthropic/claude-x", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert out.get("error_kind") == "model_provider_not_configured", out
    assert cost == 0
