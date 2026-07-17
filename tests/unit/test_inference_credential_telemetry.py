from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.services import inference_credential_telemetry as telemetry
from aios.services import model_providers
from tests.unit.conftest import fake_pool_yielding_conn


async def test_observe_legacy_env_emits_log_and_metric() -> None:
    pool = fake_pool_yielding_conn(MagicMock())
    before = telemetry.observed_total()
    with (
        patch.object(
            model_providers,
            "get_settings",
            return_value=SimpleNamespace(
                inference_credential_policy="observe_legacy_env", tenancy_posture="internal"
            ),
        ),
        patch.object(queries, "resolve_model_provider", AsyncMock(return_value=None)),
        patch.object(telemetry.log, "warning") as warning,
    ):
        auth = await model_providers._resolve_provider_auth(
            pool,
            CryptoBox(os.urandom(32)),
            account_id="acc_child",
            model="anthropic/claude-x",
            litellm_extra=None,
        )
    assert auth is None
    assert telemetry.observed_total() == before + 1
    warning.assert_called_once_with(
        "inference_credentials.legacy_env_fallback",
        account_id="acc_child",
        provider="anthropic",
        policy="observe_legacy_env",
    )
