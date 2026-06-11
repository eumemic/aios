"""Env-var credentials on the provisioning plan (#873).

Two properties of ``build_spec_from_session``:

* the resolved credential set rides ``plan.env_var_credentials`` while
  the container env stays untouched — materialization is the next slice
  (#874), so in this one the placeholder must NOT appear anywhere in
  ``spec.environment``;
* credential resolution runs BEFORE the github-clones step by design —
  its deliberate fail-hard (one corrupt blob aborts the provision) must
  raise while no GitProxy is running and no broker secret is registered,
  otherwise every failed provision would leak a live proxy.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import CryptoDecryptError
from aios.sandbox.spec import build_spec_from_session
from aios.services.vaults import ResolvedEnvVarCredential
from tests.helpers.sandbox import patch_build_spec_deps

_CRED = ResolvedEnvVarCredential(
    credential_id="vcred_01TEST",
    secret_name="GITHUB_TOKEN",
    secret_value="ghp_secret",
    allowed_hosts=("api.github.com",),
    updated_at=datetime(2026, 6, 10, tzinfo=UTC),
    placeholder="AIOS_SECRET_PLACEHOLDER_" + "ab" * 16,
)


async def test_plan_carries_resolved_creds_but_injects_nothing() -> None:
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        plan = await build_spec_from_session("sess_01TEST")

    assert plan.env_var_credentials == (_CRED,)
    # Nothing is materialized into the container in this slice: neither
    # the placeholder nor (ever) the secret may reach the env.
    env_blob = " ".join([*plan.spec.environment.keys(), *plan.spec.environment.values()])
    assert _CRED.placeholder not in env_blob
    assert _CRED.secret_value not in env_blob
    assert "GITHUB_TOKEN" not in plan.spec.environment


async def test_resolve_failure_aborts_before_git_proxy_or_broker_exist() -> None:
    clones = AsyncMock(return_value=([], None))
    broker = MagicMock()
    broker.port = 54321
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(side_effect=CryptoDecryptError("corrupt blob")),
            github_clones=clones,
            tool_broker=broker,
        ):
            stack.enter_context(ctx)

        with pytest.raises(CryptoDecryptError):
            await build_spec_from_session("sess_01TEST")

    # The fail-hard fired while nothing needed cleanup: no proxy was
    # started, no broker secret registered.
    clones.assert_not_awaited()
    broker.register_session.assert_not_called()
