"""Env-var credential materialization into the sandbox env (#873 + #874).

Three properties of ``build_spec_from_session``:

* #874: a session's ``environment_variable`` secret surfaces in the
  container env as ``secret_name=<opaque placeholder>``; the decrypted
  secret reaches neither the env nor anywhere else on the spec;
* #874: the placeholder WINS over an operator/session env var of the
  same name — the security-load-bearing precedence (a plaintext value an
  operator set under that name must not reach the container);
* #873: credential resolution runs BEFORE the github-clones step, so its
  deliberate fail-hard (one corrupt blob aborts the provision) raises
  while no GitProxy is running and no broker secret is registered —
  otherwise every failed provision would leak a live proxy.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import CryptoDecryptError
from aios.models.environments import EnvironmentConfig
from aios.models.vaults import RESERVED_SANDBOX_ENV_KEYS
from aios.sandbox.spec import build_spec_from_session
from aios.services.vaults import ResolvedEnvVarCredential
from tests.helpers.sandbox import patch_build_spec_deps

# A distinctive sentinel rather than a real-looking token, so a leak is
# unambiguous in any diff (and never a plausible credential in CI logs).
_SENTINEL_SECRET = "SENTINEL_PLAINTEXT_DO_NOT_LEAK"
_CRED = ResolvedEnvVarCredential(
    credential_id="vcred_01TEST",
    secret_name="GITHUB_TOKEN",
    secret_value=_SENTINEL_SECRET,
    allowed_hosts=("api.github.com",),
    updated_at=datetime(2026, 6, 10, tzinfo=UTC),
    placeholder="AIOS_SECRET_PLACEHOLDER_" + "ab" * 16,
)


async def test_placeholder_lands_in_env_secret_never_does() -> None:
    """The credential's ``secret_name`` surfaces set to its opaque
    placeholder; the decrypted secret appears nowhere on the spec."""
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        plan = await build_spec_from_session("sess_01TEST")

    # The placeholder IS the value of the secret_name key...
    assert plan.spec.environment["GITHUB_TOKEN"] == _CRED.placeholder
    # ...and the key set is exactly the reserved keys plus the injected
    # secret_name — nothing extra leaked into it, and the secret_name is
    # disjoint from the reserved set (blocked from claiming one at create).
    assert set(plan.spec.environment) == {*RESERVED_SANDBOX_ENV_KEYS, "GITHUB_TOKEN"}
    # The decrypted secret reaches NOTHING on the spec. Membership, not
    # dataclass ``==``: secret_value is repr=False, but an equality diff
    # across differing secrets would still render the plaintext.
    assert _SENTINEL_SECRET not in str(plan.spec)
    assert _SENTINEL_SECRET not in " ".join(plan.spec.environment.values())
    # The decrypted creds still ride the plan for #876's egress swap
    # (the secret is off the spec but must remain in worker memory).
    assert plan.env_var_credentials == (_CRED,)


@pytest.mark.parametrize("layer", ["env_config", "session_env"])
async def test_vault_placeholder_wins_over_operator_env(
    layer: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The security-load-bearing precedence: a plaintext value an operator
    set under the same name as a vault ``secret_name`` must NOT reach the
    container — the opaque placeholder wins, and the override is logged.

    Both operator-env layers are checked: the environment config's
    ``env`` and the per-session ``env`` override (they merge into one
    ``operator_env`` layer that the placeholder shadows)."""
    leak = "operator-plaintext-leak-do-not-ship"
    env_config = EnvironmentConfig(env={"GITHUB_TOKEN": leak}) if layer == "env_config" else None
    session_env = {"GITHUB_TOKEN": leak} if layer == "session_env" else None
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_config=env_config,
            session_env=session_env,
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        with caplog.at_level(logging.WARNING):
            plan = await build_spec_from_session("sess_01TEST")

    assert plan.spec.environment["GITHUB_TOKEN"] == _CRED.placeholder
    assert leak not in str(plan.spec)
    # The override is logged — the colliding KEY name, never the operator's
    # plaintext value (which is itself a secret an operator may have set).
    shadow_log = next(
        r.getMessage()
        for r in caplog.records
        if "sandbox.vault_placeholder_shadows_env" in r.getMessage()
    )
    assert "GITHUB_TOKEN" in shadow_log
    assert leak not in shadow_log


async def test_no_creds_leaves_env_untouched() -> None:
    """The empty case is byte-identical to a no-vault provision: the env
    is exactly the reserved keys, nothing injected."""
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=()),
        ):
            stack.enter_context(ctx)
        plan = await build_spec_from_session("sess_01TEST")

    assert set(plan.spec.environment) == RESERVED_SANDBOX_ENV_KEYS


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
