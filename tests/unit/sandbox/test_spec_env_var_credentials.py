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
from collections.abc import Iterable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import CryptoDecryptError
from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    UnrestrictedNetworking,
)
from aios.models.vaults import RESERVED_SANDBOX_ENV_KEYS
from aios.sandbox.spec import build_spec_from_session
from aios.services.vaults import ResolvedEnvVarCredential
from tests.helpers.sandbox import limited_env, patch_build_spec_deps

# A distinctive sentinel rather than a real-looking token, so a leak is
# unambiguous in any diff (and never a plausible credential in CI logs).
_SENTINEL_SECRET = "SENTINEL_PLAINTEXT_DO_NOT_LEAK"


def _cred(allowed_hosts: Iterable[str]) -> ResolvedEnvVarCredential:
    return ResolvedEnvVarCredential(
        credential_id="vcred_01TEST",
        secret_name="GITHUB_TOKEN",
        secret_value=_SENTINEL_SECRET,
        allowed_hosts=tuple(allowed_hosts),
        updated_at=datetime(2026, 6, 10, tzinfo=UTC),
        placeholder="AIOS_SECRET_PLACEHOLDER_" + "ab" * 16,
    )


# A credential scoped to api.github.com, plus a Limited env that contains it —
# the #879 gate now requires both, so every test injecting this cred must also
# supply a containing env (or assert the gate's rejection).
_CRED = _cred(["api.github.com"])


# The default containing env for the _CRED above.
_LIMITED_GITHUB = limited_env("api.github.com")


async def test_placeholder_lands_in_env_secret_never_does() -> None:
    """The credential's ``secret_name`` surfaces set to its opaque
    placeholder; the decrypted secret appears nowhere on the spec."""
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            # The #879 gate now requires a containing Limited env for any
            # env-var credential — supply one that covers the cred's host.
            env_config=_LIMITED_GITHUB,
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
    # Both layers carry a containing Limited env (the #879 gate requires one);
    # the env_config layer also stuffs the leak into ``env`` to prove the
    # placeholder still wins over an operator value set under the same name.
    networking = LimitedNetworking(type="limited", allowed_hosts=["api.github.com"])
    env_config = (
        EnvironmentConfig(networking=networking, env={"GITHUB_TOKEN": leak})
        if layer == "env_config"
        else EnvironmentConfig(networking=networking)
    )
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


# ── #879 provision gate (the authoritative gate) ─────────────────────────────


async def _build_with(*, env_config: object, creds: tuple[ResolvedEnvVarCredential, ...]) -> object:
    """Run ``build_spec_from_session`` with the given env config + creds."""
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_config=env_config,
            env_var_credentials=AsyncMock(return_value=creds),
        ):
            stack.enter_context(ctx)
        return await build_spec_from_session("sess_01TEST")


async def test_env_var_cred_with_unrestricted_networking_rejected() -> None:
    env = EnvironmentConfig(networking=UnrestrictedNetworking())
    with pytest.raises(ValueError, match="Limited"):
        await _build_with(env_config=env, creds=(_cred(["api.github.com"]),))


async def test_env_var_cred_with_no_env_config_rejected() -> None:
    with pytest.raises(ValueError, match="Limited"):
        await _build_with(env_config=None, creds=(_cred(["api.github.com"]),))


async def test_env_var_cred_uncovered_host_rejected_names_host() -> None:
    with pytest.raises(ValueError, match=r"evil\.example\.com"):
        await _build_with(
            env_config=limited_env("api.github.com"), creds=(_cred(["evil.example.com"]),)
        )


async def test_env_var_cred_covered_host_provisions() -> None:
    cred = _cred(["api.github.com"])
    plan = await _build_with(env_config=limited_env("api.github.com"), creds=(cred,))
    # A covered cred provisions and its placeholder lands in the env. Assert
    # against the injected cred's OWN placeholder, not the module-level _CRED's
    # — the two happen to match only because _cred() hardcodes the constant.
    assert plan.spec.environment["GITHUB_TOKEN"] == cred.placeholder  # type: ignore[attr-defined]


async def test_env_var_cred_path_prefix_provisions_host_only() -> None:
    # A path-prefixed cred is compared host-only against the env's host set.
    cred = _cred(["api.github.com/repos/eumemic"])
    plan = await _build_with(env_config=limited_env("api.github.com"), creds=(cred,))
    assert plan.spec.environment["GITHUB_TOKEN"] == cred.placeholder  # type: ignore[attr-defined]


async def test_vaults_but_no_env_var_creds_with_unrestricted_provisions() -> None:
    # Zero env-var creds ⇒ gate skipped entirely; Unrestricted is fine.
    plan = await _build_with(
        env_config=EnvironmentConfig(networking=UnrestrictedNetworking()), creds=()
    )
    assert set(plan.spec.environment) == RESERVED_SANDBOX_ENV_KEYS  # type: ignore[attr-defined]


async def test_env_multi_host_covers_each_cred_host_provisions() -> None:
    # NOTE: env ``allowed_hosts`` are bare hostnames only (HOSTNAME_RE rejects a
    # path prefix), so the plan's "env-side path prefix" case is unconstructable.
    # The host-only env parse is still exercised: a multi-host Limited env
    # covers a bare cred host that is one of several allowed.
    cred = _cred(["api.github.com"])
    plan = await _build_with(
        env_config=limited_env("pypi.org", "api.github.com"),
        creds=(cred,),
    )
    assert plan.spec.environment["GITHUB_TOKEN"] == cred.placeholder  # type: ignore[attr-defined]
