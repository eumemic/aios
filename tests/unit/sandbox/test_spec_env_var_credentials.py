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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.crypto.vault import CryptoBox
from aios.errors import CryptoDecryptError
from aios.ids import VAULT_CREDENTIAL
from aios.models.environments import EnvironmentConfig
from aios.models.vaults import RESERVED_SANDBOX_ENV_KEYS
from aios.sandbox.spec import build_spec_from_session
from aios.services.vaults import ResolvedEnvVarCredential, mint_secret_placeholder
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


# ── SecretEgressProxy lifecycle (#877) ───────────────────────────────────────


async def test_secret_proxy_constructed_and_started_when_creds_present() -> None:
    """A provision with env-var creds constructs the per-session
    ``SecretEgressProxy`` with those creds, awaits its ``start()``, and hands
    the live instance back on ``plan.secret_proxy`` for the registry to own."""
    proxy_instance = MagicMock()
    proxy_instance.start = AsyncMock()
    proxy_cls = MagicMock(return_value=proxy_instance)
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        stack.enter_context(patch("aios.sandbox.spec.SecretEgressProxy", proxy_cls))
        plan = await build_spec_from_session("sess_01TEST")

    # Constructed with the resolved creds and started exactly once.
    proxy_cls.assert_called_once_with((_CRED,))
    proxy_instance.start.assert_awaited_once()
    assert plan.secret_proxy is proxy_instance


async def test_secret_proxy_not_constructed_when_no_creds() -> None:
    """No env-var creds ⇒ the proxy is never constructed and the plan carries
    ``secret_proxy is None`` (the inert empty case)."""
    proxy_cls = MagicMock()
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=()),
        ):
            stack.enter_context(ctx)
        stack.enter_context(patch("aios.sandbox.spec.SecretEgressProxy", proxy_cls))
        plan = await build_spec_from_session("sess_01TEST")

    proxy_cls.assert_not_called()
    assert plan.secret_proxy is None


async def test_provision_time_fold_lands_cred_on_mount_snapshot() -> None:
    """The provision-time half of constraint A: the resolved cred's
    ``(VAULT_CREDENTIAL, credential_id, updated_at)`` tuple lands on
    ``plan.spec.mount_snapshot`` so the handle the backend stamps it onto
    matches the step-time echo set."""
    proxy_instance = MagicMock()
    proxy_instance.start = AsyncMock()
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("aios.sandbox.spec.SecretEgressProxy", MagicMock(return_value=proxy_instance))
        )
        plan = await build_spec_from_session("sess_01TEST")

    assert (
        VAULT_CREDENTIAL,
        _CRED.credential_id,
        _CRED.updated_at.isoformat(),
    ) in plan.spec.mount_snapshot


async def test_failure_after_proxy_starts_stops_both_proxies() -> None:
    """A raise in the post-proxy-start span (here the reserved-image gate)
    must tear down BOTH the git proxy and the secret-egress proxy.

    Both proxies and the broker secret are allocated BEFORE the spec is
    assembled; a single cleanup envelope must cover all of them so a
    failed provision can't leak a live proxy for the worker's lifetime.
    """
    git_proxy = MagicMock()
    git_proxy.stop = AsyncMock()
    secret_proxy = MagicMock()
    secret_proxy.start = AsyncMock()
    secret_proxy.stop = AsyncMock()
    broker = MagicMock()
    broker.port = 54321
    broker.register_session = MagicMock()
    broker.unregister_session = MagicMock()
    # A reserved-prefix image trips the cross-tenant snapshot gate's
    # ValueError after both proxies have started and the broker secret is
    # registered. ``MagicMock`` env_config bypasses the pydantic ``image``
    # validator so the gate (not construction) is what raises.
    env_config = MagicMock()
    env_config.image = "aios-sbx-victim"
    with contextlib.ExitStack() as stack:
        for ctx in patch_build_spec_deps(
            env_config=env_config,
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
            github_clones=AsyncMock(return_value=([], git_proxy)),
            tool_broker=broker,
        ):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("aios.sandbox.spec.SecretEgressProxy", MagicMock(return_value=secret_proxy))
        )
        with pytest.raises(ValueError, match="reserved"):
            await build_spec_from_session("sess_01TEST")

    # Everything allocated before the failure is torn down: both proxies
    # stopped and the broker secret unregistered.
    git_proxy.stop.assert_awaited_once()
    secret_proxy.stop.assert_awaited_once()
    broker.unregister_session.assert_called_once_with("sess_01TEST")


def test_placeholder_is_stable_across_recycle() -> None:
    """The placeholder is a pure function of ``(subkey, session_id,
    credential_id)`` — two mints with the same ids are equal.

    This locks the piece-4 reconstruction invariant: a recycled sandbox
    re-derives the SAME placeholder, so anything the agent persisted into
    /workspace keeps resolving through the fresh proxy. Already deterministic
    — no production change; this test documents/guards it.
    """
    box = CryptoBox(bytes(range(32)))
    subkey = box.derive_account_subkey("acct_x")
    first = mint_secret_placeholder(subkey, "sess_01TEST", "vcr_01")
    second = mint_secret_placeholder(subkey, "sess_01TEST", "vcr_01")
    assert first == second
