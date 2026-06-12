"""Workflow-run env-var credential materialization (#882)."""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.models.environments import EnvironmentConfig, LimitedNetworking, UnrestrictedNetworking
from aios.models.vaults import RESERVED_SANDBOX_ENV_KEYS
from aios.sandbox.spec import build_spec_from_run
from aios.services.vaults import ResolvedEnvVarCredential
from tests.helpers.sandbox import limited_env

_SENTINEL_SECRET = "RUN_SENTINEL_PLAINTEXT_DO_NOT_LEAK"
_RUN_ID = "run_01TEST"
_ACC = "acct_run"
_CRED = ResolvedEnvVarCredential(
    credential_id="vcred_run_01TEST",
    secret_name="PW_DEV_GATE_PASSWORD",
    secret_value=_SENTINEL_SECRET,
    allowed_hosts=("api.example.com",),
    updated_at=datetime(2026, 6, 12, tzinfo=UTC),
    placeholder="AIOS_SECRET_PLACEHOLDER_" + "cd" * 16,
)


class _Acquire:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _Pool:
    def acquire(self) -> _Acquire:
        return _Acquire()


def _run() -> SimpleNamespace:
    return SimpleNamespace(id=_RUN_ID, account_id=_ACC, environment_id="env_run")


def _patch_run_spec_deps(
    *,
    env_config: EnvironmentConfig | None = None,
    env_var_credentials: AsyncMock | None = None,
    broker: MagicMock | None = None,
    secret_proxy_instance: MagicMock | None = None,
) -> tuple[AbstractContextManager[Any], ...]:
    settings = MagicMock()
    settings.docker_image = "ghcr.io/eumemic/aios-sandbox:latest"
    settings.instance_id = "inst_TEST"
    settings.sandbox_cpu_quota = None
    settings.sandbox_memory_bytes = None
    settings.sandbox_pids_limit = None
    settings.sandbox_seccomp_profile = "/app/docker/seccomp-sandbox.json"
    settings.sandbox_runtime = None
    settings.tool_broker_socket_path = None
    if broker is None:
        broker = MagicMock()
        broker.port = 54321
        broker.register_session = MagicMock()
        broker.unregister_session = MagicMock()
    if secret_proxy_instance is None:
        secret_proxy_instance = MagicMock()
        secret_proxy_instance.start = AsyncMock()
        secret_proxy_instance.stop = AsyncMock()
    return (
        patch("aios.sandbox.spec.get_settings", return_value=settings),
        patch("aios.sandbox.spec.runtime.require_pool", return_value=_Pool()),
        patch("aios.sandbox.spec.runtime.require_tool_broker", return_value=broker),
        patch(
            "aios.sandbox.spec.queries.get_environment_config_for_id",
            AsyncMock(return_value=env_config),
        ),
        patch("aios.db.queries.workflows.get_run_for_step", AsyncMock(return_value=_run())),
        patch(
            "aios.sandbox.volumes.ensure_run_workspace_dir",
            return_value=__import__("pathlib").Path("/tmp/run-w"),
        ),
        # ``_assemble_plan`` binds the per-session attachments/uploads dirs by
        # calling these (function-locally imported from ``aios.sandbox.volumes``),
        # and they ``mkdir`` under the production workspace root. Redirect both to
        # tmp so the run-spec build never touches ``/var/lib/aios`` — mirrors the
        # session-twin's ``patch_build_spec_deps`` helper.
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=__import__("pathlib").Path("/tmp/run-a"),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_uploads_dir",
            return_value=__import__("pathlib").Path("/tmp/run-u"),
        ),
        patch(
            "aios.sandbox.spec._materialize_run_env_var_credentials",
            env_var_credentials or AsyncMock(return_value=()),
        ),
        patch(
            "aios.sandbox.spec.SecretEgressProxy",
            MagicMock(return_value=secret_proxy_instance),
        ),
    )


async def test_run_placeholder_lands_in_env_secret_never_does() -> None:
    with contextlib.ExitStack() as stack:
        for ctx in _patch_run_spec_deps(
            env_config=limited_env("api.example.com"),
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
        ):
            stack.enter_context(ctx)
        plan = await build_spec_from_run(_RUN_ID)

    assert plan.spec.environment["PW_DEV_GATE_PASSWORD"] == _CRED.placeholder
    assert plan.spec.runtime is None
    assert _SENTINEL_SECRET not in str(plan.spec)
    assert _SENTINEL_SECRET not in " ".join(plan.spec.environment.values())
    assert plan.env_var_credentials == (_CRED,)
    runtime_reserved = RESERVED_SANDBOX_ENV_KEYS - {"AIOS_RUN_ID", "AIOS_IDEMPOTENCY_KEY"}
    assert set(plan.spec.environment) == {*runtime_reserved, "PW_DEV_GATE_PASSWORD"}


async def test_run_env_var_cred_with_unrestricted_networking_rejected_before_broker() -> None:
    broker = MagicMock()
    broker.port = 54321
    broker.register_session = MagicMock()
    with contextlib.ExitStack() as stack:
        for ctx in _patch_run_spec_deps(
            env_config=EnvironmentConfig(networking=UnrestrictedNetworking()),
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
            broker=broker,
        ):
            stack.enter_context(ctx)
        with pytest.raises(ValueError, match="Limited"):
            await build_spec_from_run(_RUN_ID)
    broker.register_session.assert_not_called()


async def test_run_secret_proxy_constructed_started_and_cleaned_on_assembly_failure() -> None:
    secret_proxy = MagicMock()
    secret_proxy.start = AsyncMock()
    secret_proxy.stop = AsyncMock()
    broker = MagicMock()
    broker.port = 54321
    broker.register_session = MagicMock()
    broker.unregister_session = MagicMock()
    env_config = MagicMock()
    env_config.image = "aios-sbx-victim"
    env_config.networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])
    env_config.env = {}
    with contextlib.ExitStack() as stack:
        for ctx in _patch_run_spec_deps(
            env_config=env_config,
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
            broker=broker,
        ):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("aios.sandbox.spec.SecretEgressProxy", MagicMock(return_value=secret_proxy))
        )
        with pytest.raises(ValueError, match="reserved"):
            await build_spec_from_run(_RUN_ID)

    secret_proxy.start.assert_awaited_once()
    secret_proxy.stop.assert_awaited_once()
    broker.unregister_session.assert_called_once_with(_RUN_ID)


async def test_run_secret_proxy_start_failure_does_not_double_stop() -> None:
    secret_proxy = MagicMock()
    secret_proxy.stop = AsyncMock()

    async def _start_then_self_clean() -> None:
        await secret_proxy.stop()
        raise RuntimeError("bind failed")

    secret_proxy.start = AsyncMock(side_effect=_start_then_self_clean)
    broker = MagicMock()
    broker.port = 54321
    broker.register_session = MagicMock()
    with contextlib.ExitStack() as stack:
        for ctx in _patch_run_spec_deps(
            env_config=limited_env("api.example.com"),
            env_var_credentials=AsyncMock(return_value=(_CRED,)),
            broker=broker,
        ):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("aios.sandbox.spec.SecretEgressProxy", MagicMock(return_value=secret_proxy))
        )
        with pytest.raises(RuntimeError, match="bind failed"):
            await build_spec_from_run(_RUN_ID)

    secret_proxy.stop.assert_awaited_once()
    broker.register_session.assert_not_called()
