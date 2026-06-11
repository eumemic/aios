"""Per-environment sandbox controls: image override, snapshot budget, and the
bash-timeout ceiling resolver (issues #724, #725; durable session sandboxes).

#724 adds an optional :attr:`EnvironmentConfig.image`; durable session
sandboxes replace the former writable-layer ``disk_bytes`` cap with a
per-session ``snapshot_budget_bytes`` (over budget at teardown → flatten,
never refuse). All three default to current behavior when unset:

- ``image`` unset → the worker-global ``settings.docker_image``.
- ``snapshot_budget_bytes`` unset → the worker-global
  ``settings.sandbox_snapshot_budget_bytes`` (4 GiB by default).
- ``bash_timeout_seconds`` unset → the worker-global
  ``settings.bash_default_timeout_seconds``.

The image/budget resolution lives in ``build_spec_from_session``; the
plumbing onto the spec lives in ``_assemble_plan``; the bash ceiling
lives in ``resolve_bash_timeout_ceiling``. These tests pin each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.models.environments import EnvironmentConfig
from aios.sandbox.spec import (
    ProvisioningPlan,
    _assemble_plan,
    resolve_bash_timeout_ceiling,
)
from tests.helpers.sandbox import patch_build_spec_deps


def _call_assemble(
    *,
    image: str = "aios-sandbox:test",
    snapshot_budget_bytes: int | None = None,
    snapshot_ref: str | None = None,
    env_config: EnvironmentConfig | None = None,
) -> ProvisioningPlan:
    # ``_assemble_plan`` imports its volume helpers function-locally, so
    # patch them at the ``aios.sandbox.volumes`` source module (same shape
    # as ``test_spec_uds_url.py``).
    with (
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_uploads_dir",
            return_value=Path("/tmp/u"),
        ),
    ):
        return _assemble_plan(
            session_id="sess_01TEST",
            instance_id="inst_TEST",
            image=image,
            workspace_path=Path("/tmp/w"),
            env_config=env_config,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url="http://aios-worker:54321",
            tool_broker_secret="secret123",
            tool_socket_host_path=None,
            snapshot_ref=snapshot_ref,
            snapshot_budget_bytes=snapshot_budget_bytes,
        )


class TestAssemblePlanImageAndBudget:
    """``_assemble_plan`` copies the resolved image + snapshot budget + pointer
    onto the spec verbatim — it does not itself read settings for these."""

    def test_image_lands_on_spec(self) -> None:
        plan = _call_assemble(image="ghcr.io/eumemic/aios-sandbox:pinned")
        assert plan.spec.image == "ghcr.io/eumemic/aios-sandbox:pinned"

    def test_snapshot_budget_lands_on_spec_when_set(self) -> None:
        plan = _call_assemble(snapshot_budget_bytes=2 * 1024 * 1024 * 1024)
        assert plan.spec.snapshot_budget_bytes == 2 * 1024 * 1024 * 1024

    def test_snapshot_budget_none_by_default(self) -> None:
        plan = _call_assemble()
        assert plan.spec.snapshot_budget_bytes is None

    def test_snapshot_ref_lands_on_spec_image(self) -> None:
        """The DB snapshot pointer arrives on the spec as ``snapshot_image``;
        the registry resolves it through the store before ``backend.create``."""
        plan = _call_assemble(snapshot_ref="aios-sbx-inst_test-sess_01test:latest")
        assert plan.spec.snapshot_image == "aios-sbx-inst_test-sess_01test:latest"

    def test_snapshot_ref_none_by_default(self) -> None:
        plan = _call_assemble()
        assert plan.spec.snapshot_image is None

    def test_env_keys_and_base_image_labels_stamped(self) -> None:
        """Durable session sandboxes stamp ``aios.env_keys`` (the names of every
        run-injected env var) and ``aios.base_image`` (the chain root) so the
        commit-time scrub and accounting work off labels alone."""
        plan = _call_assemble(image="ghcr.io/eumemic/aios-sandbox:pinned")
        labels = plan.spec.labels
        assert labels["aios.base_image"] == "ghcr.io/eumemic/aios-sandbox:pinned"
        env_keys = set(labels["aios.env_keys"].split(","))
        # The names (never values) of the run-injected env — includes the broker
        # secret's KEY but the scrub only ever empties it, never reads a value.
        assert {"PATH", "TOOL_BROKER_SECRET", "AIOS_SESSION_ID"} <= env_keys

    def test_seccomp_profile_from_settings_lands_on_spec(self) -> None:
        """#807: ``_assemble_plan`` copies ``settings.sandbox_seccomp_profile``
        verbatim onto the spec. It reads ``get_settings()`` internally, so we
        patch it with a mock carrying the seccomp field plus the existing
        sandbox-cap attributes the constructor reads."""
        settings = MagicMock()
        settings.sandbox_seccomp_profile = "/x/seccomp.json"
        settings.sandbox_cpu_quota = None
        settings.sandbox_memory_bytes = None
        settings.sandbox_pids_limit = None
        with patch("aios.sandbox.spec.get_settings", return_value=settings):
            plan = _call_assemble()
        assert plan.spec.seccomp_profile == "/x/seccomp.json"


# ── build_spec_from_session resolution (#724 image, snapshot budget) ─────────


async def _build_with(
    *,
    env_config: EnvironmentConfig | None,
    docker_image: str = "ghcr.io/eumemic/aios-sandbox:latest",
    sandbox_snapshot_budget_bytes: int | None = None,
) -> ProvisioningPlan:
    from contextlib import ExitStack

    from aios.sandbox.spec import build_spec_from_session

    with ExitStack() as stack:
        for cm in patch_build_spec_deps(
            env_config=env_config,
            docker_image=docker_image,
            sandbox_snapshot_budget_bytes=sandbox_snapshot_budget_bytes,
        ):
            stack.enter_context(cm)
        return await build_spec_from_session("sess_01TEST")


class TestBuildSpecImageResolution:
    """#724: a session bound to an environment with ``image=X`` provisions
    from X; unset falls back to the global ``settings.docker_image``."""

    @pytest.mark.asyncio
    async def test_env_image_overrides_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(image="ghcr.io/eumemic/dev-env:pinned"),
            docker_image="ghcr.io/eumemic/aios-sandbox:latest",
        )
        assert plan.spec.image == "ghcr.io/eumemic/dev-env:pinned"

    @pytest.mark.asyncio
    async def test_unset_image_falls_back_to_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(packages={"pip": ["pandas"]}),
            docker_image="ghcr.io/eumemic/aios-sandbox:latest",
        )
        assert plan.spec.image == "ghcr.io/eumemic/aios-sandbox:latest"

    @pytest.mark.asyncio
    async def test_no_env_config_falls_back_to_global(self) -> None:
        """A session with no environment attached uses the global image."""
        plan = await _build_with(
            env_config=None,
            docker_image="ghcr.io/eumemic/aios-sandbox:latest",
        )
        assert plan.spec.image == "ghcr.io/eumemic/aios-sandbox:latest"

    @pytest.mark.asyncio
    async def test_reserved_prefix_image_is_rejected(self) -> None:
        """Cross-tenant gate (§5.8): the spec-build choke point rejects a
        tenant ``image`` in the reserved per-session-snapshot namespace, so a
        tenant can't mount another session's snapshot (and its baked secrets)."""
        with pytest.raises(ValueError, match="reserved"):
            await _build_with(
                env_config=EnvironmentConfig.model_construct(
                    image="aios-sbx-default-sess_victim:latest"
                ),
            )


class TestBuildSpecSnapshotBudgetResolution:
    """Per-env ``snapshot_budget_bytes`` wins; else the global default; else None."""

    @pytest.mark.asyncio
    async def test_env_budget_overrides_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(snapshot_budget_bytes=8 * 1024 * 1024 * 1024),
            sandbox_snapshot_budget_bytes=2 * 1024 * 1024 * 1024,
        )
        assert plan.spec.snapshot_budget_bytes == 8 * 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_unset_env_budget_falls_back_to_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(packages={"pip": ["x"]}),
            sandbox_snapshot_budget_bytes=2 * 1024 * 1024 * 1024,
        )
        assert plan.spec.snapshot_budget_bytes == 2 * 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_both_unset_is_none(self) -> None:
        plan = await _build_with(
            env_config=None,
            sandbox_snapshot_budget_bytes=None,
        )
        assert plan.spec.snapshot_budget_bytes is None


# ── resolve_bash_timeout_ceiling (#725) ──────────────────────────────────────


def _patch_ceiling_deps(
    *,
    env_config: EnvironmentConfig | None,
    default: int,
    load_raises: bool = False,
) -> tuple[Any, ...]:
    settings = MagicMock()
    settings.bash_default_timeout_seconds = default
    load_env = AsyncMock(return_value=env_config)
    if load_raises:
        load_env = AsyncMock(side_effect=RuntimeError("db down"))
    return (
        patch("aios.sandbox.spec.get_settings", return_value=settings),
        patch("aios.sandbox.spec.runtime.require_pool", MagicMock()),
        patch(
            "aios.sandbox.spec.sessions_service.load_session_account_id",
            AsyncMock(return_value="acct_x"),
        ),
        patch("aios.sandbox.spec._load_environment_config", load_env),
    )


async def _resolve_with(
    *,
    env_config: EnvironmentConfig | None,
    default: int = 120,
    load_raises: bool = False,
) -> int:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for cm in _patch_ceiling_deps(
            env_config=env_config, default=default, load_raises=load_raises
        ):
            stack.enter_context(cm)
        return await resolve_bash_timeout_ceiling("sess_01TEST")


class TestResolveBashTimeoutCeiling:
    @pytest.mark.asyncio
    async def test_env_override_wins(self) -> None:
        ceiling = await _resolve_with(
            env_config=EnvironmentConfig(bash_timeout_seconds=600),
            default=120,
        )
        assert ceiling == 600

    @pytest.mark.asyncio
    async def test_unset_env_uses_global_default(self) -> None:
        ceiling = await _resolve_with(
            env_config=EnvironmentConfig(packages={"pip": ["x"]}),
            default=120,
        )
        assert ceiling == 120

    @pytest.mark.asyncio
    async def test_no_env_config_uses_global_default(self) -> None:
        ceiling = await _resolve_with(env_config=None, default=120)
        assert ceiling == 120

    @pytest.mark.asyncio
    async def test_db_failure_falls_back_to_global_default(self) -> None:
        """Total by contract: a DB hiccup while resolving the per-env ceiling
        must fall back to the global default, never block the bash call."""
        ceiling = await _resolve_with(env_config=None, default=120, load_raises=True)
        assert ceiling == 120
