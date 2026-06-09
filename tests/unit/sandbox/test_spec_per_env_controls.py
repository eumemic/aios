"""Per-environment sandbox controls: image override, disk cap, and the
bash-timeout ceiling resolver (issues #724, #725).

#724 adds an optional :attr:`EnvironmentConfig.image`; #725 adds a
per-environment writable-layer disk cap and a per-environment bash
timeout ceiling. All three default to current behavior when unset:

- ``image`` unset → the worker-global ``settings.docker_image``.
- ``disk_bytes`` unset → the worker-global ``settings.sandbox_disk_bytes``
  (itself ``None`` = unbounded by default).
- ``bash_timeout_seconds`` unset → the worker-global
  ``settings.bash_default_timeout_seconds``.

The image/disk resolution lives in ``build_spec_from_session``; the
plumbing onto the spec lives in ``_assemble_plan``; the bash ceiling
lives in ``resolve_bash_timeout_ceiling``. These tests pin each.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.models.environments import EnvironmentConfig
from aios.sandbox.spec import (
    _assemble_plan,
    resolve_bash_timeout_ceiling,
)


def _call_assemble(
    *,
    image: str = "aios-sandbox:test",
    disk_bytes: int | None = None,
    env_config: EnvironmentConfig | None = None,
) -> object:
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
            disk_bytes=disk_bytes,
        )


class TestAssemblePlanImageAndDisk:
    """``_assemble_plan`` copies the resolved image + disk cap onto the
    spec verbatim — it does not itself read settings for these."""

    def test_image_lands_on_spec(self) -> None:
        plan = _call_assemble(image="ghcr.io/eumemic/aios-sandbox:pinned")
        assert plan.spec.image == "ghcr.io/eumemic/aios-sandbox:pinned"

    def test_disk_bytes_lands_on_spec_when_set(self) -> None:
        plan = _call_assemble(disk_bytes=2 * 1024 * 1024 * 1024)
        assert plan.spec.disk_bytes == 2 * 1024 * 1024 * 1024

    def test_disk_bytes_none_by_default(self) -> None:
        plan = _call_assemble()
        assert plan.spec.disk_bytes is None


# ── build_spec_from_session resolution (#724 image, #725 disk) ───────────────


def _patch_build_spec_deps(
    *,
    env_config: EnvironmentConfig | None,
    docker_image: str,
    sandbox_disk_bytes: int | None,
):
    """Context manager bundle that stubs every external dependency of
    ``build_spec_from_session`` so it runs to the ``_assemble_plan`` call
    with a synthetic environment config and synthetic settings."""
    settings = MagicMock()
    settings.docker_image = docker_image
    settings.sandbox_disk_bytes = sandbox_disk_bytes
    settings.instance_id = "inst_TEST"
    settings.sandbox_cpu_quota = None
    settings.sandbox_memory_bytes = None
    settings.sandbox_pids_limit = None
    settings.tool_broker_socket_path = None

    tool_broker = MagicMock()
    tool_broker.port = 54321
    tool_broker.register_session = MagicMock()
    tool_broker.unregister_session = MagicMock()

    return (
        patch("aios.sandbox.spec.get_settings", return_value=settings),
        patch(
            "aios.sandbox.spec.sessions_service.load_session_account_id",
            AsyncMock(return_value="acct_x"),
        ),
        patch(
            "aios.sandbox.spec._load_session_provisioning",
            # (workspace_path, env, spec_version) since #713.
            AsyncMock(return_value=("/tmp/w", {}, 0)),
        ),
        # ``build_spec_from_session`` imports these function-locally from
        # ``aios.sandbox.volumes`` (deferred import to avoid a cycle), so
        # patch them at the source module, not on ``aios.sandbox.spec``.
        patch("aios.sandbox.volumes.validate_workspace_path", MagicMock()),
        patch(
            "aios.sandbox.volumes.ensure_workspace_path",
            MagicMock(return_value=Path("/tmp/w")),
        ),
        patch(
            "aios.sandbox.spec._load_environment_config",
            AsyncMock(return_value=env_config),
        ),
        patch(
            "aios.sandbox.spec._materialize_memory_mounts",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.sandbox.spec._materialize_github_clones",
            AsyncMock(return_value=([], None)),
        ),
        patch("aios.sandbox.spec.runtime.require_pool", MagicMock()),
        patch(
            "aios.sandbox.spec.runtime.require_tool_broker",
            MagicMock(return_value=tool_broker),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_uploads_dir",
            return_value=Path("/tmp/u"),
        ),
    )


async def _build_with(
    *,
    env_config: EnvironmentConfig | None,
    docker_image: str = "ghcr.io/eumemic/aios-sandbox:latest",
    sandbox_disk_bytes: int | None = None,
):
    from contextlib import ExitStack

    from aios.sandbox.spec import build_spec_from_session

    with ExitStack() as stack:
        for cm in _patch_build_spec_deps(
            env_config=env_config,
            docker_image=docker_image,
            sandbox_disk_bytes=sandbox_disk_bytes,
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


class TestBuildSpecDiskResolution:
    """#725: per-env ``disk_bytes`` wins; else the global default; else None."""

    @pytest.mark.asyncio
    async def test_env_disk_overrides_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(disk_bytes=8 * 1024 * 1024 * 1024),
            sandbox_disk_bytes=2 * 1024 * 1024 * 1024,
        )
        assert plan.spec.disk_bytes == 8 * 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_unset_env_disk_falls_back_to_global(self) -> None:
        plan = await _build_with(
            env_config=EnvironmentConfig(packages={"pip": ["x"]}),
            sandbox_disk_bytes=2 * 1024 * 1024 * 1024,
        )
        assert plan.spec.disk_bytes == 2 * 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_both_unset_is_none(self) -> None:
        plan = await _build_with(
            env_config=None,
            sandbox_disk_bytes=None,
        )
        assert plan.spec.disk_bytes is None


# ── resolve_bash_timeout_ceiling (#725) ──────────────────────────────────────


def _patch_ceiling_deps(
    *,
    env_config: EnvironmentConfig | None,
    default: int,
    load_raises: bool = False,
):
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
