"""Sandbox provisioner pins language-package installs into /workspace (#227).

Tools the model installs at runtime (``pip install``, ``npm install -g``)
land in the container's writable layer by default, which gets reclaimed
on ``sandbox.idle_release``.  By pre-creating ``/workspace/.venv`` and
``/workspace/.npm`` at every provision and setting the matching env
vars (``VIRTUAL_ENV``, ``NPM_CONFIG_PREFIX``, ``NODE_PATH``, ``PATH``),
those installs land inside the bind mount and survive idle release.

These tests exercise the provisioner's docker-run argv composition and
the post-start setup that creates the runtime dirs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.environments import EnvironmentConfig
from aios.sandbox.container import CommandResult


def _env_dict_from_argv(argv: list[str]) -> dict[str, str]:
    """Extract --env KEY=VALUE pairs from a docker run argv into a dict.

    Later occurrences win, matching docker's last-flag-wins semantics.
    """
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--env" and i + 1 < len(argv):
            key, _, value = argv[i + 1].partition("=")
            out[key] = value
            i += 2
        else:
            i += 1
    return out


async def _capture_provision_argv(
    *,
    env_config: EnvironmentConfig | None = None,
    session_env: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """Run provision_for_session under heavy mock; return docker run argv
    and the run_command call list (for inspecting the post-start setup).
    """
    from aios.sandbox.container import ContainerHandle

    captured_argv: list[list[str]] = []
    captured_run_commands: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
        captured_argv.append(argv)
        return 0, b"container_abc123\n", b""

    async def fake_run_command(self: ContainerHandle, command: str, **kwargs: Any) -> CommandResult:
        captured_run_commands.append(((command,), kwargs))
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)

    with (
        patch.object(ContainerHandle, "run_command", fake_run_command),
        patch(
            "aios.sandbox.provisioner._load_environment_config",
            AsyncMock(return_value=env_config),
        ),
        patch("aios.sandbox.provisioner._run_docker", fake_run_docker),
        patch(
            "aios.sandbox.provisioner._load_session_provisioning",
            AsyncMock(return_value=("/tmp/ws", session_env or {})),
        ),
        patch(
            "aios.sandbox.provisioner._materialize_memory_mounts",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.sandbox.provisioner._materialize_github_clones",
            AsyncMock(return_value=([], None)),
        ),
        patch("aios.sandbox.volumes.ensure_workspace_path", return_value=Path("/tmp/ws")),
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/attachments"),
        ),
    ):
        from aios.sandbox.provisioner import provision_for_session

        await provision_for_session("sess_01TEST")

    return captured_argv[0], captured_run_commands


class TestWorkspaceRuntimeEnv:
    @pytest.mark.asyncio
    async def test_default_env_includes_venv_and_npm_pins(self) -> None:
        """Default provision (no env_config) sets VIRTUAL_ENV / NPM_CONFIG_PREFIX
        / NODE_PATH / PATH so language-package installs land inside the bind
        mount.
        """
        argv, _ = await _capture_provision_argv()
        env = _env_dict_from_argv(argv)

        assert env["VIRTUAL_ENV"] == "/workspace/.venv"
        assert env["NPM_CONFIG_PREFIX"] == "/workspace/.npm"
        assert env["NODE_PATH"] == "/workspace/.npm/lib/node_modules"

    @pytest.mark.asyncio
    async def test_path_prepends_venv_and_npm_bin(self) -> None:
        """PATH must put /workspace/.venv/bin and /workspace/.npm/bin BEFORE
        system bin dirs so the venv's python and npm-installed tools resolve
        ahead of the image's defaults.
        """
        argv, _ = await _capture_provision_argv()
        env = _env_dict_from_argv(argv)

        path_segments = env["PATH"].split(":")
        assert path_segments[0] == "/workspace/.venv/bin"
        assert path_segments[1] == "/workspace/.npm/bin"
        assert "/usr/local/bin" in path_segments
        assert "/usr/bin" in path_segments

    @pytest.mark.asyncio
    async def test_user_env_overrides_runtime_pins(self) -> None:
        """env_config.env wins over the runtime-pin defaults so operators
        can opt out (e.g. a session that needs a different VIRTUAL_ENV).
        """
        argv, _ = await _capture_provision_argv(
            env_config=EnvironmentConfig(env={"VIRTUAL_ENV": "/custom/venv"})
        )
        env = _env_dict_from_argv(argv)

        assert env["VIRTUAL_ENV"] == "/custom/venv"
        assert env["NPM_CONFIG_PREFIX"] == "/workspace/.npm"


class TestWorkspaceRuntimeDirsSetup:
    @pytest.mark.asyncio
    async def test_post_start_creates_venv_and_npm_dirs(self) -> None:
        """After docker run, the provisioner runs a setup command that
        creates ``/workspace/.venv`` (via python3 -m venv) and the
        ``/workspace/.npm/{lib,bin}`` dirs.
        """
        _, run_commands = await _capture_provision_argv()

        cmds = [pargs[0] for pargs, _pkwargs in run_commands]
        joined = "\n".join(cmds)

        assert "python3 -m venv /workspace/.venv" in joined
        assert "/workspace/.npm/lib" in joined
        assert "/workspace/.npm/bin" in joined

    @pytest.mark.asyncio
    async def test_venv_creation_is_idempotent(self) -> None:
        """The setup command must not re-run python3 -m venv when the venv
        already exists (would error out / waste cycles on every cold
        provision).  A ``[ -e .venv/bin/python ] || python3 -m venv``
        guard handles this in one shell call.
        """
        _, run_commands = await _capture_provision_argv()

        cmds = [pargs[0] for pargs, _pkwargs in run_commands]
        joined = "\n".join(cmds)

        assert "[ -e /workspace/.venv/bin/python ]" in joined
        assert "||" in joined
