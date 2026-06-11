"""Sandbox provisioning pins language-package installs into /workspace (#227).

Tools the model installs at runtime (``pip install``, ``npm install -g``)
land in the sandbox's writable layer by default, which gets reclaimed
on ``sandbox.idle_release``.  By pre-creating ``/workspace/.venv`` and
``/workspace/.npm`` at every provision and setting the matching env
vars (``VIRTUAL_ENV``, ``NPM_CONFIG_PREFIX``, ``NODE_PATH``, ``PATH``),
those installs land inside the bind mount and survive idle release.

After the SandboxBackend refactor these tests exercise:
- The ``DockerBackend.create`` argv translation (env vars on the
  spec end up as ``--env`` flags on the docker CLI).
- The ``setup.ensure_workspace_runtime_dirs`` helper (the post-create
  command that creates the venv + npm dirs idempotently).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aios.sandbox.backends.base import (
    CommandResult,
    Mount,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.setup import (
    WORKSPACE_RUNTIME_ENV,
    ensure_workspace_runtime_dirs,
)
from tests.helpers.sandbox import FakeBackend, make_handle


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


def _make_spec(environment: dict[str, str]) -> SandboxSpec:
    """Construct a SandboxSpec with the given environment, defaults elsewhere."""
    return SandboxSpec(
        session_id="sess_01TEST",
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment=environment,
        labels={"aios.managed": "true"},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )


async def _capture_docker_argv(spec: SandboxSpec, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Call DockerBackend.create(spec) with run_docker_cli patched; return argv."""
    captured: list[list[str]] = []

    async def fake_run_docker(
        argv: list[str], *, timeout_s: float = 30.0, **kwargs: Any
    ) -> tuple[int, bytes, bytes]:
        captured.append(argv)
        return 0, b"container_abc123\n", b""

    monkeypatch.setattr("aios.sandbox.backends.docker.run_docker_cli", fake_run_docker)
    await DockerBackend().create(spec)
    return captured[0]


class TestWorkspaceRuntimeEnvOnSpec:
    """The merged environment on a SandboxSpec carries the runtime pins."""

    async def test_default_env_includes_venv_and_npm_pins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = _make_spec(environment=dict(WORKSPACE_RUNTIME_ENV))
        argv = await _capture_docker_argv(spec, monkeypatch)
        env = _env_dict_from_argv(argv)

        assert env["VIRTUAL_ENV"] == "/workspace/.venv"
        assert env["NPM_CONFIG_PREFIX"] == "/workspace/.npm"
        assert env["NODE_PATH"] == "/workspace/.npm/lib/node_modules"

    async def test_path_prepends_venv_and_npm_bin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = _make_spec(environment=dict(WORKSPACE_RUNTIME_ENV))
        argv = await _capture_docker_argv(spec, monkeypatch)
        env = _env_dict_from_argv(argv)

        path_segments = env["PATH"].split(":")
        assert path_segments[0] == "/workspace/.venv/bin"
        assert path_segments[1] == "/workspace/.npm/bin"
        assert "/usr/local/bin" in path_segments
        assert "/usr/bin" in path_segments

    async def test_user_env_overrides_runtime_pins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        merged = {**WORKSPACE_RUNTIME_ENV, "VIRTUAL_ENV": "/custom/venv"}
        spec = _make_spec(environment=merged)
        argv = await _capture_docker_argv(spec, monkeypatch)
        env = _env_dict_from_argv(argv)

        assert env["VIRTUAL_ENV"] == "/custom/venv"
        assert env["NPM_CONFIG_PREFIX"] == "/workspace/.npm"


class TestWorkspaceRuntimeDirsSetup:
    """ensure_workspace_runtime_dirs runs the right command via the backend."""

    async def test_post_start_creates_venv_and_npm_dirs(self) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await ensure_workspace_runtime_dirs(backend, handle)

        commands = [c[1]["command"] for c in backend.calls if c[0] == "exec"]
        joined = "\n".join(commands)
        assert "python3 -m venv /workspace/.venv" in joined
        assert "/workspace/.npm/lib" in joined
        assert "/workspace/.npm/bin" in joined

    async def test_venv_creation_is_idempotent(self) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await ensure_workspace_runtime_dirs(backend, handle)

        commands = [c[1]["command"] for c in backend.calls if c[0] == "exec"]
        joined = "\n".join(commands)
        assert "[ -e /workspace/.venv/bin/python ]" in joined
        assert "||" in joined

    async def test_failure_is_logged_not_raised(self) -> None:
        """Failure of the setup exec must not propagate — the model can
        still operate without the persistence layer."""
        backend = FakeBackend()
        backend.next_result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="boom",
            timed_out=False,
            truncated=False,
        )
        handle = make_handle()

        # Must not raise.
        await ensure_workspace_runtime_dirs(backend, handle)
