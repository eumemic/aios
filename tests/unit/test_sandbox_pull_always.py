"""Unit tests verifying ``--pull always`` is added to the docker run argv (#567).

Ensures the flag is present and appears before the image name so Docker
pulls a fresh copy of the image on every sandbox creation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aios.sandbox.backends.base import Mount, SandboxSpec, Unrestricted
from aios.sandbox.backends.docker import DockerBackend


def _spec(**overrides: object) -> SandboxSpec:
    """Build a minimal SandboxSpec for the pull-always tests."""
    base: dict[str, object] = {
        "session_id": "sess_pull",
        "instance_id": "inst_pull",
        "workspace": Mount(
            host_path=Path("/tmp/ws"),
            sandbox_path="/workspace",
            read_only=False,
        ),
        "extra_mounts": (),
        "environment": {},
        "labels": {},
        "network_policy": Unrestricted(),
        "host_gateway_alias": None,
        "image": "ghcr.io/eumemic/aios-sandbox:test",
    }
    base.update(overrides)
    return SandboxSpec(**base)  # type: ignore[arg-type]


async def _capture_argv(spec: SandboxSpec) -> list[str]:
    """Run DockerBackend.create against a stubbed docker subprocess.

    Returns the argv the backend would have invoked.
    """
    captured: dict[str, list[str]] = {}

    async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
        captured["argv"] = argv
        return 0, b"deadbeef1234\n", b""

    with patch("aios.sandbox.backends.docker.run_docker_cli", side_effect=fake_run_docker):
        await DockerBackend().create(spec)
    return captured["argv"]


class TestPullAlwaysFlag:
    @pytest.mark.asyncio
    async def test_pull_always_flag_present(self) -> None:
        """``--pull always`` must appear in the docker run argv."""
        argv = await _capture_argv(_spec())
        assert "--pull" in argv
        i = argv.index("--pull")
        assert argv[i + 1] == "always"

    @pytest.mark.asyncio
    async def test_pull_always_precedes_image(self) -> None:
        """``--pull`` must appear before the image name in the argv."""
        spec = _spec()
        argv = await _capture_argv(spec)
        pull_index = argv.index("--pull")
        image_index = argv.index(spec.image)
        assert pull_index < image_index

    @pytest.mark.asyncio
    async def test_pull_always_absent_for_local_image(self) -> None:
        """``--pull`` must NOT appear when the image is a bare local tag."""
        spec = _spec(image="aios-sandbox:latest")
        argv = await _capture_argv(spec)
        assert "--pull" not in argv
