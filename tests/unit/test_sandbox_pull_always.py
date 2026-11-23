"""Unit tests for fresh-pull semantics on sandbox creation (#567, #390).

A registry image is pulled fresh on every cold start — but exactly ONCE, via an
explicit ``docker pull``; the container then launches from the resolved
immutable image ID with no ``--pull`` flag, so the image it runs is the one
stamped as ``aios.base_image_id`` even if the tag moves mid-create.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import Mount, SandboxSpec
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
        "network_policy": UnrestrictedNetworking(),
        "host_gateway_alias": None,
        "image": "ghcr.io/eumemic/aios-sandbox:test",
    }
    base.update(overrides)
    return SandboxSpec(**base)  # type: ignore[arg-type]


async def _capture(spec: SandboxSpec) -> dict[str, list[str]]:
    """Run DockerBackend.create against a stubbed docker subprocess.

    Returns the ``docker run`` argv (``"argv"``) and the refs passed to
    ``docker pull`` (``"pulls"``).
    """
    captured: dict[str, list[str]] = {}
    pulls: list[str] = []

    async def fake_run_docker(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s, snapshot_timeout
        if argv[1] == "run":
            captured["argv"] = argv
        if argv[1] == "pull":
            pulls.append(argv[2])
        if argv[1:3] == ["image", "inspect"]:
            return 0, b"sha256:resolved\t1\t1\t{}\n", b""
        return 0, b"deadbeef1234\n", b""

    with patch("aios.sandbox.backends.docker.run_docker_cli", side_effect=fake_run_docker):
        await DockerBackend().create(spec)
    captured["pulls"] = pulls
    return captured


class TestFreshPull:
    @pytest.mark.asyncio
    async def test_registry_image_pulled_exactly_once(self) -> None:
        spec = _spec()
        out = await _capture(spec)
        assert out["pulls"] == [spec.image]
        assert "--pull" not in out["argv"]

    @pytest.mark.asyncio
    async def test_registry_image_launched_by_resolved_id(self) -> None:
        out = await _capture(_spec())
        assert out["argv"][-1] == "sha256:resolved"

    @pytest.mark.asyncio
    async def test_localhost_registry_pulled_and_launched_by_id(self) -> None:
        spec = _spec(image="localhost:5000/foo:bar")
        out = await _capture(spec)
        assert out["pulls"] == [spec.image]
        assert "--pull" not in out["argv"]
        assert out["argv"][-1] == "sha256:resolved"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("image", ["aios-sandbox:latest", "aios-sandbox"])
    async def test_local_image_not_pulled(self, image: str) -> None:
        out = await _capture(_spec(image=image))
        assert out["pulls"] == []
        assert "--pull" not in out["argv"]
        assert out["argv"][-1] == image
