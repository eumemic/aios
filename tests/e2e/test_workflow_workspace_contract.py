"""Executed filesystem contract for shared and fresh workflow child workspaces."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import Mount, SandboxSpec
from aios.sandbox.backends.docker import DockerBackend
from tests.conftest import needs_docker
from tests.helpers.sandbox import run_sandbox

pytestmark = [needs_docker, pytest.mark.docker]
IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SECCOMP_PROFILE = str(Path(__file__).parents[2] / "docker" / "seccomp-sandbox.json")


def _spec(owner: str, workspace: Path) -> SandboxSpec:
    return SandboxSpec(
        session_id=owner,
        instance_id="workflow_workspace_contract",
        workspace=Mount(host_path=workspace, sandbox_path="/workspace"),
        extra_mounts=(),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        labels={},
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=IMAGE,
        snapshot_image=None,
        seccomp_profile=SECCOMP_PROFILE,
    )


async def test_shared_siblings_see_both_writes_while_fresh_child_is_isolated(
    tmp_path: Path,
) -> None:
    backend = DockerBackend()
    shared = tmp_path / "launcher"
    fresh = tmp_path / "fresh-child"
    shared.mkdir()
    fresh.mkdir()
    sibling_a = await backend.create(_spec("wfr_sibling_a", shared))
    sibling_b = await backend.create(_spec("wfr_sibling_b", shared))
    child_fresh = await backend.create(_spec("wfr_fresh", fresh))
    try:
        await run_sandbox(backend, sibling_a, "printf from-a > /workspace/from-a")
        assert (await run_sandbox(backend, sibling_b, "cat /workspace/from-a"))[
            1
        ].strip() == "from-a"
        await run_sandbox(backend, sibling_b, "printf from-b > /workspace/from-b")
        assert (await run_sandbox(backend, sibling_a, "cat /workspace/from-b"))[
            1
        ].strip() == "from-b"
        result = await backend.exec(
            child_fresh,
            "test ! -e /workspace/from-a && test ! -e /workspace/from-b",
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert result.exit_code == 0
    finally:
        await backend.destroy(sibling_a)
        await backend.destroy(sibling_b)
        await backend.destroy(child_fresh)
