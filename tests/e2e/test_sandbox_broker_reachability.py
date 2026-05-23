"""E2E test that a sandbox container can resolve and reach
``aios-worker`` via Docker DNS on the ``aios-sandbox`` network.

Boots a sidecar with that alias running ``python -m http.server``,
spawns a sandbox via :class:`DockerBackend` with
``host_gateway_alias=None`` (DNS-only resolution), and curls the alias
from inside the sandbox."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Mount,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.network import (
    SANDBOX_NETWORK_NAME,
    WORKER_NETWORK_ALIAS,
    ensure_sandbox_network,
)
from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SIDECAR_PORT = 7777


async def _run(argv: list[str], *, deadline_s: float) -> subprocess.CompletedProcess[str]:
    """Run a subprocess in a worker thread (async functions must not call
    blocking ``subprocess.run`` directly)."""
    return await asyncio.to_thread(
        subprocess.run,
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=deadline_s,
    )


@pytest.fixture
async def _network_ready() -> None:
    """Create the sandbox network if absent. No-op cleanup — other tests
    on this host (and the live worker) keep using the network."""
    await ensure_sandbox_network()


@pytest.fixture
async def broker_sidecar(_network_ready: None) -> AsyncIterator[None]:
    """Start an ``aios-worker``-aliased container running an HTTP server.

    Stands in for the worker's broker on the sandbox network. Cleans
    up unconditionally so a failed assertion doesn't leak the container.
    """
    name = f"aios-broker-sidecar-{uuid.uuid4().hex[:8]}"
    result = await _run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            SANDBOX_NETWORK_NAME,
            "--network-alias",
            WORKER_NETWORK_ALIAS,
            IMAGE,
            "python3",
            "-m",
            "http.server",
            str(SIDECAR_PORT),
        ],
        deadline_s=30,
    )
    if result.returncode != 0:
        pytest.fail(f"sidecar docker run failed: {result.stderr.strip()}")
    container_id = result.stdout.strip()

    try:
        yield
    finally:
        await _run(
            ["docker", "rm", "--force", container_id],
            deadline_s=15,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


async def test_sandbox_resolves_worker_alias_via_docker_dns(
    broker_sidecar: None, workspace: Path
) -> None:
    backend = DockerBackend()
    instance_id = f"test_{uuid.uuid4().hex[:8]}"
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    spec = SandboxSpec(
        session_id=session_id,
        instance_id=instance_id,
        workspace=Mount(host_path=workspace, sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
            INSTANCE_LABEL_KEY: instance_id,
            SESSION_LABEL_KEY: session_id,
        },
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image=IMAGE,
    )
    handle = await backend.create(spec)
    try:
        # ``--retry-connrefused`` absorbs the ~few-hundred-ms race between
        # docker reporting "container started" and the sidecar's
        # http.server actually binding the port.
        result = await backend.exec(
            handle,
            (
                f"curl -fs --max-time 5 --retry 10 --retry-delay 1 "
                f"--retry-connrefused http://{WORKER_NETWORK_ALIAS}:{SIDECAR_PORT}/"
            ),
            timeout_seconds=20,
            max_output_bytes=10_000,
        )
        assert result.exit_code == 0, (
            f"sandbox failed to reach {WORKER_NETWORK_ALIAS}:{SIDECAR_PORT}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
    finally:
        await backend.destroy(handle)
