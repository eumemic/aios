"""Docker runtime flag plumbing for gVisor/runsc selection (#1014)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.sandbox.backends import docker as docker_backend
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


def _spec(*, runtime: str | None = None) -> SandboxSpec:
    return SandboxSpec(
        session_id="sess_runtime",
        instance_id="inst_runtime",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
            INSTANCE_LABEL_KEY: "inst_runtime",
            SESSION_LABEL_KEY: "sess_runtime",
        },
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
        runtime=runtime,
    )


def _runtime_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--runtime"]


async def test_create_omits_runtime_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float = 30.0) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec())

    assert _runtime_values(calls[0]) == []


async def test_create_emits_configured_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float = 30.0) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc"))

    assert _runtime_values(calls[0]) == ["runsc"]


async def test_netns_sidecar_emits_runtime_when_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"", b"", False

    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_run)

    await DockerBackend().run_netns_sidecar(
        "sandbox123",
        image="aios-sandbox:test",
        script="true",
        timeout_seconds=5,
        max_output_bytes=1024,
        runtime="runsc",
    )

    assert _runtime_values(calls[0]) == ["runsc"]
    # The flag lands before the image (i.e. on the docker run options, not
    # inside the in-container command).
    assert calls[0].index("--runtime") < calls[0].index("aios-sandbox:test")


async def test_netns_sidecar_omits_runtime_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"", b"", False

    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_run)

    await DockerBackend().run_netns_sidecar(
        "sandbox123",
        image="aios-sandbox:test",
        script="true",
        timeout_seconds=5,
        max_output_bytes=1024,
    )

    assert _runtime_values(calls[0]) == []
