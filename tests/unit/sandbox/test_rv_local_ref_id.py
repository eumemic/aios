"""#390: non-registry base refs must also be stamped with their content ID.

``_is_registry_image`` only decides whether a fresh pull is safe; it must not
gate drift detection. Docker Hub short forms and bare local tags can advance
behind an unchanged ref (operator re-pull / local rebuild) just like ghcr tags.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import (
    BASE_IMAGE_ID_LABEL_KEY,
    Mount,
    SandboxBackendError,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend


def _spec(image: str) -> SandboxSpec:
    return SandboxSpec(
        session_id="s",
        instance_id="i",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=image,
    )


@pytest.mark.parametrize("image", ["myorg/image:stable", "ubuntu:latest", "aios-sandbox:latest"])
async def test_non_registry_cold_start_stamps_id(
    monkeypatch: pytest.MonkeyPatch, image: str
) -> None:
    run_argv: list[str] = []
    pulls: list[str] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s, snapshot_timeout
        if argv[1:3] == ["image", "inspect"]:
            return 0, b"sha256:local\t1\t1\t{}\n", b""
        if argv[1] == "pull":
            pulls.append(argv[2])
        if argv[1] == "run":
            run_argv.extend(argv)
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    await DockerBackend().create(_spec(image))
    assert f"{BASE_IMAGE_ID_LABEL_KEY}=sha256:local" in run_argv
    assert run_argv[-1] == "sha256:local"
    assert pulls == []  # present locally: never pulled (not a pull-always ref)


async def test_absent_local_ref_pulled_like_docker_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent locally ⇒ pull once (docker run's implicit --pull missing), then stamp."""
    pulled = False
    run_argv: list[str] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        nonlocal pulled
        del timeout_s, snapshot_timeout
        if argv[1:3] == ["image", "inspect"]:
            if not pulled:
                return 1, b"", b"Error: No such image: ubuntu:latest"
            return 0, b"sha256:hub\t1\t1\t{}\n", b""
        if argv[1] == "pull":
            pulled = True
        if argv[1] == "run":
            run_argv.extend(argv)
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    await DockerBackend().create(_spec("ubuntu:latest"))
    assert pulled
    assert f"{BASE_IMAGE_ID_LABEL_KEY}=sha256:hub" in run_argv
    assert run_argv[-1] == "sha256:hub"


async def test_resume_resolves_rebuilt_local_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """image_id on a local tag reflects a local rebuild (new content ID), no pull."""
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s, snapshot_timeout
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return 0, b"sha256:rebuilt\t1\t1\t{}\n", b""
        return 0, b"", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    assert await DockerBackend().image_id("aios-sandbox:latest") == "sha256:rebuilt"
    assert not any(c[1] == "pull" for c in calls)


def _fake_docker(
    *, pull_rc: int, calls: list[list[str]]
) -> Callable[..., Awaitable[tuple[int, bytes, bytes]]]:
    """Inspect always reports the image absent; pull returns ``pull_rc``."""

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s, snapshot_timeout
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return 1, b"", b"Error: No such image"
        if argv[1] == "pull":
            if pull_rc != 0:
                return pull_rc, b"", b"pull access denied"
            return 0, b"", b""
        return 0, b"", b""

    return fake_run


@pytest.mark.parametrize("image", ["aios-sandbox:latest", "ghcr.io/eumemic/aios-sandbox:latest"])
async def test_unpullable_ref_raises_pull_failure(
    monkeypatch: pytest.MonkeyPatch, image: str
) -> None:
    """A failed pull surfaces as the pull failure itself — not masked by a re-inspect."""
    calls: list[list[str]] = []
    monkeypatch.setattr(docker_backend, "run_docker_cli", _fake_docker(pull_rc=1, calls=calls))
    result: str | None = None
    with pytest.raises(SandboxBackendError, match="docker pull failed") as excinfo:
        result = await DockerBackend().image_id(image)
    assert result is None
    assert "pull access denied" in str(excinfo.value)
    # The failed pull aborts resolution: nothing after the pull is consulted.
    assert calls[-1][1] == "pull"


@pytest.mark.parametrize("image", ["aios-sandbox:latest", "ghcr.io/eumemic/aios-sandbox:latest"])
async def test_pull_ok_but_reinspect_absent_raises_not_found(
    monkeypatch: pytest.MonkeyPatch, image: str
) -> None:
    """Pull 'succeeds' yet the image is still absent ⇒ SandboxBackendError, not TypeError.

    ``SandboxRegistry._resolve_snapshot`` deliberately catches only
    ``SandboxBackendError``; any other exception type would break resume.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(docker_backend, "run_docker_cli", _fake_docker(pull_rc=0, calls=calls))
    with pytest.raises(SandboxBackendError, match="image not found"):
        await DockerBackend().image_id(image)
    assert any(c[1] == "pull" for c in calls)
