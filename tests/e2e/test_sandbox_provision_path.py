"""E2E: a provisioned sandbox resolves bare-name coreutils via the system PATH.

The SEV-1 regression guard for #936 (re-land of #925). #925 dropped the
explicit system PATH from the sandbox env; because ``docker --env`` does NOT
expand ``$PATH`` and the snapshot-resume/flatten path re-injects env via
``docker run --env`` with no image-config PATH, the keepalive
``CMD ["tail","-f","/dev/null"]`` could no longer resolve ``tail`` and ALL
provisioning crashlooped (#935 reverted it). The fix keeps an absolute system
PATH in :data:`aios.sandbox.setup.WORKSPACE_RUNTIME_ENV`.

These tests import the **production** ``WORKSPACE_RUNTIME_ENV`` and build the
spec's ``environment`` from it (rather than hand-writing a ``PATH``), so they
fail if PATH is ever dropped from that dict — fresh provision (case 1) AND
across a snapshot/resume round-trip (case 2, the ``docker.py::_flatten`` resume
path that re-injects env via ``docker run --env``).

They run real ``docker`` and pull/run the sandbox image; they are
``needs_docker`` + ``docker``-marked and skipped on hosts without a daemon.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    ENV_KEYS_LABEL_KEY,
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Mount,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.setup import WORKSPACE_RUNTIME_ENV
from aios.sandbox.spec import snapshot_tag
from tests.conftest import needs_docker
from tests.helpers.sandbox import run_sandbox

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SECCOMP_PROFILE = str(Path(__file__).parents[2] / "docker" / "seccomp-sandbox.json")


def _spec(
    *,
    instance_id: str,
    session_id: str,
    workspace: Path,
    snapshot_image: str | None = None,
) -> SandboxSpec:
    # The PRODUCTION runtime env (which carries the system PATH) — NOT a
    # hand-rolled ``{"PATH": ...}``, so dropping PATH from the source dict
    # fails this test rather than being masked.
    environment = dict(WORKSPACE_RUNTIME_ENV)
    return SandboxSpec(
        session_id=session_id,
        instance_id=instance_id,
        workspace=Mount(host_path=workspace, sandbox_path="/workspace"),
        extra_mounts=(),
        environment=environment,
        labels={
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
            INSTANCE_LABEL_KEY: instance_id,
            SESSION_LABEL_KEY: session_id,
            ENV_KEYS_LABEL_KEY: ",".join(sorted(environment)),
            BASE_IMAGE_LABEL_KEY: IMAGE,
        },
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=IMAGE,
        snapshot_image=snapshot_image,
        seccomp_profile=SECCOMP_PROFILE,
    )


async def test_fresh_provision_resolves_bare_name_coreutils(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    """A freshly-provisioned sandbox resolves bare-name coreutils on PATH.

    The keepalive ``CMD ["tail","-f","/dev/null"]`` needs ``tail`` on PATH to
    boot; this asserts the same property the daemon's CMD relies on, via the
    production ``WORKSPACE_RUNTIME_ENV``. If PATH is dropped from that dict,
    ``docker --env`` won't expand ``$PATH`` and these bare names won't resolve.
    """
    backend, instance_id, session_id, workspace = daemon

    handle = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    try:
        rc, out = await run_sandbox(backend, handle, "tail --version")
        assert rc == 0, f"`tail` did not resolve on a fresh provision: {out}"
        rc, out = await run_sandbox(backend, handle, "ls /tmp")
        assert rc == 0, f"`ls` did not resolve on a fresh provision: {out}"
    finally:
        await backend.destroy(handle)


async def test_snapshot_resumed_sandbox_resolves_bare_name_coreutils(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    """A snapshot-resumed sandbox still resolves bare-name coreutils on PATH.

    Guards the ``docker.py::_flatten`` resume path: PATH is re-injected via
    ``docker run --env`` (NOT the image config), so it must be in the run env
    or the resumed keepalive CMD can't resolve ``tail``. Forces the flatten
    path (``flatten_if_unique_bytes_over=0``) so the config-stripped image is
    exercised.
    """
    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    # Cycle 1: write a marker (well over the empty floor) and snapshot/flatten.
    h1 = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    await run_sandbox(backend, h1, "echo marker > /root/marker")
    await run_sandbox(backend, h1, "head -c 65536 /dev/zero > /root/blob")
    out = await backend.snapshot(
        h1.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=0
    )
    assert out.kind == "flattened", f"expected a flattened snapshot, got {out.kind}"
    await backend.destroy(h1)

    # Cycle 2: resume from the snapshot tag and resolve `tail` by bare name.
    h2 = await backend.create(
        _spec(
            instance_id=instance_id, session_id=session_id, workspace=workspace, snapshot_image=tag
        )
    )
    try:
        rc, out_txt = await run_sandbox(backend, h2, "tail --version")
        assert rc == 0, f"`tail` did not resolve after snapshot/resume: {out_txt}"
    finally:
        await backend.destroy(h2)
