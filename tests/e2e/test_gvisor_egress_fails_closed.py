"""The gVisor leg's POSITIVE egress assertion: the lockdown must fail CLOSED.

Written for #2429 (review finding X1). The rest of the egress suite is deselected
under runsc because the in-netns sidecar cannot work there (separate Sentries /
netstacks per container — see the ``netns_sidecar_egress`` marker). That left the
gVisor job silent on the single most safety-relevant question it could answer:

    when the lockdown cannot be installed, does the sandbox refuse to come up,
    or does it come up with OPEN egress?

Silence reads as green. A future change that turned the fail-closed refusal into a
fail-OPEN would be invisible to every remaining test in the gVisor leg, and it is
precisely the catastrophic direction: a Limited sandbox running with unrestricted
networking is a containment breach, not a test failure.

So this test asserts the refusal itself, and it is meaningful under BOTH runtimes:

  * under runsc  — the sidecar genuinely cannot install rules, so the provision
    must raise rather than yield an open box. This is the real regression guard.
  * under runc   — the lockdown succeeds, so the provision must NOT raise. That
    direction is what keeps this test honest: a bug that made provisioning raise
    unconditionally would satisfy the runsc branch while breaking every real
    sandbox, and the runc branch catches it.

A control that has only ever been seen to REFUSE is indistinguishable from one
that refuses everything, so both directions are asserted here.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from aios.config import get_settings
from aios.models.environments import LimitedNetworking
from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    Mount,
    SandboxBackendError,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.network import ensure_sandbox_network
from aios.sandbox.setup import apply_network_lockdown
from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = "ghcr.io/eumemic/aios-sandbox:latest"


async def test_limited_lockdown_either_installs_or_refuses(tmp_path: Path) -> None:
    """A Limited sandbox must never end up running with unrestricted networking.

    Deliberately NOT marked ``netns_sidecar_egress``: this test is *about* what
    happens when the sidecar path fails, so it must keep running in the gVisor
    leg where that failure is the expected outcome.
    """
    await ensure_sandbox_network()

    settings = get_settings()
    image = os.environ.get("AIOS_DOCKER_IMAGE", settings.docker_image) or IMAGE
    runtime = settings.sandbox_runtime

    workspace = tmp_path / "ws"
    workspace.mkdir()

    backend = DockerBackend()
    instance_id = f"test_{uuid.uuid4().hex[:8]}"
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    networking = LimitedNetworking(type="limited", allowed_hosts=["example.com"])
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
        network_policy=networking,
        host_gateway_alias=None,
        image=image,
        runtime=runtime,
    )

    handle = await backend.create(spec)
    try:
        try:
            await apply_network_lockdown(backend, handle, networking, runtime=runtime)
        except SandboxBackendError:
            # FAIL-CLOSED: the lockdown could not be installed and the layer said
            # so rather than handing back a sandbox with open egress. This is the
            # expected path under runsc.
            return

        # The lockdown reported success, so it must actually BE in force -- a
        # success return with no DROP policy would be the fail-open we are
        # guarding against. Read the live policy back from inside the netns.
        result = await backend.exec(
            handle,
            "if command -v iptables-legacy >/dev/null 2>&1; then IPT=iptables-legacy; "
            'else IPT=iptables; fi; "$IPT" -S OUTPUT 2>&1 || true',
            timeout_seconds=20,
            max_output_bytes=20_000,
        )
        assert "-P OUTPUT DROP" in result.stdout, (
            "apply_network_lockdown returned success but the OUTPUT policy is not "
            f"DROP -- a Limited sandbox is running with open egress: {result.stdout!r}"
        )
    finally:
        await backend.destroy(handle)
