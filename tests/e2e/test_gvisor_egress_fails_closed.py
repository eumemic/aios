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
            #
            # GATED ON THE RUNTIME (#2429 review, finding X4). Swallowing this
            # unconditionally would have made the test exactly the thing its own
            # docstring warns against: under runc a regression that made EVERY
            # Limited lockdown fail would land here and return GREEN. The runc
            # branch is the half that keeps this control honest, so under runc a
            # refusal must propagate. `raise` rather than pytest.fail to keep the
            # original traceback.
            if runtime != "runsc":
                raise
            return

        # The lockdown reported success, so it must actually BE in force -- a
        # success return with no DROP policy would be the fail-open we are
        # guarding against.
        #
        # READ IT BACK THROUGH THE SIDECAR, NOT backend.exec. The sandbox holds
        # no CAP_NET_ADMIN by design (docker.py: root-in-sandbox must not be able
        # to flush its own lockdown), and `iptables -S` is NOT a read-only
        # operation -- it getsockopts the table and needs that capability. An
        # in-sandbox read therefore fails with "can't initialize iptables table
        # `filter': Permission denied (you must be root)" even on a perfectly
        # locked-down box: a FALSE RED that accuses a correct sandbox of running
        # open. The first version of this test did exactly that and turned the
        # runc gate red (#2429 review, finding X1-NEW).
        #
        # The sidecar joins the same netns WITH NET_ADMIN, which is how the
        # sibling ip6tables read-back does it.
        settings_now = get_settings()
        script = (
            "if command -v iptables-legacy >/dev/null 2>&1; then IPT=iptables-legacy; "
            "else IPT=iptables; fi\n"
            '"$IPT" -S OUTPUT | grep -qx -- "-P OUTPUT DROP"'
        )
        result = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings_now.docker_image,
            script=script,
            timeout_seconds=20,
            max_output_bytes=20_000,
            runtime=runtime,
        )
        # Assert on the EXIT CODE, and deliberately do NOT swallow errors with
        # `2>&1 || true`. Laundering a failed table-init into stdout makes the
        # read succeed while proving nothing -- a fail-open dressed as a check.
        # If the table cannot be read, that is a hard failure here.
        assert result.exit_code == 0, (
            "apply_network_lockdown returned success but the OUTPUT policy is not "
            "DROP -- a Limited sandbox is running with open egress.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
    finally:
        await backend.destroy(handle)
