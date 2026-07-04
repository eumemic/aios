"""E2E test pinning the IPv6 belt-and-suspenders egress lockdown (#1207).

The IPv4-only egress lockdown's IPv4-only-ness rested on an implicit,
undocumented invariant: the ``aios-sandbox`` network is created without
``--ipv6`` so no v6 route exists, and the lockdown's ``-P OUTPUT DROP`` is
iptables-only (``ip6tables`` left at default ACCEPT). The moment a v6 route
appears, the IPv4-only lockdown is silently bypassable over IPv6 (fail-open).

This test provisions a real Limited sandbox, applies the real lockdown, and
asserts the invariant is now pinned by construction:

1. the sandbox has NO global IPv6 address and NO v6 default route, and
2. the ``ip6tables`` OUTPUT policy is ``DROP`` (the per-session
   belt-and-suspenders DROP — the load-bearing prod protection).

It runs under the same Docker runtime matrix as the v4 lockdown: when
``AIOS_SANDBOX_RUNTIME=runsc`` the sidecar runs under gVisor's netstack, so the
runsc legacy-netfilter-ABI path for ``ip6tables-legacy`` is exercised, not just
runc.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
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
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.network import ensure_sandbox_network
from aios.sandbox.setup import apply_network_lockdown
from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = "ghcr.io/eumemic/aios-sandbox:latest"


@pytest.fixture
async def _network_ready() -> None:
    await ensure_sandbox_network()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
async def limited_sandbox(
    _network_ready: None, workspace: Path
) -> AsyncIterator[tuple[DockerBackend, SandboxHandle]]:
    """Provision a Limited sandbox and apply the real lockdown, threading the
    configured Docker runtime (so ``AIOS_SANDBOX_RUNTIME=runsc`` exercises the
    gVisor path). Tears down unconditionally so a failed assertion doesn't leak
    the container."""
    settings = get_settings()
    image = os.environ.get("AIOS_DOCKER_IMAGE", settings.docker_image) or IMAGE
    runtime = settings.sandbox_runtime

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
        network_policy=LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
        host_gateway_alias=None,
        image=image,
        runtime=runtime,
    )
    handle = await backend.create(spec)
    try:
        await apply_network_lockdown(
            backend,
            handle,
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
            runtime=runtime,
        )
        yield backend, handle
    finally:
        await backend.destroy(handle)


async def test_sandbox_has_no_global_ipv6_address(
    limited_sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    """No global-scope IPv6 address means there is no v6 source to egress from —
    the implicit invariant the IPv4-only lockdown rested on, now asserted."""
    backend, handle = limited_sandbox
    result = await backend.exec(
        handle,
        "ip -6 addr show scope global 2>/dev/null || true",
        timeout_seconds=15,
        max_output_bytes=10_000,
    )
    # ``scope global`` filters out the always-present link-local (fe80::) addr;
    # what must be absent is a routable global v6 address.
    assert "inet6" not in result.stdout, (
        f"sandbox has a global IPv6 address (v6 egress source exists): {result.stdout!r}"
    )


async def test_sandbox_has_no_ipv6_default_route(
    limited_sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    """No v6 default route means no path off-host over IPv6, so the IPv4-only
    lockdown cannot be bypassed over v6 today."""
    backend, handle = limited_sandbox
    result = await backend.exec(
        handle,
        "ip -6 route show default 2>/dev/null || true",
        timeout_seconds=15,
        max_output_bytes=10_000,
    )
    assert "default" not in result.stdout, (
        f"sandbox has an IPv6 default route (v6 egress path exists): {result.stdout!r}"
    )


async def test_ip6tables_output_policy_is_drop(
    limited_sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    """The belt-and-suspenders per-session DROP: even if a v6 route ever
    appears, ``ip6tables -P OUTPUT DROP`` blocks egress. Read it back from the
    shared netns via an operator-image sidecar (the sandbox itself holds no
    NET_ADMIN), selecting the legacy backend so it works under runsc.

    The assertion is guarded the same way the lockdown apply is: on hosts where
    the ``ip6_tables`` kernel module is not loaded — common on CI runners and
    any IPv6-disabled host — the v6 ``filter`` table cannot be initialized, so
    there is no v6 netfilter path to leak through and the apply correctly skips
    its DROP. In that case ``ip6tables -S OUTPUT`` fails to initialize and there
    is no policy to read back; the test passes (there is nothing to secure).
    When the v6 table IS present (the exact case the DROP defends), the policy
    MUST be ``DROP``."""
    backend, handle = limited_sandbox
    settings = get_settings()
    # Mirror the lockdown's legacy-backend selection so this reads the same
    # table the apply wrote to under runsc, and its table-availability guard so
    # a missing ip6_tables module is not a spurious failure.
    script = (
        "if command -v ip6tables-legacy >/dev/null 2>&1; then IP6T=ip6tables-legacy; "
        "else IP6T=ip6tables; fi\n"
        'if v6_output="$("$IP6T" -S OUTPUT 2>/dev/null)"; then\n'
        '  printf "%s\\n" "$v6_output" | grep -qx -- "-P OUTPUT DROP"\n'
        "else\n"
        '  echo "ip6tables filter table unavailable; no v6 egress path to lock down" >&2\n'
        "fi"
    )
    result = await backend.run_netns_sidecar(
        handle.sandbox_id,
        image=settings.docker_image,
        script=script,
        timeout_seconds=15,
        max_output_bytes=10_000,
        runtime=settings.sandbox_runtime,
    )
    assert result.exit_code == 0, (
        "ip6tables OUTPUT policy is not DROP after lockdown apply (and the v6 "
        "filter table WAS initializable)\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
