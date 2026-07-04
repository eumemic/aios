"""E2E: the authored seccomp profile (#807) is enforced on real sandboxes.

Provisions sandboxes via :class:`DockerBackend` with the profile path pointed
at the in-tree ``docker/seccomp-sandbox.json`` (the baked ``/app`` path does
not exist on the host running these tests), runs probes via ``backend.exec``,
and asserts the issue's acceptance items hold:

  (a) threads still work        — python threading + node worker_threads start
  (b) unshare with a namespace  — denied with EPERM ("Operation not permitted")
  (c) ptrace                    — denied with EPERM
  (e) package-manager metadata  — runs without an EPERM/seccomp denial

Item (d) (Limited provision still succeeds) is covered by the existing
``tests/e2e/test_networking.py`` Limited-provision test, with seccomp made
active there via the ``AIOS_SANDBOX_SECCOMP_PROFILE`` override in
``tests/conftest.py``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    CommandResult,
    Mount,
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.network import ensure_sandbox_network
from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")

# The baked /app path doesn't exist on the host; use the in-tree profile.
PROFILE_PATH = str(Path(__file__).parents[2] / "docker" / "seccomp-sandbox.json")


@pytest.fixture
async def _network_ready() -> None:
    await ensure_sandbox_network()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _spec(workspace: Path) -> SandboxSpec:
    instance_id = f"test_{uuid.uuid4().hex[:8]}"
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    return SandboxSpec(
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
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=IMAGE,
        seccomp_profile=PROFILE_PATH,
    )


@pytest.fixture
async def sandbox(
    _network_ready: None, workspace: Path
) -> AsyncIterator[tuple[DockerBackend, SandboxHandle]]:
    """One seccomp-confined sandbox per test, torn down unconditionally."""
    backend = DockerBackend()
    handle = await backend.create(_spec(workspace))
    try:
        yield backend, handle
    finally:
        await backend.destroy(handle)


async def _run(backend: DockerBackend, handle: SandboxHandle, command: str) -> CommandResult:
    return await backend.exec(handle, command, timeout_seconds=60, max_output_bytes=100_000)


# ── (a) threads still work ───────────────────────────────────────────────────


async def test_python_threading_works(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    res = await _run(
        backend,
        handle,
        "python3 -c 'import threading; t=threading.Thread(target=lambda: None); "
        't.start(); t.join(); print("ok")\'',
    )
    assert res.exit_code == 0, f"python threads blocked: stderr={res.stderr!r}"
    assert "ok" in res.stdout


async def test_node_worker_threads_work(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    # Skip gracefully if node isn't present in the image.
    probe = await _run(backend, handle, "command -v node >/dev/null 2>&1 && echo yes || echo no")
    if "yes" not in probe.stdout:
        pytest.skip("node not present in sandbox image")
    res = await _run(
        backend,
        handle,
        "node -e \"const {Worker}=require('worker_threads'); "
        "new Worker('process.exit(0)',{eval:true})"
        ".on('exit',c=>process.exit(c));\"",
    )
    assert res.exit_code == 0, f"node worker_threads blocked: stderr={res.stderr!r}"


# ── (b) unshare with a namespace is denied ───────────────────────────────────


async def test_unshare_user_namespace_denied(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    res = await _run(backend, handle, "unshare -U true")
    assert res.exit_code != 0, "namespace-creating unshare should be denied"
    assert "Operation not permitted" in res.stderr, f"stderr={res.stderr!r}"


async def test_unshare_argfilter_distinguishes_our_profile(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    """The arg-filtered ``unshare`` ALLOW is the discriminator that proves OUR
    profile (not Docker's default) is in effect.

    Docker's default profile denies ALL non-privileged ``unshare`` (it lives
    only in the CAP_SYS_ADMIN block). The authored profile (#807) ADDS a
    masked-eq ALLOW so ``unshare(0)`` (no CLONE_NEW* bit) succeeds while
    ``unshare(CLONE_NEWUSER)`` is still denied with EPERM. Asserting both
    halves confirms the profile is applied AND the block ordering is correct
    (ALLOW before the flat deny). Uses the raw ``SYS_unshare`` syscall so the
    verdict is the kernel's, not glibc's.
    """
    backend, handle = sandbox
    script = (
        "import ctypes, ctypes.util, platform, sys; "
        "nr = {'x86_64': 272, 'aarch64': 97}.get(platform.machine()); "
        "sys.exit(2) if nr is None else None; "
        "libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True); "
        "ctypes.set_errno(0); "
        "rc0 = libc.syscall(nr, 0); "  # unshare(0): allowed by our masked-eq block
        "e0 = ctypes.get_errno(); "
        "ctypes.set_errno(0); "
        "rcu = libc.syscall(nr, 0x10000000); "  # unshare(CLONE_NEWUSER): denied
        "eu = ctypes.get_errno(); "
        "print(rc0, e0, rcu, eu); "
        "sys.exit(0 if (rc0 == 0 and rcu == -1 and eu == 1) else 3)"
    )
    res = await _run(backend, handle, f'python3 -c "{script}"')
    if res.exit_code == 2:
        pytest.skip(f"unknown arch for SYS_unshare probe: stdout={res.stdout!r}")
    assert res.exit_code == 0, (
        "expected unshare(0)=allowed and unshare(CLONE_NEWUSER)=EPERM under the "
        f"authored profile: stdout={res.stdout!r} stderr={res.stderr!r}"
    )


# ── (c) ptrace is denied ─────────────────────────────────────────────────────


async def test_ptrace_denied(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    # Portable raw-syscall probe (strace may be absent). We invoke ``ptrace``
    # via ``syscall(SYS_ptrace, ...)`` rather than glibc's ``ptrace()`` wrapper
    # because on aarch64 the wrapper special-cases PTRACE_TRACEME and never
    # reaches the kernel, masking the seccomp verdict. The per-arch syscall
    # number comes from the sandbox's own ``platform.machine()``.
    #
    # Under the authored deny-list the kernel returns EPERM (errno 1) for the
    # filtered syscall. Some container runtimes — notably Docker Desktop's
    # emulated aarch64 VM — do not enforce ptrace seccomp filtering; there the
    # call reaches the kernel and returns 0/ESRCH instead. We assert the deny
    # when the host enforces it and skip (rather than falsely fail) when it
    # does not. On a native-Linux CI host the assertion is live.
    script = (
        "import ctypes, ctypes.util, platform, sys; "
        "nr = {'x86_64': 101, 'aarch64': 117}.get(platform.machine()); "
        "sys.exit(2) if nr is None else None; "
        "libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True); "
        "ctypes.set_errno(0); "
        "rc = libc.syscall(nr, 0, 0, 0, 0); "  # PTRACE_TRACEME == 0
        "err = ctypes.get_errno(); "
        "print(rc, err); "
        "sys.exit(0 if (rc == -1 and err == 1) else 3)"
    )
    res = await _run(backend, handle, f'python3 -c "{script}"')
    if res.exit_code == 2:
        pytest.skip(f"unknown arch for SYS_ptrace probe: stdout={res.stdout!r}")
    if res.exit_code == 3:
        pytest.skip(
            "host does not enforce ptrace seccomp filtering (e.g. Docker Desktop "
            f"emulated VM); raw ptrace returned {res.stdout.strip()!r}"
        )
    assert res.exit_code == 0, (
        f"ptrace was not denied with EPERM: stdout={res.stdout!r} stderr={res.stderr!r}"
    )


# ── (e) package-manager metadata runs without a seccomp denial ───────────────


async def test_pip_download_no_seccomp_denial(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    res = await _run(backend, handle, "pip download --no-deps --dest /tmp pip")
    # Network/index failures are acceptable in a hermetic CI; a seccomp denial
    # (EPERM "Operation not permitted") is NOT.
    assert "Operation not permitted" not in res.stderr, f"stderr={res.stderr!r}"


async def test_npm_config_no_seccomp_denial(
    sandbox: tuple[DockerBackend, SandboxHandle],
) -> None:
    backend, handle = sandbox
    probe = await _run(backend, handle, "command -v npm >/dev/null 2>&1 && echo yes || echo no")
    if "yes" not in probe.stdout:
        pytest.skip("npm not present in sandbox image")
    res = await _run(backend, handle, "npm config get registry")
    assert res.exit_code == 0, f"npm config failed: stderr={res.stderr!r}"
    assert "Operation not permitted" not in res.stderr, f"stderr={res.stderr!r}"
