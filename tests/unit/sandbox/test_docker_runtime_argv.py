"""Docker runtime flag plumbing for gVisor/runsc selection (#1014)."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends import docker as docker_backend
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


@pytest.fixture(autouse=True)
def _on_x86_64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the machine every runsc test in this module implicitly assumes.

    The runsc paths refuse anything but x86_64/amd64 (the operator image's ELF
    loader lives at an x86_64 triple), so without this the runsc argv tests
    would pass or fail depending on the developer's laptop — red on Apple
    Silicon, green on a CI runner. The arch tests below override it.
    """
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")


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
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
        runtime=runtime,
    )


def _runtime_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--runtime"]


async def test_create_omits_runtime_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec())

    assert _runtime_values(calls[0]) == []
    # The operator-image mount is runsc-only: it costs a read-only image mount
    # per sandbox and needs Docker's containerd image store, neither of which
    # the runc path (the production default) has any use for.
    assert "--mount" not in calls[0]


async def test_create_emits_configured_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc"))

    assert _runtime_values(calls[0]) == ["runsc"]
    mount = calls[0][calls[0].index("--mount") + 1]
    assert mount == (
        "type=image,src=ghcr.io/eumemic/aios-sandbox:latest,dst=/run/aios-operator-root"
    )


# Every machine the operator image's x86_64 ELF paths cannot fit. aarch64 is
# the one that actually ships (``build-sandbox.yml`` publishes linux/arm64, so
# an Apple Silicon pull really does get a loader at another triple); the rest
# are here to pin that the guard is an ALLOW-list — an arch nobody thought
# about must fail closed too, not fall through to the x86_64 paths.
_UNSUPPORTED_MACHINES = ["aarch64", "arm64", "armv7l", "ppc64le", "riscv64", "i686"]


@pytest.mark.parametrize("machine", _UNSUPPORTED_MACHINES)
async def test_create_refuses_runsc_off_x86_64(
    monkeypatch: pytest.MonkeyPatch, machine: str
) -> None:
    """Refuse before ``docker run``, not inside the chroot.

    The operator chain is entered by an absolute x86_64 loader path; on any
    other machine that exec dies with an opaque ``exec format error`` after the
    sandbox is already up, or — worse — the sandbox comes up and only the
    egress apply fails, which is the shape that blackholes DNS.
    """
    monkeypatch.setattr(platform, "machine", lambda: machine)
    ran: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        ran.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    with pytest.raises(SandboxBackendError, match=f"unsupported on {machine}"):
        await DockerBackend().create(_spec(runtime="runsc"))
    assert ran == [], "the guard must fail closed before the daemon is touched"


@pytest.mark.parametrize("machine", _UNSUPPORTED_MACHINES)
async def test_netns_sidecar_refuses_runsc_off_x86_64(
    monkeypatch: pytest.MonkeyPatch, machine: str
) -> None:
    """The egress path carries the same guard as ``create``.

    It is reachable independently: a sandbox created on one worker can have its
    lockdown applied by another (and #1014 has the caller, not the backend,
    pass the runtime), so guarding only ``create`` would leave the privileged
    exec to fail inside the chroot instead.
    """
    monkeypatch.setattr(platform, "machine", lambda: machine)
    ran: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        del timeout_s
        ran.append(list(argv))
        return 0, b"", b"", False

    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_run)

    with pytest.raises(SandboxBackendError, match=f"unsupported on {machine}"):
        await DockerBackend().run_netns_sidecar(
            "sandbox123",
            image="aios-sandbox:test",
            script="true",
            timeout_seconds=5,
            max_output_bytes=1024,
            runtime="runsc",
        )
    assert ran == [], "the guard must fail closed before the daemon is touched"


@pytest.mark.parametrize("machine", ["aarch64", "arm64"])
async def test_arm64_still_gets_the_default_runtime(
    monkeypatch: pytest.MonkeyPatch, machine: str
) -> None:
    """Only runsc is refused. runc is the production default and is arch-neutral.

    Apple Silicon runs the whole test/dev stack on the default runtime; a guard
    that caught every sandbox rather than every *runsc* sandbox would take the
    platform out entirely.
    """
    monkeypatch.setattr(platform, "machine", lambda: machine)
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    async def fake_sidecar(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"", b"", False

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_sidecar)

    await DockerBackend().create(_spec())
    await DockerBackend().run_netns_sidecar(
        "sandbox123",
        image="aios-sandbox:test",
        script="true",
        timeout_seconds=5,
        max_output_bytes=1024,
        runtime=None,
    )

    assert [call[:2] for call in calls] == [["docker", "run"], ["docker", "run"]]
    assert _runtime_values(calls[0]) == []
    assert _runtime_values(calls[1]) == []


async def test_netns_sidecar_runsc_execs_into_the_target_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runsc installs the rules with ``docker exec``, not a second container.

    Two runsc containers sharing a Linux netns get separate Sentries, so a
    ``--network container:`` sidecar programs its own netstack and the target's
    tables stay empty (#2310, gvisor#170). The rules must be written from
    inside the target's own Sentry.
    """
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

    argv = calls[0]
    assert argv[:3] == ["docker", "exec", "--privileged"]
    # No second container, so no --runtime and no image argument.
    assert "run" not in argv
    assert _runtime_values(argv) == []
    assert "aios-sandbox:test" not in argv
    assert argv[argv.index("sandbox123") + 1 :][:1] == ["/run/aios-operator-root/usr/bin/busybox"]


async def test_netns_sidecar_runsc_runs_only_operator_image_binaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exec runs in the tenant's mount namespace; nothing it runs may be tenant-owned.

    Three things carry that: the ELF loader and shell are addressed inside the
    read-only operator mount (the kernel would otherwise resolve a binary's
    baked-in interpreter against the TENANT root), the preamble binds each
    command name to an operator-image path, and the environment the exec
    inherits — which is tenant-authored, ``EnvironmentConfig.env`` is free-form
    — is scrubbed of the loader/shell injection vectors.
    """
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"", b"", False

    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_run)

    await DockerBackend().run_netns_sidecar(
        "sandbox123",
        image="aios-sandbox:test",
        script="echo hi",
        timeout_seconds=5,
        max_output_bytes=1024,
        runtime="runsc",
    )

    argv = calls[0]
    root = "/run/aios-operator-root"
    target = argv.index("sandbox123")
    assert argv[target + 1 : target + 8] == [
        f"{root}/usr/bin/busybox",
        "chroot",
        root,
        "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        "--library-path",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/bin/bash",
    ], "a static chroot must hide tenant /etc/ld.so.preload before the loader starts"
    assert argv[-4:-2] == ["/usr/bin/bash", "-p"], (
        "the operator shell must come from the mount and run privileged: "
        "bash -p ignores BASH_ENV/ENV/SHELLOPTS/BASHOPTS and refuses to import "
        "BASH_FUNC_* functions from the tenant-authored container environment"
    )
    assert argv[-2] == "-c"
    envs = {argv[i + 1] for i, tok in enumerate(argv) if tok == "--env"}
    # --library-path changes where ld.so SEARCHES; it does not stop it
    # honouring these, either of which runs tenant code in the operator shell.
    assert "LD_PRELOAD=" in envs
    assert "LD_AUDIT=" in envs
    assert "LD_LIBRARY_PATH=" in envs
    # Post-chroot, ``/usr/{s,}bin`` ARE the operator image's: a command the
    # preamble forgot degrades to an operator binary, never a tenant one. The
    # pre-chroot ``{root}/...`` spelling would resolve to nothing at all.
    assert "PATH=/usr/sbin:/usr/bin" in envs
    script = argv[-1]
    assert script.endswith("echo hi")
    assert "OP=" in script
    assert "operator_exec /usr/sbin/iptables-legacy" in script


async def test_netns_sidecar_non_runsc_runtime_keeps_the_container_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-runsc runtime still joins the netns from a separate container.

    Only runsc has the separate-Sentry problem; every other runtime shares the
    netns for real, so the operator-image sidecar remains the right shape —
    and ``--runtime`` must still precede the image argument.
    """
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
        runtime="kata",
    )

    argv = calls[0]
    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "container:sandbox123"
    assert "--cap-add" in argv and argv[argv.index("--cap-add") + 1] == "NET_ADMIN"
    assert _runtime_values(argv) == ["kata"]
    assert argv.index("--runtime") < argv.index("aios-sandbox:test")
    assert argv[-4:] == ["aios-sandbox:test", "bash", "-c", "true"]


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

    argv = calls[0]
    assert argv[:2] == ["docker", "run"]
    assert _runtime_values(argv) == []
    assert argv[-4:] == ["aios-sandbox:test", "bash", "-c", "true"]
