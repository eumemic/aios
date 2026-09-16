"""Docker runtime flag plumbing for gVisor/runsc selection (#1014)."""

from __future__ import annotations

import json
import platform
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest

from aios.config import get_settings
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
from aios.sandbox.network import SANDBOX_NETWORK_NAME, WORKER_NETWORK_ALIAS


@pytest.fixture(autouse=True)
def _on_x86_64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the machine every runsc test in this module implicitly assumes.

    The runsc paths refuse anything but x86_64/amd64 (the operator image's ELF
    loader lives at an x86_64 triple), so without this the runsc argv tests
    would pass or fail depending on the developer's laptop — red on Apple
    Silicon, green on a CI runner. The arch tests below override it.
    """
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")


def _spec(
    *,
    runtime: str | None = None,
    image: str | None = None,
    seccomp_profile: str | None = None,
) -> SandboxSpec:
    # runsc sandboxes must run the operator image itself (the egress exec
    # chroots into it), so that is the default here; the runc path is
    # image-agnostic and the mismatch test overrides it.
    spec = SandboxSpec(
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
        image=image or get_settings().docker_image,
        runtime=runtime,
    )
    if seccomp_profile is not None:
        return replace(spec, seccomp_profile=seccomp_profile)
    return spec


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
    # runc reaches ``aios-worker`` through Docker's embedded DNS, so the runc
    # path neither looks the alias up nor bakes it into /etc/hosts.
    assert "--add-host" not in calls[0]
    assert not any(c[1] == "ps" for c in calls)


def _run_argv(calls: list[list[str]]) -> list[str]:
    return next(c for c in calls if len(c) >= 2 and c[1] == "run")


def _worker_endpoint(address: str = "172.18.0.9") -> bytes:
    return json.dumps(
        {SANDBOX_NETWORK_NAME: {"Aliases": [WORKER_NETWORK_ALIAS], "IPAddress": address}}
    ).encode()


def _runsc_responder(
    calls: list[list[str]], *, worker_address: str | None = "172.18.0.9"
) -> Callable[..., Awaitable[tuple[int, bytes, bytes]]]:
    """Answer the daemon calls a runsc create makes: the worker-alias probe
    (``docker ps`` + ``docker inspect``) and everything else."""

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        if argv[1] == "ps":
            return 0, b"" if worker_address is None else b"aaa\n", b""
        if (
            argv[1] == "inspect"
            and "--format" in argv
            and "{{json .NetworkSettings.Networks}}" in argv
        ):
            assert worker_address is not None
            return 0, _worker_endpoint(worker_address) + b"\n", b""
        return 0, b"deadbeefcafe\n", b""

    return fake_run


async def test_create_emits_configured_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    fake_run = _runsc_responder(calls)

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr("aios.sandbox.network.run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc"))

    run = _run_argv(calls)
    assert _runtime_values(run) == ["runsc"]
    mount = run[run.index("--mount") + 1]
    assert mount == (
        "type=image,src=ghcr.io/eumemic/aios-sandbox:latest,dst=/run/aios-operator-root"
    )
    # Derived from ``spec.image`` — the image the tenant container itself runs —
    # so the operator root can never silently disagree with the sandbox.
    assert f"src={_spec(runtime='runsc').image}," in mount


async def test_create_runsc_bakes_the_worker_alias_into_etc_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gVisor Sentry never sees the netns rules that make 127.0.0.11 answer,
    so the worker alias is resolved on the worker and passed as --add-host."""
    calls: list[list[str]] = []
    fake_run = _runsc_responder(calls, worker_address="172.19.0.4")

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr("aios.sandbox.network.run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc"))

    run = _run_argv(calls)
    assert run[run.index("--add-host") + 1] == f"{WORKER_NETWORK_ALIAS}:172.19.0.4"
    # ...and never through the embedded resolver the Sentry cannot reach.
    assert "--dns" not in run
    assert any(c[1] == "ps" for c in calls)


async def test_create_runsc_without_a_worker_on_the_network_omits_add_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No container claims the alias: create still succeeds (the sandbox that
    needs the broker fails loudly when it reaches for it) rather than inventing
    an address."""
    calls: list[list[str]] = []
    fake_run = _runsc_responder(calls, worker_address=None)

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr("aios.sandbox.network.run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc"))

    assert "--add-host" not in _run_argv(calls)


async def test_create_runsc_on_host_worker_keeps_the_host_gateway_add_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker-on-host is already hosts-based and is not on the sandbox
    network, so the alias probe never runs."""
    calls: list[list[str]] = []
    fake_run = _runsc_responder(calls, worker_address=None)

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr("aios.sandbox.network.run_docker_cli", fake_run)

    spec = replace(_spec(runtime="runsc"), host_gateway_alias=WORKER_NETWORK_ALIAS)
    await DockerBackend().create(spec)

    run = _run_argv(calls)
    assert run[run.index("--add-host") + 1] == f"{WORKER_NETWORK_ALIAS}:host-gateway"
    assert not any(c[1] == "ps" for c in calls)


async def test_create_refuses_runsc_on_a_tenant_supplied_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spec.image`` is tenant-authored (#724); an operator root is not.

    The operator root is the first thing the egress exec enters, holding
    ``NET_ADMIN`` the sandbox itself was denied — so a per-environment image
    override would otherwise get its own ``/usr/bin/busybox`` run privileged.
    Refusing beats silently mounting ``settings.docker_image`` under a sandbox
    built from something else.
    """
    ran: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        ran.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    with pytest.raises(SandboxBackendError, match="must run the operator image"):
        await DockerBackend().create(_spec(runtime="runsc", image="tenant/evil:latest"))
    assert ran == [], "the guard must fail closed before the daemon is touched"


async def test_create_allows_any_image_under_runc(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is runsc-only: runc applies its lockdown from a separate
    operator-image sidecar, so the sandbox image is free to be the tenant's."""
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec(image="tenant/custom:latest"))

    assert calls[0][-1] == "tenant/custom:latest"


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

    docker_runs = [c for c in calls if len(c) >= 2 and c[1] == "run"]
    assert len(docker_runs) == 2
    assert _runtime_values(docker_runs[0]) == []
    assert _runtime_values(docker_runs[1]) == []


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


_SECCOMP_SANDBOX = Path(__file__).parents[3] / "docker" / "seccomp-sandbox.json"


def _seccomp_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--security-opt"]


def _assert_runsc_clone3_profile(path: Path) -> None:
    assert path != _SECCOMP_SANDBOX, "runsc must not use the runc profile as-is"
    profile = json.loads(path.read_text())
    clone3_allow = next(
        blk
        for blk in profile["syscalls"]
        if blk.get("action") == "SCMP_ACT_ALLOW" and "clone3" in blk.get("names", [])
    )
    assert not clone3_allow.get("args"), "clone3 ALLOW must be unfiltered (struct flags)"
    assert not clone3_allow.get("includes")
    unshare_denied = any(
        blk.get("action") == "SCMP_ACT_ERRNO" and "unshare" in blk.get("names", [])
        for blk in profile["syscalls"]
    )
    assert unshare_denied, "CLONE_NEWUSER denial via unshare must survive the runsc copy"


async def test_create_runsc_emits_clone3_allow_seccomp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runsc OCI seccomp ignores errnoRet, so clone3 ENOSYS becomes EPERM and
    python/node threads die. The create path must swap in a profile that
    allows clone3 while keeping the authored unshare deny."""
    calls: list[list[str]] = []
    fake_run = _runsc_responder(calls)
    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    monkeypatch.setattr("aios.sandbox.network.run_docker_cli", fake_run)

    await DockerBackend().create(_spec(runtime="runsc", seccomp_profile=str(_SECCOMP_SANDBOX)))

    run = _run_argv(calls)
    seccomp = next(v for v in _seccomp_values(run) if v.startswith("seccomp="))
    _assert_runsc_clone3_profile(Path(seccomp.removeprefix("seccomp=")))


async def test_create_runc_keeps_the_authored_seccomp_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        del timeout_s
        calls.append(list(argv))
        return 0, b"deadbeefcafe\n", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)

    await DockerBackend().create(_spec(seccomp_profile=str(_SECCOMP_SANDBOX)))

    seccomp = next(v for v in _seccomp_values(calls[0]) if v.startswith("seccomp="))
    assert seccomp == f"seccomp={_SECCOMP_SANDBOX}"
