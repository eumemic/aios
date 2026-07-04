"""E2E: the durable-session-sandbox persistence contract on a REAL daemon (§10).

Exercises the snapshot verb + resume against the production-shaped flow
(``backend.snapshot`` → ``backend.create`` from the resulting tag) so the
containerd-store mechanics the design depends on are validated end to end:

  * the writable rootfs (``/root``, ``/etc``, ``/tmp``, global installs)
    survives a release/resume cycle; ``/dev/shm`` (tmpfs) and processes do NOT;
  * bind mounts are excluded from the snapshot;
  * a no-write release is ``skipped_empty`` (the containerd SizeRw floor);
  * run-injected secret VALUES are scrubbed from the committed image config;
  * resume integrity (non-empty PATH, HOME, pwd) holds after a plain commit
    AND after a flatten round-trip.

These tests run real ``docker`` and pull/run the sandbox image; they are
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
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.spec import snapshot_tag
from tests.conftest import needs_docker
from tests.helpers.sandbox import run_sandbox

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SECRET_VALUE = "s3cr3t-do-not-leak-DEADBEEF"
SECCOMP_PROFILE = str(Path(__file__).parents[2] / "docker" / "seccomp-sandbox.json")


def _spec(
    *,
    instance_id: str,
    session_id: str,
    workspace: Path,
    snapshot_image: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxSpec:
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", **(env or {})}
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


async def _write_substantial(backend: DockerBackend, handle: SandboxHandle) -> None:
    """Write well over the empty-floor (8 KiB) so the release commits a real
    layer — a no-write / sub-floor session is intentionally ``skipped_empty``
    (the identity short-circuit), which is covered by its own test."""
    await run_sandbox(backend, handle, "head -c 65536 /dev/zero > /root/blob")


async def test_filesystem_persists_processes_and_shm_do_not(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    # Cycle 1: write markers across the persistent rootfs + tmpfs, start a proc.
    h1 = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    await run_sandbox(backend, h1, "echo root > /root/marker")
    await run_sandbox(backend, h1, "echo etc > /etc/marker")
    await run_sandbox(backend, h1, "echo tmp > /tmp/marker")
    await run_sandbox(
        backend, h1, "mkdir -p /usr/local/persist && echo glob > /usr/local/persist/marker"
    )
    await run_sandbox(backend, h1, "echo shm > /dev/shm/marker")
    await run_sandbox(backend, h1, "nohup sleep 1000 >/dev/null 2>&1 &")
    await run_sandbox(
        backend, h1, "echo workspace-only > /workspace/in_ws"
    )  # bind mount — excluded
    await _write_substantial(backend, h1)

    out = await backend.snapshot(
        h1.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
    )
    assert out.kind == "committed", f"expected a committed snapshot, got {out.kind}"
    await backend.destroy(h1)
    assert await backend.image_labels(tag) is not None, "snapshot tag must exist after commit"

    # Cycle 2: resume from the snapshot on a FRESH workspace dir.
    fresh_ws = workspace.parent / "ws2"
    fresh_ws.mkdir()
    h2 = await backend.create(
        _spec(
            instance_id=instance_id, session_id=session_id, workspace=fresh_ws, snapshot_image=tag
        )
    )
    try:
        # Persistent rootfs survives.
        for path in ("/root/marker", "/etc/marker", "/tmp/marker", "/usr/local/persist/marker"):
            rc, _ = await run_sandbox(backend, h2, f"test -f {path}")
            assert rc == 0, f"{path} did not survive the snapshot/resume cycle"
        # /dev/shm (tmpfs) does NOT survive.
        rc, _ = await run_sandbox(backend, h2, "test -f /dev/shm/marker")
        assert rc != 0, "/dev/shm marker survived — tmpfs must not persist"
        # The process is dead (resume = fresh process tree).
        rc, out_txt = await run_sandbox(backend, h2, "pgrep -x sleep || echo NONE")
        assert "NONE" in out_txt, "the background process survived — resume must be a fresh boot"
        # Bind-mounted /workspace is excluded from the snapshot (fresh dir is empty).
        rc, _ = await run_sandbox(backend, h2, "test -f /workspace/in_ws")
        assert rc != 0, "bind-mount content leaked into the snapshot"

        # Second cycle: re-snapshot → resume → first-cycle markers still survive.
        out2 = await backend.snapshot(
            h2.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
        )
        assert out2.kind in ("committed", "skipped_empty")
    finally:
        await backend.destroy(h2)

    h3 = await backend.create(
        _spec(
            instance_id=instance_id, session_id=session_id, workspace=fresh_ws, snapshot_image=tag
        )
    )
    try:
        rc, _ = await run_sandbox(backend, h3, "test -f /root/marker")
        assert rc == 0, "/root/marker lost across two snapshot cycles"
    finally:
        await backend.destroy(h3)


async def test_zero_write_release_is_skipped_empty(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    """A read/chat-only session (no writes) snapshots as ``skipped_empty`` — the
    containerd-store SizeRw floor (a no-write container reports 4096, not 0)."""
    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    h1 = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    await run_sandbox(backend, h1, "true")  # no filesystem writes
    out = await backend.snapshot(
        h1.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
    )
    await backend.destroy(h1)
    assert out.kind == "skipped_empty", (
        f"a no-write release must be skipped_empty on the floor, got {out.kind}"
    )


async def test_secret_value_absent_from_committed_image_config(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    """Run-injected secret VALUES are scrubbed from the committed image config
    (a ``docker save`` / image-inspect exposure); the resumed container still
    gets the secret via ``docker run --env`` at runtime."""
    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    h1 = await backend.create(
        _spec(
            instance_id=instance_id,
            session_id=session_id,
            workspace=workspace,
            env={"OPENAI_API_KEY": SECRET_VALUE},
        )
    )
    await _write_substantial(backend, h1)  # force a non-empty committed layer
    await backend.snapshot(
        h1.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
    )
    await backend.destroy(h1)

    # Inspect the committed image's baked Config.Env — the secret value must be gone.
    rc, out, err = await _docker(["image", "inspect", "--format", "{{json .Config.Env}}", tag])
    assert rc == 0, err
    assert SECRET_VALUE not in out, "the secret VALUE leaked into the committed image config"


@pytest.mark.parametrize("flatten", [False, True], ids=["plain_commit", "flatten"])
async def test_resume_integrity_path_home_pwd(
    daemon: tuple[DockerBackend, str, str, Path], flatten: bool
) -> None:
    """A resumed container has a non-empty PATH and pwd ==/workspace, and a
    correct HOME — both after a plain commit AND after a flatten round-trip
    (the historical container-bricker + the config-strip regression).

    A plain commit PRESERVES the base image's HOME (so the resumed value equals
    a fresh base container's HOME); flatten strips all config and RESTORES the
    sandbox's canonical ``/root``. (In a current deployment the base sets
    HOME=/root, so both are /root; capturing the base HOME keeps the test
    robust to base-image build variance.)

    Either way the resumed container must satisfy the ownership invariant
    ownership-checking tools enforce -- ``stat -c %u "$HOME" == id -u`` -- so a
    HOME/uid regression that only manifests post-thaw (e.g. ``_flatten``
    restoring a foreign-owned HOME) fails here."""
    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    # The base image's HOME, as the daemon yields it for a fresh container.
    base_h = await backend.create(
        _spec(instance_id=instance_id, session_id=f"{session_id}base", workspace=workspace)
    )
    try:
        _rc, base_home = await run_sandbox(backend, base_h, 'printf %s "$HOME"')
        base_home = base_home.strip()
    finally:
        await backend.destroy(base_h)

    h1 = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    await run_sandbox(backend, h1, "echo x > /root/marker")
    await _write_substantial(backend, h1)
    # flatten_if_unique_bytes_over=0 forces the flatten path on any nonzero layer.
    await backend.snapshot(
        h1.sandbox_id,
        tag,
        empty_floor_bytes=8192,
        flatten_if_unique_bytes_over=0 if flatten else None,
    )
    await backend.destroy(h1)

    if flatten:
        # The flattened image's baked CMD is the keepalive set by ``_flatten``'s
        # ``--change`` — assert docker's actual ``--change`` parse produced the
        # absolute-path form (#925/#938), not just that we emitted the string.
        rc, out, err = await _docker(["image", "inspect", "--format", "{{json .Config.Cmd}}", tag])
        assert rc == 0, err
        assert out.strip() == '["/usr/bin/tail","-f","/dev/null"]', (
            f"flattened image CMD not the absolute-path keepalive: {out.strip()!r}"
        )

    h2 = await backend.create(
        _spec(
            instance_id=instance_id, session_id=session_id, workspace=workspace, snapshot_image=tag
        )
    )
    try:
        rc, path = await run_sandbox(backend, h2, 'printf %s "$PATH"')
        assert rc == 0 and path.strip(), "resumed container has an empty PATH (CMD would not exec)"
        rc, home = await run_sandbox(backend, h2, 'printf %s "$HOME"')
        expected_home = "/root" if flatten else base_home
        assert home.strip() == expected_home, (
            f"HOME mismatch: got {home!r}, expected {expected_home!r} (flatten={flatten})"
        )
        # The ownership invariant ownership-checking tools enforce: $HOME's
        # owner uid == the running uid, surviving the (possibly flattened)
        # round-trip.
        rc, owner_ok = await run_sandbox(
            backend, h2, 'test "$(stat -c %u "$HOME")" = "$(id -u)" && echo ok'
        )
        assert rc == 0 and owner_ok.strip() == "ok", (
            f"$HOME owner != running uid after resume (flatten={flatten}): {owner_ok!r}"
        )
        rc, pwd = await run_sandbox(backend, h2, "pwd")
        assert pwd.strip() == "/workspace", f"pwd not /workspace: {pwd!r}"
        # The marker survived the (possibly flattened) round-trip.
        rc, _ = await run_sandbox(backend, h2, "test -f /root/marker")
        assert rc == 0
    finally:
        await backend.destroy(h2)


async def test_lockdown_survives_poisoned_iptables_on_resume(
    daemon: tuple[DockerBackend, str, str, Path],
) -> None:
    """The §5.8 security fix on a real daemon: a tenant poisons the sandbox's
    own ``iptables`` (replaces it with ``exit 0``), the snapshot persists it,
    and on resume the Limited lockdown STILL applies + read-back verifies —
    because it runs from the operator-image sidecar, not the tenant FS — and the
    sandbox holds no ``NET_ADMIN`` to flush it.
    """
    from aios.models.environments import LimitedNetworking
    from aios.sandbox.setup import apply_network_lockdown

    backend, instance_id, session_id, workspace = daemon
    tag = snapshot_tag(instance_id, session_id)

    # Cycle 1: poison the in-sandbox iptables (the persisted-bypass vector).
    h1 = await backend.create(
        _spec(instance_id=instance_id, session_id=session_id, workspace=workspace)
    )
    await run_sandbox(
        backend,
        h1,
        "printf '#!/bin/sh\\nexit 0\\n' > /usr/sbin/iptables && chmod +x /usr/sbin/iptables",
    )
    await _write_substantial(backend, h1)
    await backend.snapshot(
        h1.sandbox_id, tag, empty_floor_bytes=8192, flatten_if_unique_bytes_over=None
    )
    await backend.destroy(h1)

    # Cycle 2: resume from the poisoned snapshot.
    h2 = await backend.create(
        _spec(
            instance_id=instance_id, session_id=session_id, workspace=workspace, snapshot_image=tag
        )
    )
    try:
        # Sanity: the sandbox's own iptables is the poisoned no-op.
        rc, _ = await run_sandbox(backend, h2, "test -x /usr/sbin/iptables")
        assert rc == 0

        # Apply the Limited lockdown. With the OLD in-sandbox approach this
        # would trust the poisoned ``exit 0`` and leave egress open; the sidecar
        # applies + read-back verifies REAL rules, so this returns (no raise)
        # only because the DROP policy actually landed.
        await apply_network_lockdown(
            backend,
            h2,
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
        )

        # The sandbox holds no NET_ADMIN (capability bit 12) — it cannot touch
        # netfilter to flush its own lockdown.
        rc, capeff = await run_sandbox(
            backend, h2, "grep CapEff /proc/self/status | awk '{print $2}'"
        )
        caps = int(capeff.strip(), 16)
        assert not (caps >> 12) & 1, f"sandbox must not hold NET_ADMIN; CapEff={capeff.strip()}"
    finally:
        await backend.destroy(h2)


async def _docker(args: list[str]) -> tuple[int, str, str]:
    from aios.sandbox._subprocess import run_docker_cli

    rc, out, err = await run_docker_cli(["docker", *args])
    return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
