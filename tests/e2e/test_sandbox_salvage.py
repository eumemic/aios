"""E2E: crash-corpse salvage on a real daemon (durable session sandboxes, §5.4).

The headline guarantee — *container death stops losing data* — is validated by
leaving a stopped corpse (the no-``--rm`` path) and driving the registry's
salvage preamble (``_salvage_session_corpses``) directly: it commits the corpse
to the session's snapshot tag and removes it, so a subsequent resume recovers
the pre-crash filesystem. The DB pointer write is stubbed (a fake pool) — pointer
discipline is unit-covered; this exercises the real stop→commit→rm integration.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aios.config import get_settings
from aios.harness import runtime
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
from aios.sandbox.network import ensure_sandbox_network
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.spec import snapshot_tag
from tests.conftest import needs_docker
from tests.helpers.sandbox import FakePool

pytestmark = [needs_docker, pytest.mark.docker]

IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")
SECCOMP_PROFILE = str(Path(__file__).parents[2] / "docker" / "seccomp-sandbox.json")


@pytest.fixture
def fake_pool() -> Iterator[None]:
    prev = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        yield
    finally:
        runtime.pool = prev


@pytest.fixture
async def daemon(tmp_path: Path) -> AsyncIterator[tuple[DockerBackend, str, Path]]:
    await ensure_sandbox_network()
    backend = DockerBackend()
    instance_id = get_settings().instance_id
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    try:
        yield backend, session_id, workspace
    finally:
        for ref in await backend.list_managed(instance_id=instance_id, session_id=session_id):
            await backend.force_remove(ref.sandbox_id)
        await backend.remove_image(snapshot_tag(instance_id, session_id))


def _spec(session_id: str, workspace: Path) -> SandboxSpec:
    instance_id = get_settings().instance_id
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
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
        seccomp_profile=SECCOMP_PROFILE,
    )


async def test_crash_corpse_is_salvaged_then_resumed(
    daemon: tuple[DockerBackend, str, Path], fake_pool: None
) -> None:
    backend, session_id, workspace = daemon
    instance_id = get_settings().instance_id
    tag = snapshot_tag(instance_id, session_id)
    registry = SandboxRegistry(backend=backend)

    # Provision a container, write a marker + a substantial layer, then simulate
    # a crash: stop it out of band WITHOUT removing it (the no-``--rm`` corpse).
    handle = await backend.create(_spec(session_id, workspace))
    await backend.exec(
        handle,
        "echo pre-crash > /root/survivor && head -c 65536 /dev/zero > /root/blob",
        timeout_seconds=60,
        max_output_bytes=10_000,
    )
    await backend.stop(handle.sandbox_id)  # crash: corpse remains, no snapshot yet
    assert await backend.image_labels(tag) is None, "no snapshot should exist before salvage"

    # The salvage preamble commits the corpse to the snapshot tag and removes it.
    await registry._salvage_session_corpses(session_id)
    assert await backend.image_labels(tag) is not None, "salvage must commit the corpse's FS"
    assert not await backend.list_managed(instance_id=instance_id, session_id=session_id), (
        "salvage must remove the corpse after committing it"
    )

    # Resume from the salvaged snapshot → the pre-crash marker survives.
    resumed = await backend.create(
        dataclasses.replace(_spec(session_id, workspace), snapshot_image=tag)
    )
    try:
        res = await backend.exec(
            resumed, "cat /root/survivor", timeout_seconds=60, max_output_bytes=10_000
        )
        assert "pre-crash" in res.stdout, "the pre-crash marker did not survive salvage+resume"
    finally:
        await backend.destroy(resumed)


async def test_running_corpse_is_salvaged_then_resumed(
    daemon: tuple[DockerBackend, str, Path], fake_pool: None
) -> None:
    """The one post-kill state only a real SIGKILL produces (issue #1757).

    Graceful worker shutdown STOPs every managed container (``worker.py``
    shutdown: "STOP (don't destroy)") — the corpse the test above constructs.
    A SIGKILL takes the whole process tree down without running that shutdown
    path at all, so it leaves the container **RUNNING** (Docker's daemon is a
    separate process; nothing tells it to stop the container). The salvage
    preamble must handle a running corpse identically to a stopped one — its
    own ``snapshot()`` call ``stop()``s the corpse first (idempotent either
    way) — so this pins that behavior against the real daemon rather than
    asserting it only by reading the ``stop()``-is-idempotent code path.

    Also carries an in-flight ``docker exec`` (a background bash job still
    running inside the container when the kill lands) to prove salvage
    doesn't need the corpse quiescent: ``docker stop`` sends SIGTERM then
    SIGKILL to the container's init after its grace period, which reaps the
    exec'd process along with everything else.
    """
    backend, session_id, workspace = daemon
    instance_id = get_settings().instance_id
    tag = snapshot_tag(instance_id, session_id)
    registry = SandboxRegistry(backend=backend)

    handle = await backend.create(_spec(session_id, workspace))
    await backend.exec(
        handle,
        "echo pre-crash > /root/survivor && head -c 65536 /dev/zero > /root/blob",
        timeout_seconds=60,
        max_output_bytes=10_000,
    )
    # Launch a detached in-flight background job — still running when the
    # (simulated) kill lands. Fire-and-forget: this call's own ``docker exec``
    # returns immediately (``sleep 300 &`` backgrounds inside the container),
    # so it does not itself model the crash — only leaves the corpse holding
    # a live child process, unlike the graceful-shutdown corpse above.
    await backend.exec(
        handle,
        "nohup sleep 300 >/dev/null 2>&1 & disown",
        timeout_seconds=30,
        max_output_bytes=1_000,
    )

    # Simulate a real SIGKILL: unlike the graceful-shutdown path (which STOPs
    # every managed container before the process exits), NOTHING stops this
    # container — it is left exactly as a real `kill -9` on the worker
    # process would leave it: RUNNING, with its background job still alive.
    refs_before = await backend.list_managed(instance_id=instance_id, session_id=session_id)
    assert len(refs_before) == 1
    assert refs_before[0].running is True, "the corpse must still be RUNNING (no graceful stop)"
    assert await backend.image_labels(tag) is None, "no snapshot should exist before salvage"

    # The salvage preamble commits the RUNNING corpse to the snapshot tag and
    # removes it — same call as the stopped-corpse case, no special-casing.
    await registry._salvage_session_corpses(session_id)
    assert await backend.image_labels(tag) is not None, "salvage must commit the corpse's FS"
    assert not await backend.list_managed(instance_id=instance_id, session_id=session_id), (
        "salvage must remove the corpse after committing it"
    )

    # Resume from the salvaged snapshot → the pre-crash marker survives, and
    # the background job is gone (a fresh container, not the killed one).
    resumed = await backend.create(
        dataclasses.replace(_spec(session_id, workspace), snapshot_image=tag)
    )
    try:
        res = await backend.exec(
            resumed, "cat /root/survivor", timeout_seconds=60, max_output_bytes=10_000
        )
        assert "pre-crash" in res.stdout, "the pre-crash marker did not survive salvage+resume"
    finally:
        await backend.destroy(resumed)
