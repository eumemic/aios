"""#390 race: the image a cold-start container runs must be the stamped one.

Simulates a registry whose mutable tag points at ``sha256:AAA`` for the first
pull and ``sha256:BBB`` for every later pull. If ``create()`` stamped A but let
``docker run`` pull again, the container would run B labelled A, and the next
resume would discard its snapshot as ``base_image_updated``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import BASE_IMAGE_ID_LABEL_KEY, Mount, SandboxSpec
from aios.sandbox.backends.docker import DockerBackend

TAG = "ghcr.io/eumemic/aios-sandbox:stable"


async def test_launched_image_is_the_stamped_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pulls = 0
    current: dict[str, str] = {"id": "sha256:UNPULLED"}
    run_argv: list[str] = []

    async def fake_run(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        nonlocal pulls
        del timeout_s, snapshot_timeout
        if argv[1] == "pull":
            pulls += 1
            current["id"] = "sha256:AAA" if pulls == 1 else "sha256:BBB"
            return 0, b"", b""
        if argv[1:3] == ["image", "inspect"]:
            return 0, f"{current['id']}\t1\t1\t{{}}\n".encode(), b""
        if argv[1] == "run":
            run_argv.extend(argv)
            return 0, b"deadbeefcafe\n", b""
        return 0, b"", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    spec = SandboxSpec(
        session_id="sess_race",
        instance_id="inst_race",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=UnrestrictedNetworking(),
        host_gateway_alias=None,
        image=TAG,
    )
    await DockerBackend().create(spec)

    stamped = next(
        a.split("=", 1)[1] for a in run_argv if a.startswith(f"{BASE_IMAGE_ID_LABEL_KEY}=")
    )
    # The image positional is the last token (no trailing command in create()).
    launched = run_argv[-1]
    # A ``--pull always`` on the run itself would re-resolve the tag (→ BBB).
    if "--pull" in run_argv:
        launched_ids = ["sha256:BBB"]
    elif launched == TAG:
        launched_ids = [current["id"]]
    else:
        launched_ids = [launched]
    assert launched_ids == [stamped], (
        f"stamped {stamped} but container launched from {launched_ids}; argv tail={run_argv[-3:]}"
    )
    assert launched == stamped == "sha256:AAA"
