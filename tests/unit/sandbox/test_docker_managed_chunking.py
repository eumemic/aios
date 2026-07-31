from __future__ import annotations

import json

import pytest

from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend


@pytest.mark.asyncio
async def test_list_managed_images_chunks_large_population_and_skips_failed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"sha256:{i:04d}" for i in range(2050)]
    inspect_batches: list[list[str]] = []

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        batch = argv[5:]
        inspect_batches.append(batch)
        if len(inspect_batches) == 2:
            raise SandboxBackendError("timed out")
        lines = [
            f'{iid}\t\t1\t["tag-{iid}"]\t{json.dumps({"Labels": {"aios.session_id": iid}})}'
            for iid in batch
        ]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    images = await DockerBackend().list_managed_images(instance_id="test")

    assert [len(batch) for batch in inspect_batches] == [100] * 20 + [50]
    assert len(images) == 1950
    assert ids[100] not in {image.image_id for image in images}


@pytest.mark.asyncio
async def test_list_managed_containers_chunks_and_continues_after_nonzero_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"cid-{i:04d}" for i in range(250)]
    inspect_batches: list[list[str]] = []

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        if argv[1] == "ps":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        batch = argv[4:]
        inspect_batches.append(batch)
        if len(inspect_batches) == 2:
            return 1, b"", b"daemon busy"
        return 0, ("\n".join(f"{cid}\tfalse\tsess" for cid in batch) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    refs = await DockerBackend().list_managed(instance_id="test")

    assert [len(batch) for batch in inspect_batches] == [100, 100, 50]
    assert len(refs) == 150
