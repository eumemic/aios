from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.registry import GcPressureResult, SandboxRegistry


@pytest.mark.asyncio
async def test_list_managed_images_fails_closed_when_any_inspect_batch_fails(
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
    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await DockerBackend().list_managed_images(instance_id="test")

    assert [len(batch) for batch in inspect_batches] == [100, 100]


@pytest.mark.asyncio
async def test_gc_does_not_reconcile_absence_from_failed_image_inspect_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable image batch is unknown, never evidence that its tags vanished."""
    ids = [f"sha256:{i:04d}" for i in range(101)]
    inspect_calls = 0

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        nonlocal inspect_calls
        if argv[1] == "ps":
            return 0, b"", b""
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        inspect_calls += 1
        if inspect_calls == 2:
            raise SandboxBackendError("timed out")
        batch = argv[5:]
        lines = [f'{iid}\t\t1\t["tag-{iid}"]\t{{"Labels": {{}}}}' for iid in batch]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    registry = SandboxRegistry(DockerBackend())
    reconcile = AsyncMock()
    registry._gc_reconcile_absent_pointers = reconcile  # type: ignore[method-assign]
    registry._load_gc_states = AsyncMock(return_value={})  # type: ignore[method-assign]
    registry._gc_corpse_pass = AsyncMock()  # type: ignore[method-assign]
    registry._gc_image_pass = AsyncMock(return_value=[])  # type: ignore[method-assign]
    registry._gc_canonical_store_pass = AsyncMock()  # type: ignore[method-assign]
    registry._gc_pool_budget_pass = AsyncMock(  # type: ignore[method-assign]
        return_value=GcPressureResult()
    )
    registry._gc_account_cap_pass = AsyncMock(  # type: ignore[method-assign]
        return_value=GcPressureResult()
    )
    registry._gc_reconcile_pointers = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await registry._gc_once(AsyncMock())

    reconcile.assert_not_awaited()


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
