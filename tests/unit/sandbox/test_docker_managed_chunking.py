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


def _managed_line(iid: str) -> str:
    return f'{iid}\t\t1\t["tag-{iid}"]\t{json.dumps({"Labels": {}})}'


def _is_single_probe(argv: list[str]) -> bool:
    """The verified-negative probe: ``_inspect_image_fields``'s RootFS format."""
    return argv[1:3] == ["image", "inspect"] and "RootFS" in argv[4]


@pytest.mark.asyncio
async def test_list_managed_images_fails_closed_when_a_batch_silently_omits_a_live_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short/truncated inspect read is UNKNOWN, not absence.

    The batch exits 0 but its output is missing one id, and that image still
    inspects fine. Omitting it would hand the GC a list whose absence means
    "deleted" — the exact aios#2138 inversion, reached without any batch error.
    """
    ids = [f"sha256:{i:04d}" for i in range(3)]
    dropped = ids[1]

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        if _is_single_probe(argv):
            # The omitted image is alive and well.
            return 0, f"{argv[5]}\t1\t1\t{json.dumps({'Labels': {}})}\n".encode(), b""
        lines = [_managed_line(iid) for iid in argv[5:] if iid != dropped]
        return 0, ("\n".join(lines) + "\n").encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await DockerBackend().list_managed_images(instance_id="test")


@pytest.mark.asyncio
async def test_list_managed_images_omits_only_an_image_proved_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opposite failure: failing closed on a genuine vanish never collects.

    ``docker image inspect`` exits nonzero when ANY id in the batch is gone —
    the routine race with a concurrent removal. A verified-negative probe
    ("No such image") proves absence, so the tick proceeds without it.
    """
    ids = [f"sha256:{i:04d}" for i in range(3)]
    vanished = ids[1]

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        if _is_single_probe(argv):
            return 1, b"", b"Error: No such image: " + argv[5].encode()
        lines = [_managed_line(iid) for iid in argv[5:] if iid != vanished]
        return 1, ("\n".join(lines) + "\n").encode(), b"Error: No such image: " + vanished.encode()

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    images = await DockerBackend().list_managed_images(instance_id="test")

    assert {image.image_id for image in images} == {ids[0], ids[2]}


@pytest.mark.asyncio
async def test_gc_does_not_reconcile_absence_from_a_silently_truncated_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a short read must not drive pointer clearing via ``_gc_once``."""
    ids = [f"sha256:{i:04d}" for i in range(3)]
    dropped = ids[1]

    async def fake_run(argv: list[str], **kwargs: object) -> tuple[int, bytes, bytes]:
        if argv[1] == "ps":
            return 0, b"", b""
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        if _is_single_probe(argv):
            return 0, f"{argv[5]}\t1\t1\t{json.dumps({'Labels': {}})}\n".encode(), b""
        lines = [_managed_line(iid) for iid in argv[5:] if iid != dropped]
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
