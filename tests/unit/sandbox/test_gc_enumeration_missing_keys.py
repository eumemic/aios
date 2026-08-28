"""Managed-image enumeration must survive images that omit inspect keys.

Docker's ``inspect --format`` runs under ``missingkey=error``: naming a key an
image does not carry aborts the template FOR THAT IMAGE, which silently drops
it from stdout. ``list_managed_images`` then cannot account for the id, and
its COMPLETE-OR-RAISE contract correctly refuses — so one such image fails the
whole enumeration, which fails the GC tick, hourly, for as long as it exists.

That is how it presented in production: a single image built by ``docker
import`` (every flattened snapshot; they carry no ``Parent`` key at all under
the containerd image store) among 457 managed images stopped ALL reclamation
while superseded residue accumulated. The single-id re-probe used a different
format that never touched ``.Parent``, so it succeeded — which is why the
failure surfaced as the maximally confusing "omitted by the batch inspect but
still exists".

The fix reads the inspect JSON instead of templating it, so an absent key is
``None`` rather than an error. These tests pin the class, not just the one
key: any field may be missing, null, or the wrong type, and enumeration must
still return a usable record for every id.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import pytest

from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend

_LIST_ARGV_HEAD = "images"


def _fake_cli(
    listing: list[str], records: list[dict[str, object]], *, rc: int = 0
) -> Callable[..., Awaitable[tuple[int, bytes, bytes]]]:
    """Fake ``run_docker_cli`` serving one listing and one inspect batch."""

    async def run(argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
        if argv[1] == _LIST_ARGV_HEAD:
            return 0, ("\n".join(listing) + "\n").encode(), b""
        if "--format" in argv:  # the single-id verified-negative probe
            return 1, b"", b"Error: No such image: " + argv[-1].encode()
        return rc, json.dumps(records).encode(), b""

    return run


@pytest.mark.asyncio
async def test_image_without_a_parent_key_does_not_fail_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE production regression. A ``docker import`` image has no ``Parent``.

    Before the fix this raised ``incomplete managed image enumeration`` and
    took every GC tick down with it.
    """
    flattened = {
        "Id": "sha256:flat",
        # NOTE: no "Parent" key at all — not empty, absent.
        "Size": 6_377_787_303,
        "RepoTags": ["aios-sbx-default-sess_x:latest"],
        "Config": {"Labels": {"aios.managed": "true", "aios.flattened": "true"}},
    }
    monkeypatch.setattr(docker_backend, "run_docker_cli", _fake_cli(["sha256:flat"], [flattened]))

    images = await DockerBackend().list_managed_images(instance_id="default")

    assert len(images) == 1
    assert images[0].image_id == "sha256:flat"
    assert images[0].parent_id is None, "a flattened image genuinely has no parent"
    assert images[0].size_bytes == 6_377_787_303
    assert images[0].labels["aios.flattened"] == "true"


@pytest.mark.asyncio
async def test_one_odd_image_does_not_poison_the_rest_of_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blast radius that made this so costly: 1 bad image in 457 stopped
    reclamation for all of them."""
    records: list[dict[str, object]] = [
        {
            "Id": f"sha256:{i:04d}",
            "Parent": "",
            "Size": 10,
            "RepoTags": [f"tag-{i}"],
            "Config": {"Labels": {}},
        }
        for i in range(20)
    ]
    records.append({"Id": "sha256:flat", "Size": 99, "RepoTags": [], "Config": None})
    listing = [str(r["Id"]) for r in records]
    monkeypatch.setattr(docker_backend, "run_docker_cli", _fake_cli(listing, records))

    images = await DockerBackend().list_managed_images(instance_id="default")

    assert len(images) == 21
    assert {img.image_id for img in images} == set(listing)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [
        {"Id": "sha256:a"},  # everything but the id missing
        {"Id": "sha256:a", "Parent": None, "Size": None, "RepoTags": None, "Config": None},
        {"Id": "sha256:a", "Size": "not-an-int", "RepoTags": "not-a-list", "Config": 7},
        {"Id": "sha256:a", "RepoTags": ["ok", 5, None], "Config": {"Labels": None}},
    ],
    ids=["only-id", "all-null", "wrong-types", "mixed-tag-types"],
)
async def test_degenerate_records_still_yield_a_usable_entry(
    monkeypatch: pytest.MonkeyPatch, record: dict[str, object]
) -> None:
    """Enumeration must not raise on any shape docker might emit: refusing is
    how the tick dies, and the tick dying is the whole failure mode."""
    monkeypatch.setattr(docker_backend, "run_docker_cli", _fake_cli(["sha256:a"], [record]))

    images = await DockerBackend().list_managed_images(instance_id="default")

    assert len(images) == 1
    assert images[0].image_id == "sha256:a"
    assert isinstance(images[0].size_bytes, int)
    assert isinstance(images[0].labels, dict)
    assert isinstance(images[0].repo_tags, tuple)


@pytest.mark.asyncio
async def test_unparseable_payload_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The COMPLETE-OR-RAISE contract is preserved, not traded away.

    Output we cannot parse is INDETERMINATE, not empty — reading it as empty
    would report every id in the batch as absent, and the GC reconciles DB
    pointers by absence (aios#2138).
    """

    async def run(argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
        if argv[1] == _LIST_ARGV_HEAD:
            return 0, b"sha256:a\n", b""
        return 0, b"this is not json", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", run)

    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await DockerBackend().list_managed_images(instance_id="default")


@pytest.mark.asyncio
async def test_non_list_payload_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
        if argv[1] == _LIST_ARGV_HEAD:
            return 0, b"sha256:a\n", b""
        return 0, b'{"Id": "sha256:a"}', b""  # object, not array

    monkeypatch.setattr(docker_backend, "run_docker_cli", run)

    with pytest.raises(SandboxBackendError, match="incomplete managed image enumeration"):
        await DockerBackend().list_managed_images(instance_id="default")


@pytest.mark.asyncio
async def test_batch_no_longer_passes_a_format_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the fix itself: reintroducing ``--format`` on the BATCH call
    reintroduces the missingkey class. The single-id probe may still use one."""
    seen: list[list[str]] = []

    async def run(argv: list[str], **_kw: object) -> tuple[int, bytes, bytes]:
        seen.append(argv)
        if argv[1] == _LIST_ARGV_HEAD:
            return 0, b"sha256:a\n", b""
        return 0, json.dumps([{"Id": "sha256:a", "Config": {"Labels": {}}}]).encode(), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", run)
    await DockerBackend().list_managed_images(instance_id="default")

    batch_calls = [a for a in seen if a[1:3] == ["image", "inspect"]]
    assert batch_calls, "the batch inspect must have run"
    for argv in batch_calls:
        assert "--format" not in argv
