"""Managed-image enumeration has its OWN timeout (aios#2343).

``list_managed_images`` is the one Docker call whose duration grows with the
snapshot pool (35 s for ~760 images on the containerd snapshotter). Under the
general 30 s CLI timeout the GC aborted hourly for 10+ hours exactly when the
pool was largest. The listing and its inspect batches must use
``sandbox_gc_enumeration_timeout_seconds``; every other management call keeps
``sandbox_docker_cli_timeout_seconds``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aios.config import get_settings
from aios.sandbox import _subprocess
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxBackendError
from aios.sandbox.backends.docker import DockerBackend


def _payload(ids: list[str]) -> bytes:
    return json.dumps(
        [
            {"Id": iid, "Parent": "", "Size": 1, "RepoTags": [], "Config": {"Labels": {}}}
            for iid in ids
        ]
    ).encode()


def test_enumeration_timeout_setting_defaults_and_is_distinct() -> None:
    settings = get_settings()
    assert settings.sandbox_gc_enumeration_timeout_seconds == 300.0
    assert settings.sandbox_docker_cli_timeout_seconds == 30.0


@pytest.mark.asyncio
async def test_listing_and_inspect_batches_use_the_enumeration_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIOS_SANDBOX_GC_ENUMERATION_TIMEOUT_SECONDS", "123")
    get_settings.cache_clear()
    ids = [f"sha256:{i:04d}" for i in range(150)]
    seen: list[tuple[str, Any]] = []

    async def fake_run(argv: list[str], **kwargs: Any) -> tuple[int, bytes, bytes]:
        seen.append((argv[1], kwargs.get("timeout_s")))
        if argv[1] == "images":
            return 0, ("\n".join(ids) + "\n").encode(), b""
        return 0, _payload(argv[3:]), b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    images = await DockerBackend().list_managed_images(instance_id="test")

    assert len(images) == 150
    assert seen == [("images", 123.0), ("image", 123.0), ("image", 123.0)]


@pytest.mark.asyncio
async def test_empty_listing_confirmation_uses_the_enumeration_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIOS_SANDBOX_GC_ENUMERATION_TIMEOUT_SECONDS", "77")
    get_settings.cache_clear()
    timeouts: list[Any] = []

    async def fake_run(argv: list[str], **kwargs: Any) -> tuple[int, bytes, bytes]:
        timeouts.append(kwargs.get("timeout_s"))
        return 0, b"", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", fake_run)
    assert await DockerBackend().list_managed_images(instance_id="test") == []
    assert timeouts == [77.0, 77.0]


@pytest.mark.asyncio
async def test_slow_listing_trips_the_enumeration_timeout_not_the_general_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation check: a 1 s enumeration bound fails a slow listing, while the
    general CLI timeout (left generous) is untouched."""
    monkeypatch.setenv("AIOS_SANDBOX_GC_ENUMERATION_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("AIOS_SANDBOX_DOCKER_CLI_TIMEOUT_SECONDS", "30")
    get_settings.cache_clear()
    used: list[float] = []

    async def slow_subprocess(
        argv: list[str], *, timeout_s: float
    ) -> tuple[int, bytes, bytes, bool]:
        used.append(timeout_s)
        try:
            await asyncio.wait_for(asyncio.sleep(5), timeout=timeout_s)
        except TimeoutError:
            return -1, b"", b"", True
        return 0, b"", b"", False

    monkeypatch.setattr(_subprocess, "run_subprocess_with_timeout", slow_subprocess)
    with pytest.raises(SandboxBackendError, match=r"timed out after 1\.0s"):
        await DockerBackend().list_managed_images(instance_id="test")
    assert used == [1.0]
    assert get_settings().sandbox_docker_cli_timeout_seconds == 30.0
