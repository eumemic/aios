from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aios.config import get_settings
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.docker import DockerBackend, _snapshot_timeout_s


def test_timeout_budget_uses_configured_floor_throughput_and_retry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 10.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 100e-9)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 2.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_retry_cap", 4.0)

    assert (
        _snapshot_timeout_s(100_000_000, throughput_bytes_per_second=None, retry_attempt=0) == 20.0
    )
    assert (
        _snapshot_timeout_s(100_000_000, throughput_bytes_per_second=None, retry_attempt=1) == 40.0
    )
    assert (
        _snapshot_timeout_s(100_000_000, throughput_bytes_per_second=None, retry_attempt=9) == 80.0
    )


@pytest.mark.asyncio
async def test_successful_commit_updates_and_persists_throughput(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = get_settings()
    state_path = tmp_path / "throughput.json"
    monkeypatch.setattr(settings, "sandbox_snapshot_throughput_state_path", state_path)
    monkeypatch.setattr(settings, "sandbox_snapshot_throughput_ewma_alpha", 0.5)
    times = iter((10.0, 12.0))
    monkeypatch.setattr(docker_backend, "monotonic", lambda: next(times))

    async def cli(
        argv: list[str], *, timeout_s: float = 30.0, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        return 0, b"", b""

    monkeypatch.setattr(docker_backend, "run_docker_cli", cli)
    backend = DockerBackend()
    backend._throughput_bytes_per_second = 50.0
    monkeypatch.setattr(
        backend, "_inspect_image_fields", lambda tag: _async_value(("id", 100, 2, {}))
    )
    monkeypatch.setattr(backend, "_unique_bytes", lambda fields, base: _async_value(100))

    await backend._commit("cid", "tag", [], None, timeout_s=60.0, size_rw=200)

    assert backend._throughput_bytes_per_second == 75.0
    assert '"bytes_per_second": 75.0' in state_path.read_text()


async def _async_value(value: Any) -> Any:
    return value
