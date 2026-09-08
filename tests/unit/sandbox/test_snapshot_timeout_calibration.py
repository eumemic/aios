"""Unit coverage for ``DockerBackend`` snapshot timeout budgeting.

``_snapshot_timeout_s`` has two live branches:

* the snapshot verb (``docker commit`` / flatten) is budgeted from the
  preceding ``inspect --size`` stat walk (``size_walk_seconds``) — a measured
  per-corpse budget that scales with the daemon/filesystem;
* ``docker image save`` / ``load`` are budgeted from a FIXED per-byte rate
  (``sandbox_snapshot_timeout_ns_per_byte``), clamped to a floor and scaled by
  a safety margin and a per-prior-timeout retry multiplier.

The adaptive-throughput EWMA this module used to maintain (measured from
commit/flatten, persisted to ``sandbox_snapshot_throughput_state_path``, read
on startup) is gone: it was written on every commit/flatten and loaded on every
worker startup yet never fed any timeout — the dead ``throughput_bytes_per_second``
branch of ``_snapshot_timeout_s`` that no caller ever supplied. These tests pin
both live branches AND lock the calibration seam's removal so it can't be
silently reintroduced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.docker import (
    _SNAPSHOT_TIMEOUT_FLOOR_S,
    _SNAPSHOT_WALK_SAFETY_FACTOR,
    DockerBackend,
    _snapshot_timeout_s,
)

# ── fixed-rate branch: save_image / load_image budgets ───────────────────────


def test_fixed_rate_budget_uses_configured_rate_floor_margin_and_retry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``save_image``/``load_image`` timeouts come from a FIXED per-byte rate only
    — there is no adaptive-throughput component. The estimate (size * ns_per_byte)
    is clamped to the floor, scaled by the safety margin, and scaled per prior
    timeout up to the retry cap."""
    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 10.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 100e-9)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 2.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_retry_cap", 4.0)

    # 100 MB * 100e-9 s/B = 10 s; floor 10 s; * margin 2.0 = 20 s; retry 1.0.
    assert _snapshot_timeout_s(100_000_000, retry_attempt=0) == 20.0
    # retry_attempt=1: * retry_multiplier 2.0 (under cap) = 40 s.
    assert _snapshot_timeout_s(100_000_000, retry_attempt=1) == 40.0
    # retry_attempt=9: capped at retry_cap 4.0 = 80 s.
    assert _snapshot_timeout_s(100_000_000, retry_attempt=9) == 80.0


def test_fixed_rate_budget_floors_tiny_and_unknown_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero/None writable layer still gets at least the floor (scaled by the
    safety margin) — never a sub-floor or zero budget for save/load."""
    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 10.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 100e-9)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 2.0)
    assert _snapshot_timeout_s(0) == 20.0
    assert _snapshot_timeout_s(None) == 20.0


# ── walk branch: the snapshot-verb budget ────────────────────────────────────


def test_size_walk_budget_is_independent_of_the_fixed_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot-verb budget is derived from the preceding ``inspect --size``
    stat walk: ``max(floor, walk * walk_safety_factor)``. It does NOT consult the
    per-byte rate, the configured floor, or the safety margin — the walk is the
    only input that scales it past the module floor."""
    settings = get_settings()
    # Make the per-byte path's knobs irrelevant to the walk branch.
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 1.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 1.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 100.0)

    # A walk shorter than the module floor is floored to the module floor.
    assert _snapshot_timeout_s(10**12, size_walk_seconds=1.0) == _SNAPSHOT_TIMEOUT_FLOOR_S
    # A walk past the floor is scaled by the WALK safety factor (10x), not the
    # configured safety margin (100x) and not the per-byte rate.
    assert (
        _snapshot_timeout_s(10**12, size_walk_seconds=30.0) == 30.0 * _SNAPSHOT_WALK_SAFETY_FACTOR
    )


# ── contract: the adaptive-throughput seam has been removed ─────────────────


def test_snapshot_timeout_s_no_longer_accepts_a_throughput_parameter() -> None:
    """The ``throughput_bytes_per_second`` parameter (and the adaptive branch it
    guarded) has been removed. Passing it must be a ``TypeError`` — this prevents
    silently re-introducing the dead seam that a caller could route into without
    the branch ever being reached in production."""
    with pytest.raises(TypeError):
        _snapshot_timeout_s(100_000_000, throughput_bytes_per_second=10.0)  # type: ignore[call-arg]


def test_config_drops_the_persisted_throughput_calibration_fields() -> None:
    """The persisted-calibration config fields are removed alongside the
    machinery that consumed them, so operators can't configure a subsystem that
    no longer exists."""
    settings = get_settings()
    assert not hasattr(settings, "sandbox_snapshot_throughput_ewma_alpha")
    assert not hasattr(settings, "sandbox_snapshot_throughput_state_path")


def test_docker_backend_drops_the_throughput_calibration_machinery() -> None:
    """The EWMA calibration machinery (``_throughput_bytes_per_second``,
    ``_load_throughput``, ``_record_throughput``) is gone. Construction must not
    load a calibration value and must not expose the deleted attributes/methods —
    and must not touch the persisted state file on startup."""
    backend = DockerBackend()
    assert not hasattr(backend, "_throughput_bytes_per_second")
    assert not hasattr(DockerBackend, "_load_throughput")
    assert not hasattr(DockerBackend, "_record_throughput")


# ── the save/load call sites use the fixed-rate budget ──────────────────────


class _CapturingCli:
    """Records the ``run_docker_cli`` calls so a test can assert the timeout a
    save/load call site requested."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float | None]] = []

    async def __call__(
        self, argv: list[str], *, timeout_s: float | None = None, snapshot_timeout: bool = False
    ) -> tuple[int, bytes, bytes]:
        self.calls.append((argv, timeout_s))
        return 0, b"", b""


@pytest.mark.asyncio
async def test_save_image_sizes_its_timeout_from_the_fixed_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``save_image`` (the commit-publish transport) budgets ``docker image save``
    from ``_snapshot_timeout_s(size)`` — the fixed per-byte rate only. The call
    site never supplied (and can no longer supply) an adaptive throughput."""
    settings = get_settings()
    size = 50_000_000
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 10.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 1e-3)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 1.0)

    cli = _CapturingCli()
    monkeypatch.setattr(docker_backend, "run_docker_cli", cli)
    backend = DockerBackend()

    async def _image_size(_image: str) -> int:
        return size

    monkeypatch.setattr(backend, "image_size", _image_size)

    await backend.save_image("img:tag", Path("/tmp/out.tar"))

    save_calls = [(argv, t) for argv, t in cli.calls if argv[1:3] == ["image", "save"]]
    assert len(save_calls) == 1
    _, timeout_s = save_calls[0]
    assert timeout_s == _snapshot_timeout_s(size)


@pytest.mark.asyncio
async def test_load_image_sizes_its_timeout_from_the_fixed_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``load_image`` (the resume transport) budgets ``docker image load`` from
    the on-disk image size via ``_snapshot_timeout_s`` — the fixed per-byte rate
    only, never an adaptive throughput."""
    settings = get_settings()
    size = 50_000
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_floor_seconds", 10.0)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_ns_per_byte", 1e-3)
    monkeypatch.setattr(settings, "sandbox_snapshot_timeout_safety_margin", 1.0)

    cli = _CapturingCli()
    monkeypatch.setattr(docker_backend, "run_docker_cli", cli)
    backend = DockerBackend()

    image_path = tmp_path / "img.tar"
    image_path.write_bytes(b"\0" * size)
    await backend.load_image(image_path)

    load_calls = [(argv, t) for argv, t in cli.calls if argv[1:3] == ["image", "load"]]
    assert len(load_calls) == 1
    _, timeout_s = load_calls[0]
    assert timeout_s == _snapshot_timeout_s(size)
