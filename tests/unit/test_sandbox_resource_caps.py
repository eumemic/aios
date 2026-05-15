"""Unit tests for the per-sandbox resource caps (#367 PR 9).

Snoops on the ``argv`` the docker backend builds to verify each cap
flips its corresponding ``docker run`` flag (cpu / memory + memory-swap /
pids-limit) on, and the absence of each cap leaves the corresponding
flag off so we don't accidentally pin the host's default behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aios.sandbox.backends.base import Mount, SandboxSpec, Unrestricted
from aios.sandbox.backends.docker import DockerBackend


def _spec(**overrides: object) -> SandboxSpec:
    """Build a minimal SandboxSpec with the resource caps under test."""
    base: dict[str, object] = {
        "session_id": "sess_x",
        "instance_id": "inst_test",
        "workspace": Mount(
            host_path=Path("/tmp/ws"),
            sandbox_path="/workspace",
            read_only=False,
        ),
        "extra_mounts": (),
        "environment": {},
        "labels": {},
        "network_policy": Unrestricted(),
        "host_gateway_alias": None,
        "image": "aios-sandbox:test",
    }
    base.update(overrides)
    return SandboxSpec(**base)  # type: ignore[arg-type]


async def _capture_argv(spec: SandboxSpec) -> list[str]:
    """Run DockerBackend.create against a stubbed ``docker`` subprocess.

    Returns the argv the backend would have invoked.
    """
    captured: dict[str, list[str]] = {}

    async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
        captured["argv"] = argv
        return 0, b"deadbeef1234\n", b""

    with patch("aios.sandbox.backends.docker.run_docker_cli", side_effect=fake_run_docker):
        await DockerBackend().create(spec)
    return captured["argv"]


class TestResourceCapFlags:
    @pytest.mark.asyncio
    async def test_cpu_quota_emits_cpus_flag(self) -> None:
        argv = await _capture_argv(_spec(cpu_quota=1.5))
        assert "--cpus" in argv
        i = argv.index("--cpus")
        assert argv[i + 1] == "1.5"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("quota", "expected"),
        [
            (2.0, "2"),  # whole core → no trailing ".0"
            (0.5, "0.5"),
            (0.01, "0.01"),  # settings floor — must stay plain-decimal
            (0.25, "0.25"),
        ],
    )
    async def test_cpu_quota_format_is_plain_decimal(self, quota: float, expected: str) -> None:
        """Docker's ``--cpus`` parser rejects scientific notation, and
        the previous ``:g`` format flips to ``1e-05`` around 0.0001.
        Locks the formatter at a fixed-decimal-strip-trailing-zeros
        shape so future drift can't reintroduce scientific notation.
        """
        argv = await _capture_argv(_spec(cpu_quota=quota))
        i = argv.index("--cpus")
        assert argv[i + 1] == expected

    @pytest.mark.asyncio
    async def test_memory_bytes_emits_memory_and_memory_swap(self) -> None:
        argv = await _capture_argv(_spec(memory_bytes=256 * 1024 * 1024))
        assert "--memory" in argv
        i = argv.index("--memory")
        assert argv[i + 1] == str(256 * 1024 * 1024)
        # ``--memory-swap`` is pinned to the same value so the sandbox
        # can't lean on swap to exceed the resident cap.
        assert "--memory-swap" in argv
        j = argv.index("--memory-swap")
        assert argv[j + 1] == str(256 * 1024 * 1024)

    @pytest.mark.asyncio
    async def test_pids_limit_emits_pids_limit_flag(self) -> None:
        argv = await _capture_argv(_spec(pids_limit=512))
        assert "--pids-limit" in argv
        i = argv.index("--pids-limit")
        assert argv[i + 1] == "512"

    @pytest.mark.asyncio
    async def test_no_caps_emits_no_resource_flags(self) -> None:
        """An all-defaults spec leaves the host's default behaviour intact:
        no ``--cpus``, no ``--memory*``, no ``--pids-limit``.
        """
        argv = await _capture_argv(_spec())
        assert "--cpus" not in argv
        assert "--memory" not in argv
        assert "--memory-swap" not in argv
        assert "--pids-limit" not in argv

    @pytest.mark.asyncio
    async def test_all_caps_set_together(self) -> None:
        argv = await _capture_argv(
            _spec(cpu_quota=2.0, memory_bytes=128 * 1024 * 1024, pids_limit=64)
        )
        assert "--cpus" in argv
        assert "--memory" in argv
        assert "--memory-swap" in argv
        assert "--pids-limit" in argv
