"""Unit tests for the ``docker --init`` (tini) reaper flag (issue #1421).

Session sandboxes must launch with ``--init`` so Docker runs the bundled
``docker-init`` (tini) as PID 1 and execs the keepalive CMD as its child.
tini reaps orphaned zombies automatically, so PID-cgroup slots free as soon
as a process dies — even after an OOM-kill orphans a Chromium tree — which
prevents the zombie pile-up that exhausts ``--pids-limit`` and wedges the
box with ``fork() → EAGAIN`` (a state with no in-band recovery).

These tests snoop on the ``argv`` the docker backend builds to verify the
flag is emitted by default, can be threaded off via the spec, and composes
with ``--pids-limit`` without disturbing the existing keepalive CMD.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import Mount, SandboxSpec
from aios.sandbox.backends.docker import DockerBackend


def _spec(**overrides: object) -> SandboxSpec:
    """Build a minimal SandboxSpec; ``overrides`` tweaks fields under test."""
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
        "network_policy": UnrestrictedNetworking(),
        "host_gateway_alias": None,
        "image": "aios-sandbox:test",
    }
    base.update(overrides)
    return SandboxSpec(**base)  # type: ignore[arg-type]


async def _capture_argv(spec: SandboxSpec) -> list[str]:
    """Run DockerBackend.create against a stubbed ``docker`` subprocess and
    return the argv the backend would have invoked."""
    captured: dict[str, list[str]] = {}

    async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
        captured["argv"] = argv
        return 0, b"deadbeef1234\n", b""

    with patch("aios.sandbox.backends.docker.run_docker_cli", side_effect=fake_run_docker):
        await DockerBackend().create(spec)
    return captured["argv"]


class TestInitReaperFlag:
    @pytest.mark.asyncio
    async def test_init_defaults_to_true_and_emits_flag(self) -> None:
        """The spec defaults ``init=True`` — every session sandbox must run
        under a reaping PID 1, so an all-defaults spec emits ``--init``."""
        assert _spec().init is True
        argv = await _capture_argv(_spec())
        assert "--init" in argv

    @pytest.mark.asyncio
    async def test_init_false_omits_flag(self) -> None:
        """When threaded off explicitly the backend leaves Docker's default
        (no init) in place — proving the flag is plumbed, not hard-coded."""
        argv = await _capture_argv(_spec(init=False))
        assert "--init" not in argv

    @pytest.mark.asyncio
    async def test_init_composes_with_pids_limit(self) -> None:
        """The reaper composes with ``--pids-limit``: both flags are emitted
        together so freed slots actually return under the cap."""
        argv = await _capture_argv(_spec(init=True, pids_limit=512))
        assert "--init" in argv
        assert "--pids-limit" in argv
        i = argv.index("--pids-limit")
        assert argv[i + 1] == "512"

    @pytest.mark.asyncio
    async def test_init_precedes_the_run_image(self) -> None:
        """``--init`` is a ``docker run`` option, so it must appear before the
        image positional (and thus before the keepalive CMD that runs as
        tini's child)."""
        argv = await _capture_argv(_spec())
        assert "--init" in argv
        assert argv.index("--init") < argv.index("aios-sandbox:test")
