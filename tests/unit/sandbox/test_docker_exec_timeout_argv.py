"""``DockerBackend.exec`` wraps the agent command in an in-container GNU
``timeout`` so the SIGKILL on deadline reaches the daemon-spawned workload
itself, not just the host ``docker exec`` client (issue #844 / moby#9098).

These tests pin the exact argv the backend builds and the ``timed_out``
mapping, by monkeypatching ``run_subprocess_with_timeout`` (imported into
the docker backend module at import time) to capture the call and return a
synthetic ``(rc, stdout, stderr, timed_out)`` tuple. The real docker daemon
is never touched. ``asyncio_mode = "auto"`` means ``async def test_...``
needs no decorator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.sandbox.backends import docker as docker_backend
from aios.sandbox.backends.base import SandboxHandle
from aios.sandbox.backends.docker import DockerBackend


def _install_exec_capture(
    monkeypatch: pytest.MonkeyPatch, *, rc: int = 0, timed_out: bool = False
) -> dict[str, object]:
    """Patch ``run_subprocess_with_timeout`` to record argv + timeout_s and
    return a synthetic result tuple."""
    calls: dict[str, object] = {}

    async def fake_run(argv: list[str], *, timeout_s: float) -> tuple[int, bytes, bytes, bool]:
        calls["argv"] = list(argv)
        calls["timeout_s"] = timeout_s
        return rc, b"out", b"err", timed_out

    monkeypatch.setattr(docker_backend, "run_subprocess_with_timeout", fake_run)
    return calls


def _handle() -> SandboxHandle:
    return SandboxHandle(
        session_id="sess_exec",
        sandbox_id="container_abc123",
        workspace_path=Path("/tmp/ws"),
    )


async def test_argv_wraps_command_in_container_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC1 (AC#1, #3): the in-container command is wrapped in GNU
    ``timeout -k 5 -s KILL <deadline>`` before ``bash -c``."""
    calls = _install_exec_capture(monkeypatch)
    await DockerBackend().exec(_handle(), "echo hi", timeout_seconds=30, max_output_bytes=1000)
    assert calls["argv"] == [
        "docker",
        "exec",
        "--workdir",
        "/workspace",
        "container_abc123",
        "timeout",
        "-k",
        "5",
        "-s",
        "KILL",
        "30",
        "bash",
        "-c",
        "echo hi",
    ]


async def test_argv_deadline_tracks_timeout_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC1 cont.: the deadline element (immediately before ``bash``) is the
    requested ``timeout_seconds``."""
    calls = _install_exec_capture(monkeypatch)
    await DockerBackend().exec(_handle(), "echo hi", timeout_seconds=42, max_output_bytes=1000)
    argv = calls["argv"]
    assert isinstance(argv, list)
    bash_idx = argv.index("bash")
    assert argv[bash_idx - 1] == "42"


async def test_host_backstop_exceeds_container_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC2 (AC#1): the host-side ``wait_for`` backstop is larger than the
    in-container deadline so it never pre-empts the honest in-container path."""
    calls = _install_exec_capture(monkeypatch)
    await DockerBackend().exec(_handle(), "echo hi", timeout_seconds=30, max_output_bytes=1000)
    assert calls["timeout_s"] == 40.0
    assert isinstance(calls["timeout_s"], float)
    assert calls["timeout_s"] > 30


async def test_rc_137_maps_to_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC3: a SIGKILL-path timeout exit (137) is reported as ``timed_out`` even
    though the host backstop did not fire."""
    _install_exec_capture(monkeypatch, rc=137, timed_out=False)
    result = await DockerBackend().exec(
        _handle(), "sleep 999", timeout_seconds=30, max_output_bytes=1000
    )
    assert result.timed_out is True
    assert result.exit_code == 137


async def test_rc_124_maps_to_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC4: the canonical ``timeout`` exit (124, TERM path) is also reported as
    ``timed_out``."""
    _install_exec_capture(monkeypatch, rc=124, timed_out=False)
    result = await DockerBackend().exec(
        _handle(), "sleep 999", timeout_seconds=30, max_output_bytes=1000
    )
    assert result.timed_out is True


async def test_ordinary_exit_codes_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC5: a clean exit (0) and an ordinary failure (1) are NOT flagged as
    timed out."""
    _install_exec_capture(monkeypatch, rc=0, timed_out=False)
    result = await DockerBackend().exec(
        _handle(), "true", timeout_seconds=30, max_output_bytes=1000
    )
    assert result.timed_out is False

    _install_exec_capture(monkeypatch, rc=1, timed_out=False)
    result = await DockerBackend().exec(
        _handle(), "false", timeout_seconds=30, max_output_bytes=1000
    )
    assert result.timed_out is False


async def test_host_backstop_path_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """TC6: when the host backstop fires (rc=-1, timed_out=True) the result
    still reports ``timed_out`` — the host path is preserved."""
    _install_exec_capture(monkeypatch, rc=-1, timed_out=True)
    result = await DockerBackend().exec(
        _handle(), "sleep 999", timeout_seconds=30, max_output_bytes=1000
    )
    assert result.timed_out is True


def test_cancel_description_corrected() -> None:
    """TC7 (AC#2): the cancel description no longer promises stopping a
    long-running operation, and tells the model the bash command keeps
    executing inside the sandbox."""
    from aios.tools.cancel import CANCEL_DESCRIPTION

    assert "stop a long-running operation" not in CANCEL_DESCRIPTION
    assert "keeps executing inside the sandbox" in CANCEL_DESCRIPTION
