"""Unit coverage for the e2e Chromium sandbox probe (#2320).

``assert_chromium_sandbox_on`` used to pick a renderer pid in one ``docker
exec`` and read its ``/proc`` entry in a second; a renderer reaped in between
made the gate flake (``grep: /proc/117/status: No such file or directory``).
These tests drive the REAL probe script with local ``bash`` against a
fabricated ``/proc`` tree, with a hook that churns renderers between execs —
the cross-exec window made deterministic — plus negative controls proving the
assertion still fails when the sandbox is genuinely off.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.e2e import browser_image_common as bic

INIT_FILTERS = 1
SANDBOXED_FILTERS = 3
INIT_NS = "user:[4026531837]"
RENDERER_NS = "user:[4026532999]"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


class FakeProc:
    """A fabricated ``/proc`` tree: pid 1 is container init, plus renderers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.next_pid = 100
        self._proc(1, ["/sbin/docker-init", "--", "browser-daemon"], INIT_FILTERS, INIT_NS)
        self._proc(7, ["/opt/chrome/chrome", "--type=gpu-process"], INIT_FILTERS, INIT_NS)

    def _proc(self, pid: int, argv: list[str], filters: int, ns: str) -> int:
        d = self.root / str(pid)
        (d / "ns").mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "status").write_text(f"Name:\tchrome\nSeccomp:\t2\nSeccomp_filters:\t{filters}\n")
        (d / "ns" / "user").symlink_to(ns)
        return pid

    def spawn_renderer(self, *, filters: int = SANDBOXED_FILTERS, ns: str = RENDERER_NS) -> int:
        self.next_pid += 1
        argv = ["/opt/chrome/chrome", "--type=renderer", "--renderer-client-id=5"]
        return self._proc(self.next_pid, argv, filters, ns)

    def kill_renderers(self) -> None:
        for d in list(self.root.iterdir()):
            if d.name not in {"1", "7"}:
                shutil.rmtree(d)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def _install(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeProc,
    after_exec: Callable[[], None] | None = None,
) -> list[str]:
    """Route ``docker exec C bash -c S`` to local bash over the fake tree; run
    ``after_exec`` after every exec (the window between round trips)."""
    calls: list[str] = []
    real_run = subprocess.run

    def fake_run(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ["docker", "exec"] and argv[3:5] == ["bash", "-c"], argv
        script = argv[5].replace("/proc", str(fake.root))
        calls.append(script)
        r = real_run(
            ["bash", "-c", script], capture_output=True, text=True, check=False, timeout=timeout
        )
        if after_exec is not None:
            after_exec()
        return r

    clock = FakeClock()
    monkeypatch.setattr(bic, "run", fake_run)
    monkeypatch.setattr("tests.e2e.browser_image_common.time.monotonic", clock.monotonic)
    monkeypatch.setattr("tests.e2e.browser_image_common.time.sleep", clock.sleep)
    return calls


@pytest.fixture
def fake(tmp_path: Path) -> FakeProc:
    return FakeProc(tmp_path / "proc")


def test_renderer_reaped_between_execs_does_not_flake(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProc
) -> None:
    """The #2320 race, deterministic: after EVERY exec the live renderer is
    reaped and a fresh one spawned. A two-exec select-then-inspect probe can
    never read the pid it selected (red at the deadline); the atomic probe
    reads whichever renderer is alive inside its single exec."""
    fake.spawn_renderer()

    def churn() -> None:
        fake.kill_renderers()
        fake.spawn_renderer()

    calls = _install(monkeypatch, fake, after_exec=churn)
    bic.assert_chromium_sandbox_on("c")
    assert len(calls) == 1, "selection and inspection must share ONE exec"


def test_waits_for_a_renderer_to_appear(monkeypatch: pytest.MonkeyPatch, fake: FakeProc) -> None:
    """No renderer yet (page still opening) is the one thing re-probed."""

    def spawn() -> None:
        fake.spawn_renderer()

    calls = _install(monkeypatch, fake, after_exec=spawn)
    bic.assert_chromium_sandbox_on("c")
    assert len(calls) == 2


def test_no_renderer_ever_fails_at_deadline(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProc
) -> None:
    _install(monkeypatch, fake)
    with pytest.raises(AssertionError, match="no readable Chromium renderer"):
        bic.assert_chromium_sandbox_on("c")


def test_negative_control_seccomp_layer_off(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProc
) -> None:
    """NEGATIVE CONTROL: a renderer running WITHOUT Chromium's extra seccomp
    filters (docker's profile alone) must fail — and on the first probe, not
    be retried away."""
    fake.spawn_renderer(filters=INIT_FILTERS)
    calls = _install(monkeypatch, fake)
    with pytest.raises(AssertionError, match="seccomp-bpf layer is NOT applied"):
        bic.assert_chromium_sandbox_on("c")
    assert len(calls) == 1


def test_negative_control_seccomp_off_under_churn(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProc
) -> None:
    """The fix must not let renderer churn mask a real regression: unsandboxed
    renderers that keep dying still fail the gate."""
    fake.spawn_renderer(filters=INIT_FILTERS)

    def churn() -> None:
        fake.kill_renderers()
        fake.spawn_renderer(filters=INIT_FILTERS)

    _install(monkeypatch, fake, after_exec=churn)
    with pytest.raises(AssertionError, match="seccomp-bpf layer is NOT applied"):
        bic.assert_chromium_sandbox_on("c")


def test_negative_control_namespace_layer_off(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProc
) -> None:
    fake.spawn_renderer(ns=INIT_NS)
    _install(monkeypatch, fake)
    with pytest.raises(AssertionError, match="namespace sandbox is NOT active"):
        bic.assert_chromium_sandbox_on("c")


def test_probe_does_not_select_itself(fake: FakeProc) -> None:
    """Run against the REAL /proc with no renderer: the probe's own bash
    process must not match the renderer needle (it would carry init-like
    evidence and report a false 'sandbox off')."""
    r = subprocess.run(
        ["bash", "-c", bic.sandbox_probe_script()], capture_output=True, text=True, check=False
    )
    assert r.returncode == 1, r.stdout
