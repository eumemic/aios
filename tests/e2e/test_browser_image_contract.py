"""Contract tests for the aios-browser Docker image (jarbot#106 Phase 2).

Asserts every property the aios worker depends on when it drives an account's
browser container: the runtime posture (uid 1000, daemon CMD, no listeners),
the installed wire contract being byte-identical to the worker's
``src/aios/sandbox/browser_protocol.py``, the ``browser-cli`` exit-code matrix
against a live daemon, and — the load-bearing security assertions — that
Chromium's own layered sandbox is ON under ``docker/seccomp-browser.json``
while mount-namespace creation stays denied, with a NEGATIVE control proving
the authored profile is what makes the launch possible at all.

These tests shell out to the Docker CLI directly — no aios harness, no
Postgres, no async. They require only a Docker daemon plus the image under
test. The whole suite is opt-in: it skips when ``AIOS_SANDBOX_BROWSER_IMAGE``
is unset, so the docker e2e shard stays green on deployments (and PRs) where
the browser image is not wired up.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import needs_docker
from tests.e2e.browser_image_common import (
    IMAGE,
    REPO_ROOT,
    SANDBOX_SECCOMP,
    assert_chromium_sandbox_on,
    assert_mount_ns_denied,
    invoke,
    invoke_raw,
    make_plane,
    run,
    start_browser_container,
    wait_ready,
)

pytestmark = [
    needs_docker,
    pytest.mark.docker,
    pytest.mark.skipif(
        not IMAGE, reason="AIOS_SANDBOX_BROWSER_IMAGE not set; browser-image suite is opt-in"
    ),
]


def _docker_run(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """``docker run --rm IMAGE *args`` — plain posture probes of image defaults."""
    return run(["docker", "run", "--rm", IMAGE, *args], timeout=timeout)


@pytest.fixture(scope="module")
def pulled_image() -> str:
    """Ensure IMAGE is present locally; pull only if absent (a fresh CI build
    under the same tag must not be clobbered by an older registry image)."""
    if run(["docker", "image", "inspect", IMAGE], timeout=10).returncode == 0:
        return IMAGE
    result = run(["docker", "pull", IMAGE], timeout=600)
    if result.returncode != 0:
        pytest.fail(f"could not pull {IMAGE!r}: {result.stderr.strip()}")
    return IMAGE


@pytest.fixture(scope="module")
def daemon(pulled_image: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A live browser container (driver ready, Chromium launched) shared by
    the exit-code-matrix and sandbox-probe tests. Torn down unconditionally."""
    plane = make_plane(tmp_path_factory.mktemp("browser-plane"))
    name = f"aios-browser-contract-{uuid.uuid4().hex[:8]}"
    start_browser_container(pulled_image, plane, name)
    try:
        wait_ready(name)
        yield name
    finally:
        run(["docker", "rm", "--force", name], timeout=30)


# -- image posture -------------------------------------------------------------


def test_runs_as_uid_1000_with_matching_home(pulled_image: str) -> None:
    """USER 1000:1000 = browser_protocol.PLANE_OWNER_UID, resolving to the
    named ``aios`` user, with $HOME owned by the running uid (Chromium refuses
    a foreign-owned home)."""
    r = _docker_run(
        "bash",
        "-c",
        'id -u; id -un; test "$(stat -c %u "$HOME")" = "$(id -u)" && echo home-ok',
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1000", "aios", "home-ok"], r.stdout


def test_image_cmd_is_absolute_daemon(pulled_image: str) -> None:
    """CMD is the daemon by absolute path (and WORKDIR is the plane mount);
    the worker never overrides either, so a bare-name CMD would couple
    container init to PATH (#925/#938)."""
    r = run(
        [
            *("docker", "inspect", "--format"),
            "{{json .Config.Cmd}}\t{{.Config.WorkingDir}}",
            pulled_image,
        ],
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    cmd_json, workdir = r.stdout.strip().split("\t")
    assert json.loads(cmd_json) == ["/usr/local/bin/aios-browser-driver"]
    assert workdir == "/workspace"
    r = _docker_run("test", "-x", "/usr/local/bin/aios-browser-driver")
    assert r.returncode == 0, r.stderr


def test_chromium_installed_world_readable(pulled_image: str) -> None:
    """PLAYWRIGHT_BROWSERS_PATH persisted to runtime and pointing at a tree
    the uid-1000 daemon can execute (root installed it at build time)."""
    r = _docker_run(
        "bash",
        "-c",
        'test "$PLAYWRIGHT_BROWSERS_PATH" = /opt/ms-playwright '
        "&& ls /opt/ms-playwright/chromium-*/chrome-linux*/chrome",
    )
    assert r.returncode == 0, f"chromium binary not found/readable: {r.stderr or r.stdout}"


def test_protocol_bytes_match_checkout(pulled_image: str) -> None:
    """The installed ``aios_browser_driver.browser_protocol`` is byte-identical
    to the worker's ``src/aios/sandbox/browser_protocol.py`` — the single-source
    overwrite in the Dockerfile actually happened, so the wire contract cannot
    fork between worker and driver."""
    r = _docker_run(
        "python3",
        "-c",
        "import aios_browser_driver.browser_protocol as m, sys;"
        "sys.stdout.buffer.write(open(m.__file__, 'rb').read())",
    )
    assert r.returncode == 0, r.stderr
    repo = (REPO_ROOT / "src" / "aios" / "sandbox" / "browser_protocol.py").read_text()
    assert r.stdout == repo, (
        "installed browser_protocol.py differs from src/aios/sandbox/browser_protocol.py — "
        "the image was built from a different protocol revision (stale tag?)"
    )


# -- exit-code matrix (the readiness proof) ------------------------------------


class TestExitCodeContract:
    """The worker's transport currency: exit 0 iff a response document was
    produced (including ``ok: false``); nonzero = browser unavailable."""

    def test_status_ok(self, daemon: str) -> None:
        doc = invoke(daemon, {"op": "status", "timeout_ms": 10_000}, wrapper_s=15)
        assert doc["ok"] is True
        assert doc["boot"]
        assert "signed_in_hosts" in doc["data"]

    def test_malformed_request_is_exit_zero_invalid_request(self, daemon: str) -> None:
        """browser-cli forwards bytes verbatim; the DAEMON rejects garbage with
        a full ``invalid_request`` envelope at exit 0."""
        r = invoke_raw(daemon, "this is not json", wrapper_s=15)
        assert r.returncode == 0, f"stderr={r.stderr!r}"
        doc = json.loads(r.stdout)
        assert doc["ok"] is False
        assert doc["error"]["code"] == "invalid_request"

    def test_unknown_op_is_exit_zero(self, daemon: str) -> None:
        doc = invoke(daemon, {"op": "frobnicate", "timeout_ms": 10_000}, wrapper_s=15)
        assert doc["ok"] is False
        assert doc["error"]["code"] == "unknown_op"

    def test_daemon_down_is_nonzero_connect_exit(self, pulled_image: str, tmp_path: Path) -> None:
        """With no daemon bound, browser-cli must exit 7 (connect failure) —
        never 0, never 137 — after its bounded connect retry."""
        plane = make_plane(tmp_path)
        name = f"aios-browser-nodaemon-{uuid.uuid4().hex[:8]}"
        # CMD override: container alive, daemon NOT running.
        start_browser_container(pulled_image, plane, name, command=["sleep", "300"])
        try:
            r = invoke_raw(name, json.dumps({"op": "status", "timeout_ms": 2_000}), wrapper_s=15)
            assert r.returncode == 7, (
                f"expected exit 7 (connect failure), got {r.returncode}: "
                f"stdout={r.stdout!r} stderr={r.stderr!r}"
            )
        finally:
            run(["docker", "rm", "--force", name], timeout=30)

    def test_usage_error_is_exit_two(self, daemon: str) -> None:
        r = run(["docker", "exec", daemon, "browser-cli"], timeout=30)
        assert r.returncode == 2, f"stdout={r.stdout!r} stderr={r.stderr!r}"


# -- Chromium sandbox (the load-bearing security assertions) -------------------


def test_chromium_sandbox_is_on(daemon: str) -> None:
    """PRIMARY sandbox-ON assertion (plan CI-C2) — renderer namespaced AND
    double-filtered; see :func:`assert_chromium_sandbox_on` for the /proc
    rationale. The isolation gate re-runs the same assertion against a
    container provisioned through the REAL ``build_spec_from_browser`` path."""
    assert_chromium_sandbox_on(daemon)


def test_chromium_launch_fails_under_sandbox_profile(pulled_image: str, tmp_path: Path) -> None:
    """NEGATIVE control (plan CI-C2): under ``seccomp-sandbox.json`` (which
    denies unprivileged userns creation) Chromium cannot construct its sandbox
    and hard-aborts; the daemon crashes visibly, so the container EXITS
    nonzero. This proves the authored browser profile is load-bearing — if
    Chromium ever ran happily under the sandbox profile, either the sandbox
    silently degraded or the profiles converged; both are red."""
    plane = make_plane(tmp_path)
    name = f"aios-browser-negctl-{uuid.uuid4().hex[:8]}"
    start_browser_container(pulled_image, plane, name, seccomp=SANDBOX_SECCOMP)
    try:
        # docker wait blocks until the container exits and prints the exit
        # code; the in-container timeout(1) is not involved, so bound it with
        # the subprocess timeout and treat expiry as "still running".
        try:
            waited = run(["docker", "wait", name], timeout=90)
        except Exception:
            pytest.fail(
                "container still running under the sandbox seccomp profile — Chromium "
                "launched WITHOUT its namespace sandbox, or the profiles no longer differ"
            )
        assert waited.returncode == 0, waited.stderr
        assert int(waited.stdout.strip()) != 0, "daemon exited 0 after a failed Chromium launch"
    finally:
        run(["docker", "rm", "--force", name], timeout=30)


def test_mount_namespace_denied_userns_permitted(daemon: str) -> None:
    """The authored mask's deny half, probed live in the daemon's own
    container — see :func:`assert_mount_ns_denied` for the userns-first
    ordering that keeps the probe non-vacuous."""
    assert_mount_ns_denied(daemon)


def test_no_tcp_listeners(daemon: str) -> None:
    """No process in the container LISTENs on any TCP port — the daemon speaks
    AF_UNIX only and Chromium's CDP rides playwright's --remote-debugging-pipe.
    /proc/net state 0A = LISTEN."""
    r = run(
        ["docker", "exec", daemon, "bash", "-c", "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null"],
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    listeners = [
        line for line in r.stdout.splitlines() if len(line.split()) > 3 and line.split()[3] == "0A"
    ]
    assert listeners == [], f"unexpected TCP listeners in browser container:\n{listeners}"


# -- multi-arch manifest -------------------------------------------------------


def test_manifest_includes_amd64_and_arm64(pulled_image: str) -> None:
    """Published images should be multi-arch. Queries the REMOTE manifest;
    skips gracefully for local-only tags and offline runs."""
    result = run(["docker", "buildx", "imagetools", "inspect", pulled_image], timeout=30)
    if result.returncode != 0:
        pytest.skip(f"imagetools inspect unavailable for {pulled_image}: {result.stderr.strip()}")
    output = result.stdout + result.stderr
    assert "linux/amd64" in output
    assert "linux/arm64" in output
