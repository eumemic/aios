"""Shared helpers for the browser-image e2e suites (jarbot#106 Phase 2).

One test-side statement each of the cross-package contracts the suites
drive, so they cannot drift apart (an earlier revision's readiness poll had
already dropped ``--workdir`` from its private copy):

* :func:`invoke_argv` — the worker's EXACT exec shape
  (``src/aios/sandbox/browser.py`` + ``backends/docker.py``):
  ``docker exec --workdir /workspace C timeout -k 5 -s KILL N bash -c
  "browser-cli invoke '<payload>'"``.
* :func:`start_browser_container` — the production ``docker run`` recipe from
  ``build_spec_from_browser`` + ``DockerBackend.create``: plane mount, the
  authored seccomp profile, ``--ipc private``, ``no-new-privileges``,
  ``--init``, and the default resource caps (2 CPU / 2 GiB+swap-pinned /
  512 pids — ``src/aios/config.py`` ``sandbox_browser_*``), so the suites
  exercise the capped regime production runs in. Omitted knowingly: the
  ``aios-browser`` network (topology is the isolation gate's job),
  ``--interactive`` (the daemon never reads stdin), labels (GC bookkeeping),
  and ``--pull always`` (CI pre-pulls the immutable sha tag).
* :func:`assert_chromium_sandbox_on` / :func:`assert_mount_ns_denied` — the
  two live security probes (Chromium's own sandbox engaged; the authored
  mask's deny half), shared between the image-contract suite (hand-built
  ``docker run``) and the isolation gate (the REAL provisioning path).

Deliberately aios-import-free, like the suites themselves: image smoke must
not break when the aios tree does.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

# Read directly from env. The name is stable — env_prefix="AIOS_" + field
# "sandbox_browser_image" in src/aios/config.py. Empty ⇒ the suites skip.
IMAGE = os.environ.get("AIOS_SANDBOX_BROWSER_IMAGE", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
BROWSER_SECCOMP = REPO_ROOT / "docker" / "seccomp-browser.json"
SANDBOX_SECCOMP = REPO_ROOT / "docker" / "seccomp-sandbox.json"


def run(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)


def invoke_argv(container: str, payload: str, *, wrapper_s: int) -> list[str]:
    """The worker's exact exec shape around one ``browser-cli invoke``.

    ``payload`` is the raw request string (callers ``json.dumps`` well-formed
    requests; the malformed-request test passes garbage verbatim — browser-cli
    forwards bytes as-is by contract).
    """
    command = f"browser-cli invoke {shlex.quote(payload)}"
    return [
        *("docker", "exec", "--workdir", "/workspace", container),
        *("timeout", "-k", "5", "-s", "KILL", str(wrapper_s)),
        *("bash", "-c", command),
    ]


def invoke_raw(
    container: str, payload: str, *, wrapper_s: int = 40
) -> subprocess.CompletedProcess[str]:
    return run(invoke_argv(container, payload, wrapper_s=wrapper_s), timeout=wrapper_s + 15)


def invoke(container: str, request: dict[str, Any], *, wrapper_s: int = 40) -> dict[str, Any]:
    """One well-formed invoke; fails the test on any transport-level
    (nonzero-exit) fault and returns the parsed response document."""
    r = invoke_raw(container, json.dumps(request), wrapper_s=wrapper_s)
    assert r.returncode == 0, f"transport fault (rc={r.returncode}): {r.stderr.strip()[:500]}"
    doc: dict[str, Any] = json.loads(r.stdout)
    return doc


def world_writable(plane: Path) -> None:
    """chmod the plane tree open to the uid-1000 container.

    Tests run as an arbitrary host uid; in production the root worker chowns
    the plane to the workspaces owner (uid 1000) instead. One statement of the
    accommodation, shared with the isolation gate's real-provisioning tests."""
    for p in (plane, *plane.iterdir()):
        p.chmod(0o777)


def make_plane(root: Path) -> Path:
    """A plane dir (the five production subdirs) the uid-1000 container can
    write regardless of the host uid."""
    plane = root / "plane"
    for sub in ("profile", "shots", "frames", "downloads", "input"):
        (plane / sub).mkdir(parents=True)
    world_writable(plane)
    return plane


def start_browser_container(
    image: str,
    plane: Path,
    name: str,
    *,
    seccomp: Path = BROWSER_SECCOMP,
    env: dict[str, str] | None = None,
    command: list[str] | None = None,
) -> None:
    """``docker run`` the way the worker does (see module docstring). ``env``
    exists for the hermetic-test knob, ``command`` for daemon-suppressed
    containers, ``seccomp`` for the negative control."""
    argv = [
        *("docker", "run", "--detach", "--name", name),
        *("--volume", f"{plane}:/workspace"),
        *("--security-opt", "no-new-privileges"),
        *("--security-opt", f"seccomp={seccomp}"),
        *("--ipc", "private"),
        *("--cpus", "2"),
        *("--memory", "2147483648", "--memory-swap", "2147483648"),
        *("--pids-limit", "512"),
        "--init",
    ]
    for key, value in (env or {}).items():
        argv += ["--env", f"{key}={value}"]
    argv.append(image)
    argv.extend(command or [])
    r = run(argv, timeout=120)
    assert r.returncode == 0, f"docker run failed: {r.stderr.strip()}"


def wait_ready(container: str, *, deadline_s: float = 90) -> None:
    """Poll ``status`` until the driver answers ok (Chromium takes a moment to
    launch on first boot)."""
    payload = json.dumps({"op": "status", "timeout_ms": 10_000})
    deadline = time.monotonic() + deadline_s
    last: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        last = invoke_raw(container, payload, wrapper_s=15)
        if last.returncode == 0 and json.loads(last.stdout).get("ok"):
            return
        time.sleep(1.0)
    detail = f"rc={last.returncode} stdout={last.stdout!r} stderr={last.stderr!r}" if last else ""
    pytest.fail(f"driver never became ready in {container}: {detail}")


# CLONE_NEW* flag values (linux/sched.h) for the namespace probes.
CLONE_NEWNS = 0x20000
CLONE_NEWUSER = 0x10000000


def assert_mount_ns_denied(container: str) -> None:
    """The authored mask's DENY half, probed live. ORDER MATTERS: the probe
    first enters a fresh user namespace — what Chromium's zygote needs, and
    the step that grants CAP_SYS_ADMIN *in that namespace* — and only then
    attempts ``unshare(CLONE_NEWNS)``. The kernel would permit that second
    call (the process holds CAP_SYS_ADMIN in its userns), so EPERM there
    proves the seccomp mask and nothing else. Probing NEWNS from the initial
    namespace would be vacuous: the kernel itself EPERMs unprivileged
    mount-namespace creation with or without seccomp.

    Complements :func:`assert_chromium_sandbox_on`, which only proves the
    ALLOW half (any profile permissive enough lets Chromium's sandbox
    engage — including one hand-weakened beyond the authored mask)."""
    probe = (
        "import ctypes;"
        "libc = ctypes.CDLL(None, use_errno=True);"
        f"ruser = libc.unshare({CLONE_NEWUSER}); euser = ctypes.get_errno();"
        f"rns = libc.unshare({CLONE_NEWNS}); ens = ctypes.get_errno();"
        "print(ruser, euser, rns, ens)"
    )
    r = run(["docker", "exec", container, "python3", "-c", probe], timeout=60)
    assert r.returncode == 0, r.stderr
    ruser, _euser, rns, ens = (int(x) for x in r.stdout.split())
    assert ruser == 0, f"unshare(CLONE_NEWUSER) must succeed under the browser profile: {r.stdout}"
    assert (rns, ens) == (-1, 1), (
        f"unshare(CLONE_NEWNS) from inside a userns must fail EPERM (the seccomp mask), "
        f"got rc={rns} errno={ens}"
    )


def renderer_pids(container: str) -> list[str]:
    """PIDs of Chromium renderer processes inside the container."""
    script = (
        "for p in /proc/[0-9]*/cmdline; do "
        "tr '\\0' ' ' < \"$p\" 2>/dev/null | grep -q -- --type=renderer "
        '&& basename "$(dirname "$p")"; done; true'
    )
    r = run(["docker", "exec", container, "bash", "-c", script], timeout=30)
    return [line.strip() for line in r.stdout.splitlines() if line.strip().isdigit()]


def assert_chromium_sandbox_on(container: str) -> None:
    """PRIMARY sandbox-ON assertion (plan CI-C2), via the two /proc-observable
    facts of Chromium's layered sandbox:

    * layer 1 (namespace): the renderer's user namespace differs from the
      container init's;
    * layer 2 (seccomp-bpf): the renderer carries MORE seccomp filters than
      init. ``Seccomp: 2`` alone is vacuous — docker's container-level profile
      puts EVERY process in filter mode — so the load-bearing signal is the
      ``Seccomp_filters`` count (kernel >= 5.9): init has only docker's, the
      renderer has docker's plus Chromium's own.

    chrome://sandbox scraping is deliberately not used (unreliable under new
    headless). Shared between the image-contract suite (hand-built ``docker
    run``) and the isolation gate (the REAL ``build_spec_from_browser``
    provisioning path) so the two probes cannot drift apart."""
    deadline = time.monotonic() + 30
    pids: list[str] = []
    while time.monotonic() < deadline and not pids:
        pids = renderer_pids(container)
        if not pids:
            time.sleep(1.0)
    assert pids, "no Chromium renderer process found — did the persistent context open a page?"

    pid = pids[0]
    r = run(
        [
            *("docker", "exec", container, "bash", "-c"),
            f"grep -h '^Seccomp_filters:' /proc/{pid}/status /proc/1/status"
            f" && readlink /proc/{pid}/ns/user /proc/1/ns/user",
        ],
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    filters_line_r, filters_line_i, renderer_ns, init_ns = r.stdout.strip().splitlines()
    renderer_filters = int(filters_line_r.split()[-1])
    init_filters = int(filters_line_i.split()[-1])
    assert renderer_filters > init_filters, (
        f"renderer has {renderer_filters} seccomp filter(s) vs init's {init_filters} — "
        "Chromium's own seccomp-bpf layer is NOT applied (docker's profile alone)"
    )
    assert renderer_ns != init_ns, (
        "renderer shares the container's user namespace — Chromium's namespace "
        f"sandbox is NOT active ({renderer_ns})"
    )
