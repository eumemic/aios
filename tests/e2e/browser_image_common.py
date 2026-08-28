"""Shared helpers for the two browser-image e2e suites (jarbot#106 Phase 2).

One test-side statement each of the two cross-package contracts both suites
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


def make_plane(root: Path) -> Path:
    """A plane dir (the five production subdirs) the uid-1000 container can
    write regardless of the host uid."""
    plane = root / "plane"
    for sub in ("profile", "shots", "frames", "downloads", "input"):
        (plane / sub).mkdir(parents=True)
    for p in (plane, *plane.iterdir()):
        p.chmod(0o777)
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
