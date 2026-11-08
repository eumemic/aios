"""Contract tests for the aios-sandbox-dev image (aios#2585).

``docker/Dockerfile.sandbox-dev`` is the base sandbox plus a browser toolchain:
Google Chrome stable for Playwright's ``channel: "chrome"``, the OS libraries
for Playwright's bundled Chromium, Node 22 LTS, and a pinned playwright-core.
The image exists so e2e-capable build lanes can run browser tests in a
LIMITED-networking environment that cannot reach any browser download host.

The BASE contract is not restated here. ``build-sandbox-dev.yml`` runs
``tests/e2e/test_sandbox_image_contract.py`` unchanged against this image, with
``AIOS_DOCKER_IMAGE`` pointed at it. The one test it deselects is the
amd64+arm64 manifest check, because this image is amd64-only by design. Its
single-arch replacement is :func:`TestDevHygiene.test_manifest_includes_amd64`.

Every test in :class:`TestDevAdditions` asserts something the dev image ADDS.
Each must therefore FAIL against the base image. The workflow's
negative-control step runs that class with ``AIOS_SANDBOX_DEV_IMAGE`` pointed
at the base ``:smoked`` digest, and requires every test to fail. That is
what shows the checks discriminate. :class:`TestDevHygiene` holds properties
that are also true of the base, such as "no unreachable apt source left
behind", so it is excluded from the negative control.

The runtime probes use the worker's own posture: the authored
``docker/seccomp-sandbox.json``, ``no-new-privileges`` and ``--ipc private``.
They also use ``--network none``, which proves nothing is fetched at runtime.
Chrome's own sandbox needs unprivileged user namespaces, which that profile
denies, so Chrome runs with ``--no-sandbox``. Playwright already passes that by
default when it launches as root.

Opt-in. The suite skips when ``AIOS_SANDBOX_DEV_IMAGE`` is unset, so the
generic docker e2e shard (which builds only the base image) does not run it.
The module is free of aios package imports, matching the other image suites.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import needs_docker

IMAGE = os.environ.get("AIOS_SANDBOX_DEV_IMAGE", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.sandbox-dev"
SANDBOX_SECCOMP = REPO_ROOT / "docker" / "seccomp-sandbox.json"

CHROME_EXECUTABLE = "/opt/google/chrome/chrome"  # Playwright's linux ``chrome`` channel path
PLAYWRIGHT_CORE = "/opt/aios-sandbox-dev/node_modules/playwright-core"

pytestmark = [
    needs_docker,
    pytest.mark.docker,
    pytest.mark.skipif(
        not IMAGE, reason="AIOS_SANDBOX_DEV_IMAGE not set; sandbox-dev image suite is opt-in"
    ),
]


def _dockerfile_arg(name: str) -> str:
    """The default of ``ARG <name>=<value>`` in Dockerfile.sandbox-dev (the pin)."""
    m = re.search(rf"(?m)^ARG {re.escape(name)}=(\S+)\s*$", DOCKERFILE.read_text())
    assert m is not None, f"ARG {name}=... not found in {DOCKERFILE}"
    return m.group(1)


def _run(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)


def _docker_run_offline(
    image: str, *args: str, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """``docker run`` with NO network, in the worker's sandbox security posture."""
    argv = [
        *("docker", "run", "--rm"),
        *("--network", "none"),
        *("--security-opt", "no-new-privileges"),
        *("--security-opt", f"seccomp={SANDBOX_SECCOMP}"),
        *("--ipc", "private"),
        image,
        *args,
    ]
    return _run(argv, timeout=timeout)


def _image_labels(image: str) -> dict[str, str]:
    # The whole-Config form: ``.Config.Labels`` errors on a label-less image.
    r = _run(["docker", "image", "inspect", "--format", "{{json .Config}}", image], timeout=30)
    assert r.returncode == 0, f"docker image inspect failed: {r.stderr.strip()}"
    labels = (json.loads(r.stdout) or {}).get("Labels") or {}
    return {str(k): str(v) for k, v in labels.items()}


# Prints one line per leftover of the .deb's Google apt-repo registration, and
# nothing when the image is clean. ``$1`` is an optional root prefix, so the
# unit suite can run the same script against a filesystem fixture.
#
# The .deb ships ``/etc/cron.daily/google-chrome`` as a SYMLINK to
# ``/opt/google/chrome/cron/google-chrome``. GNU ``grep -r`` does not follow
# symlinks met during traversal, so the old ``grep -rIl`` scan could not see a
# leftover cron job. The script therefore names each known path explicitly
# (``-e`` or ``-L``, so a dangling link also counts) and scans with ``grep -R``.
GOOGLE_REPO_HYGIENE_SCRIPT = r"""
R="${1:-}"
for p in /etc/cron.daily/google-chrome \
         /etc/apt/sources.list.d/google-chrome.list \
         /etc/apt/sources.list.d/google-chrome.sources; do
  if [ -e "$R$p" ] || [ -L "$R$p" ]; then echo "present: $p"; fi
done
grep -RIl 'dl\.google\.com' "$R/etc/apt/" "$R/etc/cron.daily/" 2>/dev/null
if grep -qs '^repo_add_once="true"' "$R/etc/default/google-chrome"; then
  echo 'repo_add_once="true" in /etc/default/google-chrome'
fi
true
"""


def _is_registry_ref(image: str) -> bool:
    """True when ``image`` names a registry host (``ghcr.io/...``, ``host:5000/...``)."""
    first, sep, _ = image.partition("/")
    return bool(sep) and ("." in first or ":" in first or first == "localhost")


@pytest.fixture(scope="module")
def dev_image() -> str:
    """Ensure IMAGE is present locally; pull only if absent (see the base suite)."""
    if _run(["docker", "image", "inspect", IMAGE], timeout=10).returncode == 0:
        return IMAGE
    pulled = _run(["docker", "pull", IMAGE], timeout=600)
    if pulled.returncode != 0:
        pytest.fail(f"Could not pull {IMAGE!r}: {pulled.stderr.strip()}")
    return IMAGE


# A page whose body is produced by JavaScript, so ``--dump-dom`` returning the
# marker proves a real render with script execution, not just an echo of the
# URL. A ``data:`` URL needs no network.
_RENDER_MARKER = "aios-sandbox-dev-rendered-42"
_DATA_URL = (
    "data:text/html,<html><body><script>"
    "document.body.textContent='aios-sandbox-dev-rendered-'+(6*7)"
    "</script></body></html>"
)

# Launch Chrome through Playwright's ``chrome`` channel, the exact option
# aios-console's playwright.config.ts uses, then render, read back and close.
_PLAYWRIGHT_SCRIPT = f"""
const {{ chromium }} = require({json.dumps(PLAYWRIGHT_CORE)});
(async () => {{
  const browser = await chromium.launch({{ channel: "chrome" }});
  const page = await browser.newPage();
  await page.setContent("<p id=m>aios-playwright-chrome-ok</p>");
  const text = await page.textContent("#m");
  const version = browser.version();
  await browser.close();
  console.log(JSON.stringify({{ text, version }}));
}})().catch((err) => {{ console.error(err); process.exit(1); }});
"""


class TestDevAdditions:
    """What the dev image ADDS. Every test here must FAIL against the base image
    (the workflow's negative control enforces that)."""

    def test_chrome_at_playwright_channel_path(self, dev_image: str) -> None:
        r = _docker_run_offline(dev_image, "test", "-x", CHROME_EXECUTABLE)
        assert r.returncode == 0, f"{CHROME_EXECUTABLE} is not executable: {r.stderr.strip()}"

    def test_google_chrome_version_is_the_pin(self, dev_image: str) -> None:
        r = _docker_run_offline(dev_image, "google-chrome", "--version")
        assert r.returncode == 0, f"google-chrome --version failed: {r.stderr.strip()}"
        expected = _dockerfile_arg("CHROME_VERSION").rsplit("-", 1)[0]  # strip the deb revision
        assert re.fullmatch(rf"Google Chrome {re.escape(expected)}\s*", r.stdout), (
            f"installed Chrome {r.stdout.strip()!r} != Dockerfile pin {expected!r}"
        )

    def test_headless_chrome_renders_offline(self, dev_image: str) -> None:
        r = _docker_run_offline(
            dev_image,
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--dump-dom",
            _DATA_URL,
        )
        assert r.returncode == 0, f"headless Chrome failed: {r.stderr.strip()[-2000:]}"
        assert _RENDER_MARKER in r.stdout, f"rendered DOM lacks the marker: {r.stdout[-1000:]!r}"

    def test_playwright_chrome_channel_launches_offline(self, dev_image: str) -> None:
        r = _docker_run_offline(dev_image, "node", "-e", _PLAYWRIGHT_SCRIPT)
        assert r.returncode == 0, (
            f"Playwright chrome-channel launch failed: {r.stderr.strip()[-2000:]}"
        )
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert out["text"] == "aios-playwright-chrome-ok", out
        expected = _dockerfile_arg("CHROME_VERSION").rsplit("-", 1)[0]
        assert out["version"] == expected, f"Playwright drove {out['version']!r}, not {expected!r}"

    def test_bundled_chromium_os_deps_installed(self, dev_image: str) -> None:
        """Playwright's own dependency check for its bundled Chromium reports
        nothing missing. A repo on the default ``chromium`` channel then needs
        only its browser binary."""
        r = _docker_run_offline(
            dev_image, "node", f"{PLAYWRIGHT_CORE}/cli.js", "install-deps", "--dry-run", "chromium"
        )
        assert r.returncode == 0, r.stderr.strip()
        assert "All system dependencies are installed" in r.stdout, r.stdout

    def test_node_is_pinned_lts(self, dev_image: str) -> None:
        """``node`` on PATH is the pinned LTS (>= 20 satisfies aios-console's
        engines). The base image ships Debian's Node 18."""
        r = _docker_run_offline(dev_image, "node", "--version")
        assert r.returncode == 0, r.stderr.strip()
        assert r.stdout.strip() == f"v{_dockerfile_arg('NODE_VERSION')}", r.stdout

    def test_node_resolves_under_worker_path(self, dev_image: str) -> None:
        """The worker injects its own PATH (``WORKSPACE_RUNTIME_ENV``), not the
        image's. Under that PATH ``node`` must still be the pinned one."""
        path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        r = _run(
            [
                *("docker", "run", "--rm", "--network", "none", "--env", f"PATH={path}"),
                *(dev_image, "bash", "-c", "command -v node && node --version"),
            ],
            timeout=60,
        )
        assert r.returncode == 0, r.stderr.strip()
        assert r.stdout.split() == ["/usr/local/bin/node", f"v{_dockerfile_arg('NODE_VERSION')}"]

    def test_version_labels_match_pins(self, dev_image: str) -> None:
        labels = _image_labels(dev_image)
        for label, arg in (
            ("aios.sandbox_dev.chrome_version", "CHROME_VERSION"),
            ("aios.sandbox_dev.node_version", "NODE_VERSION"),
            ("aios.sandbox_dev.playwright_core_version", "PLAYWRIGHT_CORE_VERSION"),
        ):
            assert labels.get(label) == _dockerfile_arg(arg), f"{label}={labels.get(label)!r}"
        assert labels.get("aios.sandbox_dev.base_image", "").startswith(
            "ghcr.io/eumemic/aios-sandbox"
        ), labels


class TestDevHygiene:
    """Properties that hold for the base too, so they are not in the negative control."""

    def test_no_google_apt_source_left_behind(self, dev_image: str) -> None:
        """The .deb's postinst registers dl.google.com as an apt source and a cron
        job that re-adds it. The lane cannot reach dl.google.com, so a leftover
        source would break every in-lane ``apt-get update``.

        See :data:`GOOGLE_REPO_HYGIENE_SCRIPT`: the cron job is a SYMLINK, so a
        plain ``grep -r`` cannot see it."""
        r = _docker_run_offline(dev_image, "bash", "-c", GOOGLE_REPO_HYGIENE_SCRIPT)
        assert r.returncode == 0, r.stderr.strip()
        assert r.stdout.strip() == "", f"Google apt source/cron left behind: {r.stdout.strip()}"

    def test_repo_playwright_not_shadowed(self, dev_image: str) -> None:
        """The baked playwright-core lives in a private prefix. A bare
        ``require('playwright-core')`` from a repo dir must NOT resolve to it, or
        it would shadow the repo's own pinned version."""
        r = _docker_run_offline(
            dev_image,
            "bash",
            "-c",
            "cd /workspace && node -e \"require.resolve('playwright-core')\"",
        )
        assert r.returncode != 0, f"playwright-core resolves globally: {r.stdout.strip()}"

    def test_manifest_includes_amd64(self, dev_image: str) -> None:
        """Single-arch replacement for the base suite's amd64+arm64 check
        (deselected for this image): amd64 must be present.

        A registry ref (non-PR runs: the pushed ``sha-*`` tag) must have a
        manifest that ``imagetools inspect`` can read, and a failure there is a
        FAILURE, not a skip. A local-only tag (PR runs: ``aios-sandbox-dev:ci``,
        never pushed) has no registry manifest, so the locally built image's
        own architecture is checked instead. Nothing here skips."""
        if _is_registry_ref(dev_image):
            r = _run(["docker", "buildx", "imagetools", "inspect", dev_image], timeout=30)
            assert r.returncode == 0, f"imagetools inspect {dev_image} failed: {r.stderr.strip()}"
            assert "linux/amd64" in r.stdout + r.stderr, (r.stdout + r.stderr)[:500]
        else:
            r = _run(
                ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", dev_image],
                timeout=30,
            )
            assert r.returncode == 0, f"docker image inspect failed: {r.stderr.strip()}"
            assert r.stdout.strip() == "linux/amd64", r.stdout.strip()
