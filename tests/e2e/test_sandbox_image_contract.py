"""Contract tests for the aios-sandbox Docker image.

Pulls the configured sandbox image and asserts that every binary and
runtime property the aios worker depends on is present and functional.
Each assertion is a separate test function so CI can pinpoint failures
precisely.

These tests shell out to the Docker CLI directly -- no aios harness,
no Postgres, no async. They require only a Docker daemon.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import needs_docker

pytestmark = needs_docker

# Read directly from env to keep this file free of aios package imports.
# The env var name is stable -- derived from env_prefix="AIOS_" + field "docker_image"
# in src/aios/config.py.
IMAGE = os.environ.get("AIOS_DOCKER_IMAGE", "ghcr.io/eumemic/aios-sandbox:latest")


@pytest.fixture(scope="module")
def pulled_image() -> str:
    """Ensure IMAGE is present locally; pull from registry only if absent.

    Skipping the pull when the tag already resolves locally is essential
    in CI, where the Dockerfile-changed workflow builds the image under
    ``ghcr.io/eumemic/aios-sandbox:latest`` immediately before the e2e
    suite runs. An unconditional ``docker pull`` would overwrite that
    fresh local build with the older registry tag, masking the very
    change the build was meant to validate.

    Module-scoped so this check runs at most once per pytest invocation.
    """
    check = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if check.returncode == 0:
        return IMAGE
    result = subprocess.run(
        ["docker", "pull", IMAGE],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"Could not pull {IMAGE!r}: {result.stderr.strip()}")
    return IMAGE


def _docker_run(
    image: str, *args: str, volume: str | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    """Run ``docker run --rm IMAGE *args`` and return the completed process.

    If *volume* is given it is passed as ``--volume <volume>`` before the image name.
    """
    cmd = ["docker", "run", "--rm"]
    if volume is not None:
        cmd += ["--volume", volume]
    cmd.append(image)
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)


# -- binary availability -------------------------------------------------------


@pytest.mark.parametrize(
    "binary",
    [
        "bash",
        "python3",
        "python3.13",
        "rg",  # ripgrep -- required by the glob and grep tools
        "curl",
        "git",
        "iptables",  # required for limited-networking mode
        "jq",  # JSON-from-bash composition, esp. piping `mcp <server> <tool>`
        "mcp",  # sandbox-native MCP CLI bridge (baked from repo bin/mcp; #378, fixed #392)
        "node",  # so agents can run npm packages without first apt-installing the runtime
        "npm",
        "tail",  # the image CMD is `tail -f /dev/null`
        "cat",
        "head",
        "grep",
        "find",
        "sed",
        "awk",
    ],
)
def test_binary_available(pulled_image: str, binary: str) -> None:
    """Verify every binary the harness depends on is present in the image.

    Each binary gets its own test-case ID in CI output so failures pinpoint
    the exact missing binary.
    """
    r = _docker_run(pulled_image, "which", binary)
    assert r.returncode == 0, f"{binary!r} not found: {r.stderr}"


# -- runtime behaviour ---------------------------------------------------------


class TestRuntimeBehaviour:
    """Verify the functional contracts the harness relies on."""

    def test_bash_minus_c_executes(self, pulled_image: str) -> None:
        """The harness passes every exec call as ``bash -c <command>``."""
        r = _docker_run(pulled_image, "bash", "-c", "echo contract-ok")
        assert r.returncode == 0, r.stderr
        assert "contract-ok" in r.stdout

    def test_python3_version_is_313(self, pulled_image: str) -> None:
        """Image must ship Python 3.13 (base: python:3.13-slim-bookworm)."""
        r = _docker_run(pulled_image, "python3", "--version")
        assert r.returncode == 0, r.stderr
        version_output = r.stdout + r.stderr
        # Matches Python 3.13, 3.14, etc. — forward-compatible minimum-version check.
        match = re.search(r"Python (\d+)\.(\d+)", version_output)
        assert match is not None, f"could not parse version from: {version_output!r}"
        major, minor = int(match.group(1)), int(match.group(2))
        assert (major, minor) >= (3, 13), f"expected Python >= 3.13, got {version_output.strip()!r}"

    def test_python3_venv_creation(self, pulled_image: str) -> None:
        """setup.py calls ``python3 -m venv /workspace/.venv`` on first provision."""
        r = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "python3 -m venv /tmp/test-venv && test -f /tmp/test-venv/bin/python",
        )
        assert r.returncode == 0, f"venv creation failed: {r.stderr}"

    def test_rg_search(self, pulled_image: str) -> None:
        """rg must work end-to-end (glob and grep tools both invoke it)."""
        r = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "echo 'hello world' > /tmp/test.txt && rg 'hello' /tmp/test.txt",
        )
        assert r.returncode == 0, r.stderr
        assert "hello" in r.stdout

    def test_workspace_is_workdir(self, pulled_image: str) -> None:
        """WORKDIR /workspace -- exec calls default cwd to /workspace."""
        r = _docker_run(pulled_image, "bash", "-c", "pwd")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "/workspace"

    def test_runs_as_root(self, pulled_image: str) -> None:
        """Dockerfile has no USER directive so iptables (requires root) can run."""
        r = _docker_run(pulled_image, "bash", "-c", "id -u")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "0", f"expected uid 0, got {r.stdout.strip()!r}"


# -- workspace mount -----------------------------------------------------------


class TestWorkspaceMount:
    """Verify the /workspace bind-mount contract."""

    def test_container_can_write_to_mounted_workspace(
        self, pulled_image: str, tmp_path: Path
    ) -> None:
        """The harness mounts a host directory at /workspace; the container
        must be able to write there so tool outputs survive the exec call."""
        sentinel = tmp_path / "sentinel.txt"
        result = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "echo 'mounted-ok' > /workspace/sentinel.txt",
            volume=f"{tmp_path}:/workspace",
        )
        assert result.returncode == 0, result.stderr
        assert sentinel.exists(), "sentinel file not created on host"
        assert "mounted-ok" in sentinel.read_text()

    def test_container_can_read_from_mounted_workspace(
        self, pulled_image: str, tmp_path: Path
    ) -> None:
        """Files placed in the host workspace are visible to the container."""
        (tmp_path / "host-file.txt").write_text("from-host\n")
        result = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "cat /workspace/host-file.txt",
            volume=f"{tmp_path}:/workspace",
        )
        assert result.returncode == 0, result.stderr
        assert "from-host" in result.stdout


# -- architecture --------------------------------------------------------------


class TestArchitecture:
    """Verify multi-arch and platform properties."""

    def test_image_architecture_matches_runner(self, pulled_image: str) -> None:
        """The pulled image should run on the current host without emulation."""
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Architecture}}",
                pulled_image,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            pytest.skip(f"docker inspect failed: {result.stderr.strip()}")
        image_arch = result.stdout.strip()
        machine = platform.machine()
        expected_map = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
        expected = expected_map.get(machine)
        if expected is None:
            pytest.skip(f"unknown host machine type: {machine!r}")
        assert image_arch == expected, (
            f"image architecture {image_arch!r} does not match runner {machine!r}"
        )

    def test_manifest_includes_amd64_and_arm64(self, pulled_image: str) -> None:
        """Published images should be multi-arch (amd64 + arm64).

        Note: ``docker buildx imagetools inspect`` queries the remote registry
        manifest, not the local pull cache. This test requires outbound network
        access to the registry and skips gracefully when unavailable.

        Skips if ``docker buildx`` or ``imagetools`` sub-command is unavailable.
        """
        result = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", pulled_image],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"docker buildx imagetools not available: {result.stderr.strip()}")
        output = result.stdout + result.stderr
        assert "linux/amd64" in output, f"amd64 not found in manifest: {output[:500]}"
        assert "linux/arm64" in output, f"arm64 not found in manifest: {output[:500]}"
