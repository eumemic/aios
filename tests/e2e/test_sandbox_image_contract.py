"""Contract tests for the aios-sandbox Docker image.

Pulls the configured sandbox image and asserts that every binary and
runtime property the aios worker depends on is present and functional.
Each assertion is a separate test function so CI can pinpoint failures
precisely.

These tests shell out to the Docker CLI directly -- no aios harness,
no Postgres, no async. They require only a Docker daemon.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import needs_docker

pytestmark = [needs_docker, pytest.mark.docker]

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
    image: str,
    *args: str,
    volume: str | None = None,
    user: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run ``docker run --rm IMAGE *args`` and return the completed process.

    If *volume* is given it is passed as ``--volume <volume>`` before the image name.
    If *user* is given it is passed as ``--user <user>`` before the image name.
    """
    cmd = ["docker", "run", "--rm"]
    if volume is not None:
        cmd += ["--volume", volume]
    if user is not None:
        cmd += ["--user", user]
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
        # gVisor's netstack (runsc) implements legacy netfilter, NOT nft; the
        # lockdown sidecar selects iptables-legacy when present (#1022), so the
        # image MUST ship it. It's the iptables package's legacy alternative.
        "iptables-legacy",
        # IPv6 belt-and-suspenders egress DROP (#1207). The Limited lockdown
        # sidecar mirrors the v4 -P OUTPUT DROP on ip6tables, selecting the
        # legacy backend (runsc netstack speaks the legacy ABI, not nft), so the
        # operator image MUST ship both ip6tables and its legacy alternative.
        "ip6tables",
        "ip6tables-legacy",
        "update-ca-certificates",  # egress-CA trust install (sandbox/setup.py)
        "jq",  # JSON-from-bash composition, esp. piping `tool <name> '{...}'`
        "tool",  # sandbox-native broker CLI (baked from repo bin/tool; #635)
        "node",  # so agents can run npm packages without first apt-installing the runtime
        "npm",
        "tail",  # the image CMD is `/usr/bin/tail -f /dev/null`
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


def test_tail_at_absolute_path(pulled_image: str) -> None:
    """The image CMD is `/usr/bin/tail -f /dev/null` (absolute path — see
    docker/Dockerfile.sandbox and DockerBackend._flatten). Pin the binary at
    that exact path so a base-image swap relocating `tail` is caught in CI,
    not at container init with an opaque `exec` failure (#925, #938)."""
    r = _docker_run(pulled_image, "test", "-x", "/usr/bin/tail")
    assert r.returncode == 0, f"/usr/bin/tail not executable in image: {r.stderr}"


def test_image_cmd_uses_absolute_tail(pulled_image: str) -> None:
    """The image's configured CMD must invoke tail by absolute path.

    test_tail_at_absolute_path proves the binary exists at /usr/bin/tail;
    this proves the image is actually configured to *use* it. Together they
    catch both a base-image swap that relocates tail and a regression that
    reverts the CMD to bare `tail` (PATH-dependent, the #938/#925 failure)."""
    r = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config.Cmd}}", pulled_image],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert r.returncode == 0, f"docker inspect failed: {r.stderr}"
    assert json.loads(r.stdout) == ["/usr/bin/tail", "-f", "/dev/null"], (
        f"image CMD is {r.stdout.strip()}, expected absolute-path tail keepalive"
    )


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
        """The image's python3 ships a working venv module (the model may create venvs)."""
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

    def test_named_user_1000_is_aios(self, pulled_image: str) -> None:
        """`docker exec --user 1000:1000` (the follow-up agent-exec uid) must
        resolve to the named `aios` user, not a bare anonymous uid. The image
        keeps no top-level USER directive (see test_runs_as_root); the uid is
        supplied per-exec."""
        r = _docker_run(pulled_image, "id", "-u", user="1000:1000")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "1000", f"expected uid 1000, got {r.stdout.strip()!r}"
        r = _docker_run(pulled_image, "id", "-un", user="1000:1000")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "aios", f"expected user aios, got {r.stdout.strip()!r}"

    def test_user_1000_has_writable_home(self, pulled_image: str) -> None:
        """uid 1000 owns /home/aios (created via --create-home) and ENV
        HOME=/home/aios applies under `--user` (docker --user does not reset
        HOME), so the agent can write to $HOME."""
        r = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "touch $HOME/.probe && echo ok",
            user="1000:1000",
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok", (
            f"expected writable $HOME, got {r.stdout.strip()!r}: {r.stderr}"
        )

    def test_home_env_is_aios_home_for_root(self, pulled_image: str) -> None:
        """Root (default) execs inherit ENV HOME=/home/aios too — the ENV is
        unconditional, not gated on uid."""
        r = _docker_run(pulled_image, "bash", "-c", "echo $HOME")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "/home/aios", f"expected /home/aios, got {r.stdout.strip()!r}"

    def test_trust_store_layout_is_debian(self, pulled_image: str) -> None:
        """The trust-store contract the egress-CA wiring depends on: a
        Debian-layout aggregate bundle at /etc/ssl/certs/ca-certificates.crt
        (where SSL_CERT_FILE / REQUESTS_CA_BUNDLE point) and the drop-in
        directory update-ca-certificates folds into it. Custom per-env
        images (#724) that diverge must override the trust-store env vars
        via their environment's ``env`` config."""
        r = _docker_run(
            pulled_image,
            "bash",
            "-c",
            "test -s /etc/ssl/certs/ca-certificates.crt && test -d /usr/local/share/ca-certificates",
        )
        assert r.returncode == 0, f"Debian trust-store layout missing: {r.stderr}"

    def test_tool_broker_url_default_in_image_env(self, pulled_image: str) -> None:
        """Dockerfile sets a UDS default so bash scripts reading
        $TOOL_BROKER_URL directly don't fail when env injection lapses
        (issue #698)."""
        r = _docker_run(pulled_image, "bash", "-c", "echo $TOOL_BROKER_URL")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "unix:///var/run/aios/tool-broker.sock"


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


# -- operator prewarm bake (#1348) ---------------------------------------------
#
# These shell out to the Docker CLI directly (no aios harness): bake a prewarm
# image from the base by committing a CA-installed container with the two
# prewarm labels, then assert (a) the labels are stamped exactly (and the
# managed/instance/session labels are NOT — GC-invisibility by construction),
# (b) the egress CA is already in the trust store in the baked image (so the
# cold-start CA exec is genuinely redundant), and (c) the iptables lockdown
# DROP policy can STILL be applied against a container started from the
# prewarmed image — the lockdown is never baked (§5.8). Label key strings are
# inlined to keep this module free of aios package imports (see module
# docstring); they mirror ``PREWARM_LABEL_KEY``/``BASE_IMAGE_LABEL_KEY`` etc.
# in ``src/aios/sandbox/backends/base.py``.

_PREWARM_LABEL_KEY = "aios.prewarmed"
_BASE_IMAGE_LABEL_KEY = "aios.base_image"
_MANAGED_LABEL_KEY = "aios.managed"
_INSTANCE_LABEL_KEY = "aios.instance_id"
_SESSION_LABEL_KEY = "aios.session_id"

# A self-signed CA cert generated once at import time so the bake test does not
# depend on AIOS_EGRESS_CA_KEY (the real CA derivation is operator-side; the
# trust-store mechanism under test is identical for any PEM).
_TEST_CA_PEM = subprocess.run(
    [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "/dev/null",
        "-subj",
        "/CN=aios-prewarm-test-ca",
        "-days",
        "1",
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=30,
).stdout


@pytest.fixture()
def prewarm_image(pulled_image: str) -> str:
    """Bake a prewarm image from the base: run → install CA → commit + labels.

    Mirrors the operator ``bake_prewarm_image`` path with the Docker CLI. Yields
    the prewarm tag and force-removes it (and any leftover container) on teardown.
    """
    tag = "aios-prewarm-e2e:test"
    container = "aios-prewarm-e2e-bake"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False, timeout=30)
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False, timeout=30)

    # 1. Plain run of the base (keepalive CMD), no aios.* labels.
    run = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            container,
            pulled_image,
            "/usr/bin/tail",
            "-f",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode == 0, f"prewarm run failed: {run.stderr.strip()}"

    # 2. Install the CA into the trust store (the amortized setup exec). Pass
    #    the PEM via ``docker exec --env`` so it lands INSIDE the container
    #    (host env is not forwarded into the container shell).
    install = subprocess.run(
        [
            "docker",
            "exec",
            "--env",
            f"CA_PEM={_TEST_CA_PEM}",
            container,
            "bash",
            "-c",
            "printf '%s' \"$CA_PEM\" > /usr/local/share/ca-certificates/aios-egress.crt "
            "&& update-ca-certificates",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert install.returncode == 0, f"prewarm CA install failed: {install.stderr.strip()}"

    # 3. Commit + stamp BOTH prewarm labels (and NONE of managed/instance/session).
    commit = subprocess.run(
        [
            "docker",
            "commit",
            "--change",
            f"LABEL {_BASE_IMAGE_LABEL_KEY}={pulled_image}",
            "--change",
            f"LABEL {_PREWARM_LABEL_KEY}={pulled_image}",
            container,
            tag,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert commit.returncode == 0, f"prewarm commit failed: {commit.stderr.strip()}"
    # 4. Remove the transient container.
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False, timeout=30)

    try:
        yield tag
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False, timeout=30
        )
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False, timeout=30)


class TestPrewarmBake:
    def test_prewarm_labels_stamped_exactly(self, prewarm_image: str, pulled_image: str) -> None:
        """The prewarm image carries BOTH prewarm labels = base ref and NONE of
        the managed/instance/session labels (GC-invisibility by construction)."""
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", prewarm_image],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr.strip()
        labels = json.loads(result.stdout.strip()) or {}
        assert labels.get(_PREWARM_LABEL_KEY) == pulled_image
        assert labels.get(_BASE_IMAGE_LABEL_KEY) == pulled_image
        for forbidden in (_MANAGED_LABEL_KEY, _INSTANCE_LABEL_KEY, _SESSION_LABEL_KEY):
            assert forbidden not in labels, f"prewarm image must not carry {forbidden}"

    def test_egress_ca_already_in_trust_store(self, prewarm_image: str) -> None:
        """The CA is baked into the trust store, so the cold-start CA exec is
        genuinely redundant for a container started from the prewarm image."""
        result = _docker_run(
            prewarm_image,
            "grep",
            "-l",
            "aios-prewarm-test-ca",
            "/etc/ssl/certs/ca-certificates.crt",
            timeout=30,
        )
        # grep -l prints the filename on a match; a baked CA ⇒ rc 0.
        assert result.returncode == 0, (
            "egress CA not found in the prewarm image trust store: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    def test_lockdown_still_appliable_against_prewarm_image(self, prewarm_image: str) -> None:
        """The iptables DROP lockdown is NEVER baked (§5.8): a container started
        from the prewarmed image starts with an open OUTPUT policy and the
        lockdown can still be applied fresh (proving it is not persisted)."""
        container = "aios-prewarm-e2e-lockdown"
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False, timeout=30
        )
        try:
            run = subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--cap-add",
                    "NET_ADMIN",
                    prewarm_image,
                    "/usr/bin/tail",
                    "-f",
                    "/dev/null",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if run.returncode != 0:
                pytest.skip(f"could not start NET_ADMIN container: {run.stderr.strip()}")
            # Baked image must NOT carry a DROP policy (lockdown never baked).
            before = subprocess.run(
                ["docker", "exec", container, "iptables-legacy", "-S", "OUTPUT"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert before.returncode == 0, before.stderr.strip()
            assert "-P OUTPUT DROP" not in before.stdout, (
                "prewarm image must not ship a baked DROP policy"
            )
            # The lockdown can be applied fresh — proving it runs at provision time.
            applied = subprocess.run(
                ["docker", "exec", container, "iptables-legacy", "-P", "OUTPUT", "DROP"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert applied.returncode == 0, applied.stderr.strip()
            after = subprocess.run(
                ["docker", "exec", container, "iptables-legacy", "-S", "OUTPUT"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert "-P OUTPUT DROP" in after.stdout
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, check=False, timeout=30
            )
