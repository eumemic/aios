"""Drift checks for the informational gVisor validation workflow."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "gvisor-validation.yml"


def _workflow_text() -> str:
    return _WORKFLOW.read_text()


def test_gvisor_validation_workflow_exists_with_informational_triggers() -> None:
    workflow = _workflow_text()

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "cron: '" in workflow
    assert "pull_request:" not in workflow
    assert "branches: [master]" not in workflow


def test_gvisor_workflow_installs_and_smokes_runsc_runtime() -> None:
    workflow = _workflow_text()

    assert "https://storage.googleapis.com/gvisor/releases" in workflow
    assert "sudo apt-get install -y runsc" in workflow
    assert '"runtimes":{"runsc":{"path":"/usr/bin/runsc"}}' in workflow
    assert "sudo systemctl restart docker" in workflow
    assert "docker run --runtime=runsc --rm alpine echo ok" in workflow


def test_gvisor_workflow_mirrors_docker_e2e_setup_and_runs_runsc_shard() -> None:
    workflow = _workflow_text()

    for snippet in [
        "uses: actions/checkout@v4",
        "uses: astral-sh/setup-uv@v4",
        "enable-cache: true",
        "run: uv python install 3.13",
        "run: uv sync --dev",
        "docker build -t aios-sandbox:ci -f docker/Dockerfile.sandbox .",
        "AIOS_DOCKER_IMAGE: aios-sandbox:ci",
    ]:
        assert snippet in workflow

    # ``perf`` is the ADVISORY wall-clock scaling backstop (#1661); excluding it
    # matches ``code-validation.yml``'s gating run. Pinned as an exact string so
    # the exclusion cannot quietly grow into ``not docker_slow``-style selectors
    # that would hide real runsc failures — the only narrowing permitted here is
    # by MARK, and only of the advisory one.
    assert (
        "AIOS_SANDBOX_RUNTIME=runsc uv run pytest tests/e2e -q -m 'docker and not perf' "
        "-n 4 --dist=loadfile --junitxml=e2e-results.xml" in workflow
    )
    assert "continue-on-error" not in workflow, (
        "the runsc shard must stay gating: excluding the advisory perf mark is "
        "not a licence to make docker failures non-fatal"
    )


def test_gvisor_workflow_proves_the_daemon_supports_the_operator_image_mount() -> None:
    """The runsc egress path needs ``--mount type=image`` (Engine 28 + containerd store).

    Netfilter must be programmed from inside the target Sentry (#2310), using
    binaries from a read-only operator image mount. If the runner's daemon
    ignores the ``containerd-snapshotter`` feature flag, every Limited
    provisioning fails deep in the suite with an unrelated-looking error, so the
    capability is probed directly right after the daemon restart.
    """
    workflow = _workflow_text()

    assert '"containerd-snapshotter"' in workflow
    assert '--mount "type=image,src=alpine,dst=/mnt/operator-root"' in workflow


def test_gvisor_workflow_groups_junit_failures_by_risk_in_summary() -> None:
    workflow = _workflow_text()

    assert "if: failure()" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "e2e-results.xml" in workflow
    assert "## gVisor validation failures by risk" in workflow
    assert "### networking" in workflow
    assert "### snapshot" in workflow
    assert "### other" in workflow
