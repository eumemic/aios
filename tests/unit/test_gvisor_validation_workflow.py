"""Drift checks for the informational gVisor validation workflow."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "gvisor-validation.yml"


def _workflow_text() -> str:
    return _WORKFLOW.read_text()


def test_gvisor_validation_workflow_exists_with_informational_triggers() -> None:
    """The job must never gate a PR -- but it MUST validate master on every push.

    The original contract (#1020) was "manual and weekly scheduled triggers",
    written to keep this job OUT of the required PR check set. That purpose is
    preserved here and still asserted: no ``pull_request:`` trigger, so no PR
    can ever be blocked by it.

    The ``branches: [master]`` ban was a different thing: it pinned the SPELLING
    of "informational" to "never runs on a push", and that cost a real
    regression. On 2026-09-16 #2432 took this suite from 1 failing test to 11,
    and with only a weekly cron on master the failure went unannounced -- it
    surfaced only because a human hand-dispatched two runs and diffed the
    failing test NAMES. A check that validates master once a week is not
    validating master; the exposure window was ~7 days.

    A ``push: branches: [master]`` trigger runs POST-merge. It cannot block a
    PR, so it does not join the required check set and the #1020 intent holds.
    What it changes is how long a gVisor regression on master lives: ~7 minutes
    instead of ~7 days.

    This is the same lesson as the selector assertion below (#2429 review,
    finding X3): guard the MEANING, not the spelling.
    """
    workflow = _workflow_text()

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "cron: '" in workflow

    # THE LOAD-BEARING ASSERTION: never gate a pull request.
    assert "pull_request:" not in workflow

    # Post-merge validation of master is REQUIRED, not merely permitted.
    assert "push:" in workflow
    assert "branches: [master]" in workflow

    # Serialise per-ref. Three runs raced at one SHA on 2026-09-16 and two were
    # auto-cancelled; a `cancelled` run shows in the commit check-runs list as a
    # non-success, indistinguishable from a real failure at a glance. And the
    # in-flight run must NOT be killed -- this job rewrites /etc/docker/daemon.json
    # and restarts the daemon, so a half-killed run can leave global Docker state
    # reconfigured for whatever runs next on that host.
    assert "concurrency:" in workflow
    assert "cancel-in-progress: false" in workflow


def test_gvisor_workflow_installs_and_smokes_runsc_runtime() -> None:
    workflow = _workflow_text()

    assert "https://storage.googleapis.com/gvisor/releases" in workflow
    assert "sudo apt-get install -y runsc" in workflow
    assert (
        '"runtimes":{"runsc":{"path":"/usr/bin/runsc",'
        '"runtimeArgs":["--oci-seccomp","--overlay2=none"]}}' in workflow
    )
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

    # Assert the SEMANTIC properties of the gVisor selector rather than a
    # byte-exact command line. The previous exact-string pin broke the moment
    # the selector legitimately changed, which tells you the test was guarding
    # the spelling instead of the meaning (#2429 review, finding X3).
    assert "AIOS_SANDBOX_RUNTIME=runsc" in workflow
    assert "uv run pytest tests/e2e" in workflow
    assert "--junitxml=e2e-results.xml" in workflow
    assert "-n 4 --dist=loadfile" in workflow

    # The gVisor leg must deselect the netns-sidecar egress tests: that path
    # cannot work under runsc (separate netstacks per container), so those
    # tests can never pass here. It must NOT deselect anything else -- a green
    # run has to still mean "aios works under gVisor".
    assert "not netns_sidecar_egress" in workflow
    assert (
        "-m 'docker and not netns_sidecar_egress and not runsc_dns_unresolved and not perf'"
        in workflow
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
