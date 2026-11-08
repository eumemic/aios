"""Drift checks for the aios-sandbox-dev image pipeline (aios#2585).

The e2e contract (``tests/e2e/test_sandbox_dev_image_contract.py``) proves the
BUILT image. These checks pin the wiring around it, which no e2e run can see:

- ``build-sandbox-dev.yml`` triggers on every input (push AND pull_request, the
  same list), so ``:smoked`` cannot go stale and a PR cannot skip the gates.
- It rebuilds when the BASE image publishes (``workflow_run`` on the base
  workflow's exact ``name:``, because a rename would silently orphan it).
- It never pushes from a pull request.
- Every download in the Dockerfile is checksum-pinned.
- The shared base image's CI filter does not fire on the dev Dockerfile, so a
  dev-only PR does not force a base rebuild.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e.test_sandbox_dev_image_contract import GOOGLE_REPO_HYGIENE_SCRIPT, _is_registry_ref

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_DEV_WORKFLOW = _WORKFLOWS / "build-sandbox-dev.yml"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.sandbox-dev"


def _load(path: Path) -> dict[Any, Any]:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    return data


def _on(workflow: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML (YAML 1.1) parses the bare key ``on`` as boolean True.
    on = workflow.get("on", workflow.get(True))
    assert isinstance(on, dict)
    return on


def test_triggers_on_every_input_for_push_and_pr() -> None:
    on = _on(_load(_DEV_WORKFLOW))
    push_paths = on["push"]["paths"]
    assert on["push"]["branches"] == ["master"]
    assert on["pull_request"]["paths"] == push_paths, "PR and push path filters diverged"
    for entry in (
        "docker/Dockerfile.sandbox-dev",
        ".github/workflows/build-sandbox-dev.yml",
        "tests/e2e/test_sandbox_dev_image_contract.py",
        "tests/e2e/test_sandbox_image_contract.py",
        "docker/seccomp-sandbox.json",
    ):
        assert entry in push_paths, f"{entry!r} missing from build-sandbox-dev.yml paths"


def test_rebuilds_when_base_image_publishes() -> None:
    on = _on(_load(_DEV_WORKFLOW))
    base_name = _load(_WORKFLOWS / "build-sandbox.yml")["name"]
    assert on["workflow_run"]["workflows"] == [base_name], (
        "workflow_run must name the base workflow exactly; a rename orphans the trigger"
    )
    assert on["workflow_run"]["branches"] == ["master"]
    build_if = _load(_DEV_WORKFLOW)["jobs"]["build"]["if"]
    assert "workflow_run.conclusion == 'success'" in build_if


def test_never_pushes_from_a_pull_request() -> None:
    jobs = _load(_DEV_WORKFLOW)["jobs"]
    build_steps = jobs["build"]["steps"]
    push_step = next(s for s in build_steps if s.get("uses", "").startswith("docker/build-push"))
    assert push_step["with"]["push"] == "${{ github.event_name != 'pull_request' }}"
    for step in jobs["smoke"]["steps"]:
        run = step.get("run", "")
        if "imagetools create" in run:
            assert step.get("if") == "github.event_name != 'pull_request'", step
        if step.get("uses", "").startswith("docker/build-push"):
            assert step["with"]["push"] is False, step


def test_smoke_runs_base_contract_dev_contract_and_negative_control() -> None:
    smoke = _load(_DEV_WORKFLOW)["jobs"]["smoke"]["steps"]
    runs = "\n".join(str(s.get("run", "")) for s in smoke)
    assert "pytest tests/e2e/test_sandbox_image_contract.py" in runs
    # Only the arm64 manifest check is deselected (the image is amd64-only).
    assert re.findall(r"--deselect (\S+)", runs) == [
        "tests/e2e/test_sandbox_image_contract.py::TestArchitecture::test_manifest_includes_amd64_and_arm64"
    ]
    assert "pytest tests/e2e/test_sandbox_dev_image_contract.py" in runs
    assert 'AIOS_SANDBOX_DEV_IMAGE="$BASE_REF"' in runs, "negative control must target the base"
    names = [s.get("name", "") for s in smoke]
    assert names.index(next(n for n in names if n.startswith("Negative control"))) < names.index(
        "Retag sha → :smoked"
    ), ":smoked must be stamped only after the negative control"


def test_every_download_is_checksum_pinned() -> None:
    text = _DOCKERFILE.read_text()
    adds = re.findall(r"(?m)^ADD (?:.*\\\n)*.*", text)
    assert len(adds) == 3, adds
    for add in adds:
        assert re.search(r"--checksum=sha256:[0-9a-f]{64}", add), add
    assert "google-chrome-stable_${CHROME_VERSION}_amd64.deb" in text
    for run in _run_instructions(text):
        assert not re.search(r"\b(curl|wget)\b", run), f"unpinned download in a RUN: {run}"


def _run_instructions(dockerfile: str) -> list[str]:
    """Every RUN instruction, INCLUDING its backslash-continuation lines."""
    # ``(?:.*\\\n)*`` first: the obvious ``.*(?:\\\n.*)*`` lets the greedy
    # ``.*`` eat the trailing backslash, the group then matches zero times, and
    # the regex silently stops at the first line.
    return re.findall(r"(?m)^RUN (?:.*\\\n)*.*", dockerfile)


def test_run_scan_sees_continuation_lines() -> None:
    """The download scan must catch a curl that is not on the RUN's first line."""
    sample = "RUN apt-get update \\\n && curl -fsSL https://x/y | sh\nRUN true\n"
    runs = _run_instructions(sample)
    assert len(runs) == 2
    assert re.search(r"\b(curl|wget)\b", runs[0])
    assert len(_run_instructions(_DOCKERFILE.read_text())) == 5


def test_base_sandbox_filter_ignores_dev_dockerfile() -> None:
    workflow = (_WORKFLOWS / "code-validation.yml").read_text()
    m = re.search(r"grep -qE '([^']*)'(?:(?!grep -qE).)*?sandbox_changed=true", workflow, re.DOTALL)
    assert m is not None
    assert not re.search(m.group(1), "docker/Dockerfile.sandbox-dev")
    assert re.search(m.group(1), "docker/Dockerfile.sandbox")


# --- Google apt-repo hygiene script (aios#2586 review finding A) -------------
#
# The e2e hygiene test runs GOOGLE_REPO_HYGIENE_SCRIPT inside the image. These
# run the SAME script against a filesystem fixture laid out like the .deb's
# install, so its discrimination is proven without docker. The crux: the .deb
# ships /etc/cron.daily/google-chrome as a SYMLINK, which ``grep -r`` skips.


def _hygiene(root: Path) -> str:
    r = subprocess.run(
        ["bash", "-c", GOOGLE_REPO_HYGIENE_SCRIPT, "hygiene", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def _deb_layout(root: Path, *, cron: bool, sources: bool, repo_add_once: str) -> None:
    """The .deb + postinst layout, minus whatever the Dockerfile's ``rm`` removed."""
    cron_target = root / "opt/google/chrome/cron/google-chrome"
    cron_target.parent.mkdir(parents=True)
    cron_target.write_text("REPOCONFIG=deb [arch=amd64] https://dl.google.com/linux/chrome/deb/\n")
    (root / "etc/cron.daily").mkdir(parents=True)
    (root / "etc/apt/sources.list.d").mkdir(parents=True)
    (root / "etc/apt/sources.list.d/debian.sources").write_text("URIs: http://deb.debian.org/\n")
    (root / "etc/default").mkdir(parents=True)
    (root / "etc/default/google-chrome").write_text(f'repo_add_once="{repo_add_once}"\n')
    if cron:
        (root / "etc/cron.daily/google-chrome").symlink_to(cron_target)
    if sources:
        (root / "etc/apt/sources.list.d/google-chrome.sources").write_text(
            "URIs: https://dl.google.com/linux/chrome/deb/\n"
        )


def test_hygiene_clean_layout_passes(tmp_path: Path) -> None:
    _deb_layout(tmp_path, cron=False, sources=False, repo_add_once="false")
    assert _hygiene(tmp_path) == ""


def test_hygiene_catches_leftover_cron_symlink(tmp_path: Path) -> None:
    """The regression the old ``grep -rIl`` scan missed: drop the cron ``rm``."""
    _deb_layout(tmp_path, cron=True, sources=False, repo_add_once="false")
    old_scan = subprocess.run(
        ["grep", "-rIl", "dl.google.com", f"{tmp_path}/etc/apt/", f"{tmp_path}/etc/cron.daily/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert old_scan.stdout.strip() == "", "fixture no longer reproduces the grep -r blind spot"
    assert "present: /etc/cron.daily/google-chrome" in _hygiene(tmp_path)


def test_hygiene_catches_dangling_cron_symlink(tmp_path: Path) -> None:
    _deb_layout(tmp_path, cron=True, sources=False, repo_add_once="false")
    (tmp_path / "opt/google/chrome/cron/google-chrome").unlink()
    assert "present: /etc/cron.daily/google-chrome" in _hygiene(tmp_path)


def test_hygiene_catches_leftover_apt_source(tmp_path: Path) -> None:
    _deb_layout(tmp_path, cron=False, sources=True, repo_add_once="false")
    assert "present: /etc/apt/sources.list.d/google-chrome.sources" in _hygiene(tmp_path)


def test_hygiene_catches_repo_add_once_true(tmp_path: Path) -> None:
    _deb_layout(tmp_path, cron=False, sources=False, repo_add_once="true")
    assert 'repo_add_once="true"' in _hygiene(tmp_path)


def test_dockerfile_removes_every_path_the_hygiene_script_checks() -> None:
    text = _DOCKERFILE.read_text()
    paths = re.findall(
        r"(/etc/(?:cron\.daily|apt/sources\.list\.d)/google-chrome\S*?)(?=[\s\\;])",
        GOOGLE_REPO_HYGIENE_SCRIPT,
    )
    assert len(paths) == 3, paths
    for path in paths:
        assert path in text, f"Dockerfile.sandbox-dev does not rm {path}"


# --- Negative-control verdict (aios#2586 review finding B) -------------------
#
# Extract the exact JUnit-verdict script from the workflow step and run it on
# synthetic reports. An ERRORED control run must fail the step.


def _negative_control_verdict_script() -> str:
    steps = _load(_DEV_WORKFLOW)["jobs"]["smoke"]["steps"]
    run = next(s["run"] for s in steps if s.get("name", "").startswith("Negative control"))
    m = re.search(r"python3 - <<'EOF'\n(.*?)\nEOF", run, re.DOTALL)
    assert m is not None, "negative-control JUnit verdict heredoc not found"
    return m.group(1)


@pytest.mark.parametrize(
    ("tests", "failures", "errors", "skipped", "ok"),
    [
        (8, 8, 0, 0, True),
        (8, 0, 8, 0, False),  # every test ERRORED (e.g. fixture pytest.fail)
        (8, 7, 1, 0, False),  # one errored, the rest failed
        (8, 7, 0, 0, False),  # one passed against the base
        (8, 7, 0, 1, False),  # one skipped
        (0, 0, 0, 0, False),  # nothing collected
    ],
)
def test_negative_control_verdict(
    tmp_path: Path, tests: int, failures: int, errors: int, skipped: int, ok: bool
) -> None:
    (tmp_path / "negative-control.xml").write_text(
        f'<testsuites><testsuite name="pytest" tests="{tests}" failures="{failures}"'
        f' errors="{errors}" skipped="{skipped}"/></testsuites>'
    )
    r = subprocess.run(
        [sys.executable, "-c", _negative_control_verdict_script()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (r.returncode == 0) is ok, r.stdout + r.stderr


@pytest.mark.parametrize(
    ("image", "registry"),
    [
        ("ghcr.io/eumemic/aios-sandbox-dev:sha-abc1234", True),
        ("localhost:5000/aios-sandbox-dev:ci", True),
        ("aios-sandbox-dev:ci", False),
        ("eumemic/aios-sandbox-dev:ci", False),
    ],
)
def test_manifest_check_is_strict_for_registry_refs(image: str, registry: bool) -> None:
    """The amd64 manifest check must FAIL (not skip) for a pushed ref (finding C)."""
    assert _is_registry_ref(image) is registry
