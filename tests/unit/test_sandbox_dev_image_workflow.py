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
from pathlib import Path
from typing import Any

import yaml

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
    adds = re.findall(r"(?m)^ADD .*(?:\\\n.*)*", text)
    assert len(adds) == 3, adds
    for add in adds:
        assert re.search(r"--checksum=sha256:[0-9a-f]{64}", add), add
    assert "google-chrome-stable_${CHROME_VERSION}_amd64.deb" in text
    assert not re.search(r"(?m)^RUN .*\b(curl|wget)\b", text), "unpinned download in a RUN"


def test_base_sandbox_filter_ignores_dev_dockerfile() -> None:
    workflow = (_WORKFLOWS / "code-validation.yml").read_text()
    m = re.search(r"grep -qE '([^']*)'(?:(?!grep -qE).)*?sandbox_changed=true", workflow, re.DOTALL)
    assert m is not None
    assert not re.search(m.group(1), "docker/Dockerfile.sandbox-dev")
    assert re.search(m.group(1), "docker/Dockerfile.sandbox")
