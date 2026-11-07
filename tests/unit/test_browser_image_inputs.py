"""``scripts/browser_image_inputs.py`` — the aios-browser inputs-digest label.

``build-browser-image.yml`` stamps the image with the digest; code-validation's
``pull_browser`` stale-probe requires the pulled ``:smoked`` label to equal the
checkout's digest. These tests pin that the digest moves on every build input
(and only on build inputs), and that the check fails closed (#2570).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from scripts.browser_image_inputs import (
    LABEL,
    REPO_ROOT,
    DigestError,
    check_labels,
    input_paths,
    inputs_digest,
    main,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal committed tree with the browser image's input layout."""
    files = {
        "docker/Dockerfile.browser": "FROM python:3.13-slim\n",
        ".dockerignore": "tests\npackages/aios-browser-driver/tests\n",
        "src/aios/sandbox/browser_protocol.py": "PROTOCOL = 1\n",
        "packages/aios-browser-driver/pyproject.toml": "[project]\n",
        "packages/aios-browser-driver/requirements.lock": "playwright==1\n",
        "packages/aios-browser-driver/aios_browser_driver/daemon.py": "x = 1\n",
        "packages/aios-browser-driver/tests/test_daemon.py": "def test(): ...\n",
        "src/aios/other.py": "y = 1\n",
    }
    for rel, body in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    return tmp_path


def test_input_set_is_dockerfile_context_minus_driver_tests(repo: Path) -> None:
    assert input_paths(repo) == [
        ".dockerignore",
        "docker/Dockerfile.browser",
        "packages/aios-browser-driver/aios_browser_driver/daemon.py",
        "packages/aios-browser-driver/pyproject.toml",
        "packages/aios-browser-driver/requirements.lock",
        "src/aios/sandbox/browser_protocol.py",
    ]


@pytest.mark.parametrize(
    "changed",
    [
        "docker/Dockerfile.browser",  # e.g. a Chromium pin / apt bump (#2569)
        ".dockerignore",
        "src/aios/sandbox/browser_protocol.py",
        "packages/aios-browser-driver/aios_browser_driver/daemon.py",
        "packages/aios-browser-driver/requirements.lock",
    ],
)
def test_digest_moves_on_every_build_input(repo: Path, changed: str) -> None:
    before = inputs_digest(repo)
    (repo / changed).write_text((repo / changed).read_text() + "# changed\n")
    assert inputs_digest(repo) != before


def test_digest_moves_on_a_new_driver_file(repo: Path) -> None:
    before = inputs_digest(repo)
    new = repo / "packages/aios-browser-driver/aios_browser_driver/new.py"
    new.write_text("z = 1\n")
    _git(repo, "add", str(new))
    assert inputs_digest(repo) != before


def test_digest_ignores_non_inputs(repo: Path) -> None:
    before = inputs_digest(repo)
    for rel in ("packages/aios-browser-driver/tests/test_daemon.py", "src/aios/other.py"):
        (repo / rel).write_text("changed = True\n")
    # Untracked artefacts (bytecode, editor droppings) never reach the digest.
    cache = repo / "packages/aios-browser-driver/aios_browser_driver/__pycache__"
    cache.mkdir()
    (cache / "daemon.cpython-313.pyc").write_bytes(b"\0")
    assert inputs_digest(repo) == before


def test_digest_is_rename_sensitive(repo: Path) -> None:
    """Length framing: moving bytes between files changes the digest."""
    before = inputs_digest(repo)
    a = repo / "packages/aios-browser-driver/pyproject.toml"
    b = repo / "packages/aios-browser-driver/requirements.lock"
    a_text, b_text = a.read_text(), b.read_text()
    a.write_text(a_text + b_text[:3])
    b.write_text(b_text[3:])
    assert inputs_digest(repo) != before


def test_real_input_set_matches_the_browser_changed_ci_filter() -> None:
    """Everything hashed is a path ``browser_changed`` rebuilds on — the two
    definitions of "browser-image input" must not drift apart."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    m = re.search(r"grep -qE '([^']*)'(?:(?!grep -qE).)*?browser_changed=true", workflow, re.DOTALL)
    assert m is not None
    escapees = [p for p in input_paths() if not re.search(m.group(1), p)]
    assert not escapees, f"hashed but not in the browser_changed filter: {escapees}"


def test_matching_label_passes() -> None:
    check_labels(json.dumps({LABEL: "sha256:abc", "other": "x"}), "sha256:abc")


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ("null", "no io.eumemic"),
        ("{}", "no io.eumemic"),
        (json.dumps({LABEL: ""}), "no io.eumemic"),
        (json.dumps({LABEL: "sha256:old"}), "differs"),
        ("", "unparseable"),
        ("[1]", "not an object"),
    ],
)
def test_stale_or_missing_label_fails_closed(labels: str, match: str) -> None:
    with pytest.raises(DigestError, match=match):
        check_labels(labels, "sha256:abc")


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["label"]) == 0
    assert capsys.readouterr().out.strip() == LABEL
    assert main(["digest"]) == 0
    digest = capsys.readouterr().out.strip()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert main(["check", json.dumps({LABEL: digest})]) == 0
    assert main(["check", json.dumps({LABEL: "sha256:stale"})]) == 1
    assert "::error::image io.eumemic" in capsys.readouterr().out


def test_publish_workflow_stamps_the_label() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "build-browser-image.yml").read_text()
    assert "scripts/browser_image_inputs.py label" in workflow
    assert "scripts/browser_image_inputs.py digest" in workflow
    assert "${{ steps.inputs.outputs.label }}" in workflow
