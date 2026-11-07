"""``scripts/browser_chromium_pin.py`` — the aios-browser ``:smoked`` Chromium gate.

code-validation's ``Pull browser image from GHCR`` step runs the CLI against
the pulled image's ``chrome --version``; a non-zero exit routes into the local
rebuild. These tests pin that it fails closed on any mismatch with the
``docker/Dockerfile.browser`` pin, and execute the step's real ``run:``
script against a stubbed ``docker``/``uv`` to prove the wiring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.browser_chromium_pin import (
    CHROMIUM_FLOOR,
    DOCKERFILE,
    PinError,
    check_image_version,
    fmt,
    main,
    pinned_version,
    reported_version,
)
from scripts.browser_image_inputs import LABEL, inputs_digest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dockerfile(*versions: str) -> str:
    return "\n".join(
        f"ADD https://storage.googleapis.com/chrome-for-testing-public/{v}/linux64/chrome-linux64.zip /c.zip"
        for v in versions
    )


def _banner(version: str) -> str:
    return f"Google Chrome for Testing {version}\n"


def test_real_dockerfile_pin_parses_and_meets_security_floor() -> None:
    pin = pinned_version(DOCKERFILE.read_text())
    assert pin >= CHROMIUM_FLOOR, (
        f"Dockerfile.browser pins {fmt(pin)}, below the security floor {fmt(CHROMIUM_FLOOR)}"
    )


def test_pin_requires_every_arch_url_to_agree() -> None:
    assert pinned_version(_dockerfile("156.0.8078.80", "156.0.8078.80")) == (156, 0, 8078, 80)
    with pytest.raises(PinError, match="disagreeing"):
        pinned_version(_dockerfile("156.0.8078.80", "156.0.8078.4"))


def test_pin_missing_fails() -> None:
    with pytest.raises(PinError, match="no chrome-for-testing-public"):
        pinned_version("FROM python:3.13-slim-bookworm\n")


def test_matching_image_passes() -> None:
    dockerfile = _dockerfile("156.0.8078.80")
    assert check_image_version(_banner("156.0.8078.80"), dockerfile) == (156, 0, 8078, 80)


@pytest.mark.parametrize(
    ("image", "relation"),
    [
        # The #2569 case: protocol-fresh :smoked still on playwright's 156.0.8078.4.
        ("156.0.8078.4", "older than"),
        ("156.0.8078.73", "older than"),
        ("155.0.9999.99", "older than"),
        # Newer than HEAD's pin is also not the image HEAD builds — rebuild.
        ("156.0.8078.81", "newer than"),
    ],
)
def test_mismatched_image_fails_closed(image: str, relation: str) -> None:
    with pytest.raises(PinError, match=f"{relation} the Dockerfile.browser pin 156.0.8078.80"):
        check_image_version(_banner(image), _dockerfile("156.0.8078.80"))


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Chromium 156.0.8078.80",  # playwright's bundled build, not CfT
        "Google Chrome for Testing 156.0.8078",
        "Google Chrome for Testing 156.0.8078.80 beta",
        "bash: /opt/chrome-for-testing/chrome: No such file or directory",
    ],
)
def test_unrecognised_version_output_fails_closed(output: str) -> None:
    with pytest.raises(PinError, match="unrecognised"):
        reported_version(output)


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    pin = fmt(pinned_version(DOCKERFILE.read_text()))
    assert main(["pin"]) == 0
    assert capsys.readouterr().out.strip() == pin
    assert main(["check", _banner(pin)]) == 0
    assert main(["check", _banner("156.0.8078.4")]) == 1
    assert "::error::image Chromium 156.0.8078.4 is older than" in capsys.readouterr().out


def _e2e_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    )
    steps: list[dict[str, Any]] = workflow["jobs"]["e2e"]["steps"]
    return steps


def _step(*, key: str, value: str) -> dict[str, Any]:
    (step,) = [s for s in _e2e_steps() if s.get(key) == value]
    return step


def test_pull_browser_failure_routes_into_the_local_rebuild() -> None:
    """The gate only self-heals because ``pull_browser`` is
    ``continue-on-error`` and the build step keys on its failure."""
    assert _step(key="id", value="pull_browser")["continue-on-error"] is True
    build = _step(key="name", value="Build browser image")
    assert "steps.pull_browser.outcome == 'failure'" in build["if"]


_FAKE_DOCKER = """#!/usr/bin/env bash
# Stand-in for the pulled image: ``docker run --rm IMAGE CMD...`` runs CMD
# with the image's env, the protocol module print reads $FAKE_PROTOCOL.
case "$1" in
  pull|tag) exit 0 ;;
  image)
    if [[ "$*" == *".Config.Labels"* ]]; then echo "$FAKE_LABELS"; exit 0; fi
    echo "image registered: sha256:fake"; exit 0 ;;
  run)
    shift 3
    if [[ "$1" == python3 ]]; then exec cat "$FAKE_PROTOCOL"; fi
    AIOS_BROWSER_CHROMIUM_EXECUTABLE="$FAKE_CHROME" exec "$@" ;;
esac
echo "unexpected docker invocation: $*" >&2
exit 99
"""

_FAKE_UV = """#!/usr/bin/env bash
[[ "$1" == run && "$2" == python ]] || { echo "unexpected uv invocation: $*" >&2; exit 98; }
shift 2
exec "$FAKE_PYTHON" "$@"
"""


def _run_pull_step(
    tmp_path: Path,
    *,
    chrome: str | None,
    protocol: bytes | None = None,
    labels: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``pull_browser``'s real ``run:`` script under GHA's default
    ``bash -e`` against a fake image whose Chromium is the ``chrome`` script
    body (``None``: the executable does not exist)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("docker", _FAKE_DOCKER), ("uv", _FAKE_UV)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    fake_chrome = tmp_path / "chrome"
    if chrome is not None:
        fake_chrome.write_text(f"#!/usr/bin/env bash\n{chrome}\n")
        fake_chrome.chmod(0o755)
    fake_protocol = tmp_path / "protocol.py"
    fake_protocol.write_bytes(
        protocol
        if protocol is not None
        else (_REPO_ROOT / "src" / "aios" / "sandbox" / "browser_protocol.py").read_bytes()
    )
    script = _step(key="id", value="pull_browser")["run"]
    # Keep parallel test workers off the step's fixed scratch path.
    assert "/tmp/installed-protocol.py" in script
    script = script.replace("/tmp/installed-protocol.py", str(tmp_path / "installed-protocol.py"))
    (tmp_path / "step.sh").write_text(script)
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_CHROME": str(fake_chrome),
        "FAKE_PROTOCOL": str(fake_protocol),
        "FAKE_PYTHON": sys.executable,
        # Default: an image stamped from exactly this checkout's inputs.
        "FAKE_LABELS": labels
        if labels is not None
        else json.dumps({LABEL: inputs_digest(), "org.opencontainers.image.title": "x"}),
    }
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", str(tmp_path / "step.sh")],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pull_step_keeps_an_image_shipping_the_pin(tmp_path: Path) -> None:
    pin = fmt(pinned_version(DOCKERFILE.read_text()))
    r = _run_pull_step(tmp_path, chrome=f"echo 'Google Chrome for Testing {pin}'")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"matches the Dockerfile.browser pin {pin}" in r.stdout


@pytest.mark.parametrize(
    "chrome",
    [
        # The #2569 case: protocol-fresh :smoked still on playwright's 156.0.8078.4.
        "echo 'Google Chrome for Testing 156.0.8078.4'",
        "echo 'Google Chrome for Testing 999.0.0.0'",
        "true",  # empty output
        "echo 'Google Chrome for Testing 156.0.8078.80'; exit 1",  # probe errored
        None,  # executable missing
    ],
)
def test_pull_step_fails_closed_on_a_stale_or_unprobeable_chromium(
    tmp_path: Path, chrome: str | None
) -> None:
    r = _run_pull_step(tmp_path, chrome=chrome)
    assert r.returncode != 0, r.stdout + r.stderr


def test_pull_step_still_fails_on_protocol_drift(tmp_path: Path) -> None:
    pin = fmt(pinned_version(DOCKERFILE.read_text()))
    r = _run_pull_step(
        tmp_path, chrome=f"echo 'Google Chrome for Testing {pin}'", protocol=b"# stale\n"
    )
    assert r.returncode != 0
    assert "browser_protocol.py differs from checkout" in r.stdout


def _current_banner() -> str:
    return f"echo 'Google Chrome for Testing {fmt(pinned_version(DOCKERFILE.read_text()))}'"


@pytest.mark.parametrize(
    "labels",
    [
        # A :smoked published before #2570 carries no digest label at all.
        "null",
        "{}",
        json.dumps({"org.opencontainers.image.title": "aios-browser"}),
        # Protocol + Chromium match, but some other input (driver module,
        # requirements.lock, Dockerfile apt layer) moved on master since.
        json.dumps({LABEL: "sha256:" + "0" * 64}),
        "",
    ],
)
def test_pull_step_fails_closed_on_stale_or_missing_inputs_digest(
    tmp_path: Path, labels: str
) -> None:
    r = _run_pull_step(tmp_path, chrome=_current_banner(), labels=labels)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "browser-image inputs" in r.stdout
