"""``scripts/browser_chromium_pin.py`` — the aios-browser ``:smoked`` Chromium gate.

code-validation's ``Pull browser image from GHCR`` step runs the CLI against
the pulled image's ``chrome --version``; a non-zero exit routes into the local
rebuild. These tests pin that it fails closed on any lag behind the
``docker/Dockerfile.browser`` pin, and that the workflow actually invokes it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
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


def test_pull_browser_step_runs_the_chromium_gate() -> None:
    """The gate only routes into the rebuild if it runs INSIDE the
    ``continue-on-error`` ``pull_browser`` step (whose failure the build step
    keys on), and it must probe the pulled image's pinned executable."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    m = re.search(
        r"\n      - name: Pull browser image from GHCR\n(.*?)\n      - name: ", workflow, re.S
    )
    assert m is not None
    step = m.group(1)
    assert "id: pull_browser" in step
    assert "continue-on-error: true" in step
    probe = step.index(
        "docker run --rm aios-browser:ci bash -c '\"$AIOS_BROWSER_CHROMIUM_EXECUTABLE\" --version'"
    )
    gate = step.index('scripts/browser_chromium_pin.py check "$chromium_version" ||')
    assert probe < gate
    assert "exit 1" in step[gate:]
    assert "steps.pull_browser.outcome == 'failure'" in workflow
