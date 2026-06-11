"""Dockerfile-contract tests for the signal-cli SPQR hotfix (#907).

These pin the from-source rebuild of signal-cli at upstream ``bf1376d``
plus the Mode-B receipt-guard patch.  They are the red->green target for
the hotfix: the stock 0.14.3 native-tarball Dockerfile fails most of them.

The Dockerfile is read as text (not parsed) — the contract is "this
instruction is present", which is what a regression would silently drop.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).parent.parent / "Dockerfile"
PATCH = Path(__file__).parent.parent / "signal-cli-modeb-receipt-guard.patch"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    # Read inside a fixture (not at import time) so a missing/unreadable
    # Dockerfile yields a clean named test failure instead of crashing
    # pytest collection.
    return DOCKERFILE.read_text()


def test_pins_full_upstream_commit_sha(dockerfile_text: str) -> None:
    assert "bf1376d74da494d687d4ee60abc20d288ab4fa40" in dockerfile_text


def test_is_multistage_with_jdk25_build_stage(dockerfile_text: str) -> None:
    from_lines = [
        line for line in dockerfile_text.splitlines() if line.lstrip().startswith("FROM ")
    ]
    assert len(from_lines) >= 2
    # A build stage on the full JDK 25 image (the ``-jre`` runtime image
    # cannot compile, so match the JDK image specifically).
    build_stage = [line for line in from_lines if "zulu-openjdk:25" in line and "-jre" not in line]
    assert build_stage, from_lines


def test_applies_modeb_patch(dockerfile_text: str) -> None:
    assert re.search(r"git apply\b.*modeb", dockerfile_text) is not None


def test_patch_file_exists_on_disk() -> None:
    assert PATCH.is_file()
    patch_text = PATCH.read_text()
    assert "IncomingMessageHandler.java" in patch_text
    assert "getSourceServiceId() == null" in patch_text


def test_retains_version_smoke(dockerfile_text: str) -> None:
    assert re.search(r"signal-cli\s+--version", dockerfile_text) is not None


def test_has_revert_to_stock_marker(dockerfile_text: str) -> None:
    marker_lines = [
        line
        for line in dockerfile_text.splitlines()
        if line.lstrip().startswith("#") and "REVERT-TO-STOCK" in line.upper()
    ]
    assert marker_lines, "expected a REVERT-TO-STOCK marker comment"
    # The marker block must reference both this issue and the upstream bug.
    assert "#907" in dockerfile_text
    assert "#2059" in dockerfile_text


def test_runtime_stage_is_python_base(dockerfile_text: str) -> None:
    # The build stage is JDK 25, but the FINAL runtime stage must stay a
    # Python image — the connector runs ``python -m aios_signal``, so a
    # regression that shipped a JRE-only runtime base would have no Python.
    assert "FROM python:3.13-slim-bookworm" in dockerfile_text


def test_no_active_native_tarball_download(dockerfile_text: str) -> None:
    native_tarball = re.compile(r"signal-cli-.*-Linux-native\.tar\.gz")
    active = [
        line
        for line in dockerfile_text.splitlines()
        if native_tarball.search(line) and not line.lstrip().startswith("#")
    ]
    assert not active, f"native-tarball download must be commented out, found: {active}"
