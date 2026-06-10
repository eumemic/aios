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

DOCKERFILE = Path(__file__).parent.parent / "Dockerfile"
PATCH = Path(__file__).parent.parent / "signal-cli-modeb-receipt-guard.patch"

DOCKERFILE_TEXT = DOCKERFILE.read_text()


def test_pins_full_upstream_commit_sha() -> None:
    assert "bf1376d74da494d687d4ee60abc20d288ab4fa40" in DOCKERFILE_TEXT


def test_is_multistage_with_jdk25_build_stage() -> None:
    from_lines = [
        line for line in DOCKERFILE_TEXT.splitlines() if line.lstrip().startswith("FROM ")
    ]
    assert len(from_lines) >= 2
    # A build stage on the full JDK 25 image (the ``-jre`` runtime image
    # cannot compile, so match the JDK image specifically).
    build_stage = [line for line in from_lines if "zulu-openjdk:25" in line and "-jre" not in line]
    assert build_stage, from_lines


def test_applies_modeb_patch() -> None:
    assert re.search(r"git apply\b.*modeb", DOCKERFILE_TEXT) is not None


def test_patch_file_exists_on_disk() -> None:
    assert PATCH.is_file()
    patch_text = PATCH.read_text()
    assert "IncomingMessageHandler.java" in patch_text
    assert "getSourceServiceId() == null" in patch_text


def test_retains_version_smoke() -> None:
    assert re.search(r"signal-cli\s+--version", DOCKERFILE_TEXT) is not None


def test_has_revert_to_stock_marker() -> None:
    marker_lines = [
        line
        for line in DOCKERFILE_TEXT.splitlines()
        if line.lstrip().startswith("#") and "REVERT-TO-STOCK" in line.upper()
    ]
    assert marker_lines, "expected a REVERT-TO-STOCK marker comment"
    # The marker block must reference both this issue and the upstream bug.
    assert "#907" in DOCKERFILE_TEXT
    assert "#2059" in DOCKERFILE_TEXT


def test_no_active_native_tarball_download() -> None:
    native_tarball = re.compile(r"signal-cli-.*-Linux-native\.tar\.gz")
    active = [
        line
        for line in DOCKERFILE_TEXT.splitlines()
        if native_tarball.search(line) and not line.lstrip().startswith("#")
    ]
    assert not active, f"native-tarball download must be commented out, found: {active}"
