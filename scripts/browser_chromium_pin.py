#!/usr/bin/env python3
"""The aios-browser image's Chromium pin, and the check that an image ships it.

``docker/Dockerfile.browser`` is the single source of truth for the Chrome for
Testing build the image installs: the version is read out of its download URLs
(never restated here). :data:`CHROMIUM_FLOOR` is the security floor that pin —
and so every built image — must meet; raise it when a new advisory lands.

CLI (stdlib-only, run by code-validation's ``Pull browser image from GHCR``
stale-probe)::

    python3 scripts/browser_chromium_pin.py check "$(docker run ... --version)"

exits non-zero unless the reported version EQUALS the Dockerfile pin. A pulled
``:smoked`` whose protocol module matches HEAD can still ship an older Chromium
(the tag lags master's ``build-browser-image`` republish), so equality — not
"at or above the floor" — is what proves the image is the one HEAD builds.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.browser"

# Chrome stable's 2026-10-27 security release. The Dockerfile pin must be at
# or above this (unit-tested), and the contract e2e asserts it on the built
# binary.
CHROMIUM_FLOOR = (156, 0, 8078, 73)

type Version = tuple[int, ...]

_CFT_URL = re.compile(r"https://storage\.googleapis\.com/chrome-for-testing-public/([0-9.]+)/")
# ``chrome --version`` prints e.g. "Google Chrome for Testing 156.0.8078.80".
_VERSION_OUTPUT = re.compile(r"^Google Chrome for Testing (\d+(?:\.\d+){3})$")


class PinError(ValueError):
    """The Dockerfile pin or an image's reported version is unusable or wrong."""


def _parse(version: str) -> Version:
    parts = version.split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        raise PinError(f"not a four-part Chrome version: {version!r}")
    return tuple(int(p) for p in parts)


def fmt(version: Version) -> str:
    return ".".join(map(str, version))


def pinned_version(dockerfile_text: str) -> Version:
    """The Chrome for Testing version every arch's download URL pins.

    Fails if there is no CfT URL, or if the per-arch URLs disagree (a
    half-bumped pin would build different Chromium per platform)."""
    versions = set(_CFT_URL.findall(dockerfile_text))
    if not versions:
        raise PinError("no chrome-for-testing-public download URL in Dockerfile.browser")
    if len(versions) != 1:
        raise PinError(f"Dockerfile.browser pins disagreeing CfT versions: {sorted(versions)}")
    return _parse(versions.pop())


def reported_version(version_output: str) -> Version:
    """Parse ``chrome --version`` output; anything but a CfT banner fails."""
    m = _VERSION_OUTPUT.match(version_output.strip())
    if m is None:
        raise PinError(f"unrecognised chrome --version output: {version_output.strip()!r}")
    return _parse(m.group(1))


def check_image_version(version_output: str, dockerfile_text: str) -> Version:
    """Return the pin when the image reports exactly it; raise otherwise."""
    pin = pinned_version(dockerfile_text)
    got = reported_version(version_output)
    if got != pin:
        relation = "older than" if got < pin else "newer than"
        raise PinError(
            f"image Chromium {fmt(got)} is {relation} the Dockerfile.browser pin {fmt(pin)}"
        )
    return pin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pin", help="print the Dockerfile.browser Chrome for Testing pin")
    check = sub.add_parser("check", help="fail unless VERSION_OUTPUT reports exactly the pin")
    check.add_argument("version_output", help='e.g. "Google Chrome for Testing 156.0.8078.80"')
    args = parser.parse_args(argv)

    text = DOCKERFILE.read_text()
    try:
        if args.command == "pin":
            print(fmt(pinned_version(text)))
        else:
            pin = check_image_version(args.version_output, text)
            print(f"image Chromium matches the Dockerfile.browser pin {fmt(pin)}")
    except PinError as exc:
        # stdout: GHA only parses workflow commands (``::error::``) there.
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
