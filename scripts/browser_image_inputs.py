#!/usr/bin/env python3
"""A digest over every build input of the aios-browser image.

The ``:smoked`` stale-probe in code-validation's ``Pull browser image from
GHCR`` step originally compared only the installed ``browser_protocol.py``
(and, since #2571, the Chromium version). Any OTHER image change — a driver
module, ``requirements.lock``, an apt/base-image edit in
``docker/Dockerfile.browser`` — still let a stale ``:smoked`` pass the probe
until master's ``build-browser-image`` republished (#2570).

So ``build-browser-image.yml`` stamps the image with :data:`LABEL` = the
:func:`inputs_digest` of the commit it built, and the probe recomputes the
digest from the PR checkout and requires the pulled image's label to EQUAL it.
Any input drift (or an unlabeled image) fails closed into the local rebuild.

The input set mirrors what ``docker build -f docker/Dockerfile.browser .``
can see and use: the Dockerfile itself, ``.dockerignore`` (it shapes the
context), the ``src/`` protocol module COPYed over the vendored copy, and
every committed file of the driver package except its ``.dockerignore``d
``tests/``. ``tests/unit/test_detect_filter_sync.py`` pins this set against
the ``browser_changed`` CI filter so the two cannot drift apart.

CLI (stdlib-only)::

    python3 scripts/browser_image_inputs.py digest          # print the digest
    python3 scripts/browser_image_inputs.py check "$(docker image inspect IMG \
        --format '{{json .Config.Labels}}')"                 # exit 1 unless equal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# OCI label the publish workflow stamps on the image.
LABEL = "io.eumemic.aios.browser-inputs-digest"

_FIXED_INPUTS = (
    "docker/Dockerfile.browser",
    ".dockerignore",
    "src/aios/sandbox/browser_protocol.py",
)
_DRIVER_DIR = "packages/aios-browser-driver/"
_DRIVER_EXCLUDE = ":!packages/aios-browser-driver/tests"


class DigestError(ValueError):
    """The image's label does not match the checkout's inputs digest."""


def input_paths(repo_root: Path = REPO_ROOT) -> list[str]:
    """Sorted repo-relative POSIX paths of every browser-image build input.

    Driver files come from ``git ls-files`` so untracked artefacts
    (``__pycache__``, editor droppings) never perturb the digest."""
    out = subprocess.run(
        ["git", "ls-files", "-z", _DRIVER_DIR, _DRIVER_EXCLUDE],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    driver = [p.decode() for p in out.stdout.split(b"\0") if p]
    if not driver:
        raise DigestError(f"git ls-files found no committed files under {_DRIVER_DIR}")
    return sorted({*_FIXED_INPUTS, *driver})


def inputs_digest(repo_root: Path = REPO_ROOT) -> str:
    """``sha256:<hex>`` over (path, content) of every input, length-framed so
    no rename/concatenation can collide with a different tree."""
    h = hashlib.sha256()
    for rel in input_paths(repo_root):
        data = (repo_root / rel).read_bytes()
        name = rel.encode()
        h.update(len(name).to_bytes(8, "big") + name)
        h.update(len(data).to_bytes(8, "big") + data)
    return f"sha256:{h.hexdigest()}"


def check_labels(labels_json: str, expected: str) -> None:
    """Raise unless the image's :data:`LABEL` (from ``docker image inspect
    --format '{{json .Config.Labels}}'``) equals the checkout's digest."""
    try:
        labels = json.loads(labels_json)
    except json.JSONDecodeError as exc:
        raise DigestError(f"unparseable image labels {labels_json.strip()!r}: {exc}") from exc
    if labels is not None and not isinstance(labels, dict):
        raise DigestError(f"image labels are not an object: {labels_json.strip()!r}")
    got = (labels or {}).get(LABEL)
    if not got:
        raise DigestError(
            f"image has no {LABEL} label (published before #2570) — cannot prove it matches HEAD"
        )
    if got != expected:
        raise DigestError(
            f"image {LABEL} {got} differs from the checkout's browser-image inputs {expected}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("digest", help="print the checkout's browser-image inputs digest")
    sub.add_parser("label", help="print the OCI label key the digest is stamped under")
    check = sub.add_parser("check", help="fail unless the image's label equals the inputs digest")
    check.add_argument("labels_json", help="docker image inspect --format '{{json .Config.Labels}}'")
    args = parser.parse_args(argv)

    try:
        if args.command == "label":
            print(LABEL)
        elif args.command == "digest":
            print(inputs_digest())
        else:
            expected = inputs_digest()
            check_labels(args.labels_json, expected)
            print(f"image browser-image inputs digest matches the checkout {expected}")
    except DigestError as exc:
        # stdout: GHA only parses workflow commands (``::error::``) there.
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
