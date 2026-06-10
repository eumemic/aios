"""Drift checks for the CI detect-filter and sandbox build trigger.

Three invariants are asserted:

1. The ``code-validation.yml`` detect-step regex matches root generated and
   config artifacts (``Dockerfile``, ``compose.yml``, ``openapi.json``) so
   PRs that touch only those files still run CI.

2. Every committed file that is a generated artifact (``openapi.json`` plus
   the entire ``packages/aios-sdk/aios_sdk/_generated/`` tree) is matched by
   that same regex — so future additions to the generated surface don't silently
   gain a docs-only skip path.  Run ``scripts/regen-client.sh`` and add the new
   prefix to the detect-filter regex if this test fails.

3. ``.github/workflows/build-sandbox.yml`` triggers on ``bin/tool`` changes,
   because ``docker/Dockerfile.sandbox`` COPYs that binary into the image — a
   tool-only master push must rebuild the image, not silently skip it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _detect_regex() -> str:
    """Extract the ERE pattern from the grep -qE line in code-validation.yml."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    m = re.search(r"grep -qE '([^']*)'", workflow)
    assert m is not None, "Could not find 'grep -qE ...' line in code-validation.yml"
    return m.group(1)


@pytest.mark.parametrize("path", ["Dockerfile", "compose.yml", "openapi.json"])
def test_new_root_paths_match_detect_filter(path: str) -> None:
    """Root generated/config artifacts must not slip through the docs-only skip."""
    pattern = _detect_regex()
    assert re.search(pattern, path), (
        f"{path!r} does not match the detect-filter regex in code-validation.yml;\n"
        f"add it to the grep -qE alternation so PRs touching only this file still run CI.\n"
        f"Pattern: {pattern!r}"
    )


def test_detect_filter_matches_all_generated_artifacts() -> None:
    """Every committed generated artifact must be matched by the detect-filter.

    The detect regex in code-validation.yml must cover openapi.json and all
    files under packages/aios-sdk/aios_sdk/_generated/ so a PR that regenerates
    any of them always triggers CI.  If this test fails, add the new path prefix
    to the grep -qE alternation in code-validation.yml.
    """
    pattern = _detect_regex()

    generated_dir = _REPO_ROOT / "packages" / "aios-sdk" / "aios_sdk" / "_generated"
    # Only committed files — exclude __pycache__/*.pyc and other untracked artefacts.
    artifact_paths = ["openapi.json"] + [
        p.relative_to(_REPO_ROOT).as_posix()
        for p in generated_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]

    unmatched = [p for p in artifact_paths if not re.search(pattern, p)]

    assert not unmatched, (
        f"{len(unmatched)} generated artifact(s) not matched by the detect-filter regex "
        f"in code-validation.yml.  First 10: {unmatched[:10]}\n"
        f"Pattern: {pattern!r}"
    )


def test_build_sandbox_triggers_on_bin_tool() -> None:
    """build-sandbox.yml must list bin/tool in its on.push.paths trigger.

    docker/Dockerfile.sandbox COPYs bin/tool into the image at build time.
    A master push that updates only bin/tool must therefore trigger a sandbox
    rebuild — otherwise the published image silently ships a stale binary.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "build-sandbox.yml").read_text()
    assert re.search(r"(?m)^\s*-\s*bin/tool\s*$", workflow), (
        "bin/tool is not listed in the on.push.paths trigger of build-sandbox.yml.\n"
        "Add '- bin/tool' to the paths list so a tool-only push rebuilds the sandbox image.\n"
        "(docker/Dockerfile.sandbox line 63: COPY bin/tool /usr/local/bin/tool)"
    )
