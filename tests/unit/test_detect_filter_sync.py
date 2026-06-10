"""Drift checks for the CI detect-filter and sandbox build trigger.

The ``run_checks`` filter in ``code-validation.yml`` gates the heavy
validation pipeline; this module asserts that it stays in sync with the
generated surface so a regen-only PR can never take the docs-only skip path:

- The detect regex matches ``openapi.json`` and every committed file under
  ``packages/aios-sdk/aios_sdk/_generated/``.  Run ``scripts/regen-client.sh``
  and add the new prefix to the regex if this test fails.
- The detect regex matches the root config/generated artifacts ``Dockerfile``,
  ``compose.yml``, and ``openapi.json``.
- ``.github/workflows/build-sandbox.yml`` triggers on ``bin/tool`` changes,
  because ``docker/Dockerfile.sandbox`` COPYs that binary into the image — a
  tool-only master push must rebuild the image, not silently skip it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _detect_regex() -> str:
    """Extract the ERE pattern from the ``run_checks`` grep -qE in code-validation.yml.

    The workflow has two ``grep -qE`` lines — the ``run_checks`` filter and a
    later ``sandbox_changed`` filter.  Key on the ``run_checks=true`` that
    follows the pattern (with no intervening ``grep -qE``) so a reordered or
    inserted grep can't silently make this test guard the wrong filter.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    m = re.search(r"grep -qE '([^']*)'(?:(?!grep -qE).)*?run_checks=true", workflow, re.DOTALL)
    assert m is not None, "Could not find the run_checks 'grep -qE ...' in code-validation.yml"
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

    # ``git ls-files`` gives exactly the committed files (repo-relative POSIX
    # paths) — no __pycache__, no untracked/gitignored artefacts.
    out = subprocess.run(
        ["git", "ls-files", "packages/aios-sdk/aios_sdk/_generated/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    sdk_files = out.stdout.split()
    assert sdk_files, "git ls-files found no committed files under aios_sdk/_generated/"

    artifact_paths = ["openapi.json", *sdk_files]
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
