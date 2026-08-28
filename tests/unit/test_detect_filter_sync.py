"""Drift checks for the CI detect-filter and sandbox build trigger.

The ``run_checks`` filter in ``code-validation.yml`` gates the heavy
validation pipeline; this module asserts that it stays in sync with the
generated surface so a regen-only PR can never take the docs-only skip path:

- The detect regex matches ``openapi.json`` and every committed file under
  ``packages/aios-sdk/aios_sdk/_generated/``.  Run ``scripts/regen-client.sh``
  and add the new prefix to the regex if this test fails.
- The detect regex matches the root config/generated artifacts ``Dockerfile``,
  ``compose.yml``, and ``openapi.json``.
- The per-image ``sandbox_changed`` / ``browser_changed`` filters each match
  exactly their image's build inputs and nothing of the other's, so a PR
  touching one image never forces a rebuild of the other and never pulls a
  stale ``:smoked`` for its own.
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


def _filter_regex(output: str) -> str:
    """Extract the ERE pattern of the ``grep -qE`` that sets ``<output>=true``.

    The workflow has three ``grep -qE`` filters (``run_checks``,
    ``sandbox_changed``, ``browser_changed``).  Key on the ``<output>=true``
    that follows the pattern with no intervening ``grep -qE``, so a reordered
    or inserted grep can't silently make a test guard the wrong filter.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "code-validation.yml").read_text()
    m = re.search(rf"grep -qE '([^']*)'(?:(?!grep -qE).)*?{output}=true", workflow, re.DOTALL)
    assert m is not None, f"Could not find the {output} 'grep -qE ...' in code-validation.yml"
    return m.group(1)


def _ls_files(prefix: str, *pathspecs: str) -> list[str]:
    """Committed repo-relative POSIX paths under ``prefix`` — no __pycache__,
    no untracked/gitignored artefacts. Extra pathspecs (e.g. ``:!…/tests``)
    narrow the listing."""
    out = subprocess.run(
        ["git", "ls-files", prefix, *pathspecs],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = out.stdout.split()
    assert files, f"git ls-files found no committed files under {prefix}"
    return files


@pytest.mark.parametrize("path", ["Dockerfile", "compose.yml", "openapi.json", ".dockerignore"])
def test_new_root_paths_match_detect_filter(path: str) -> None:
    """Root generated/config artifacts must not slip through the docs-only
    skip (``.dockerignore`` is a browser-image input — ``browser_changed`` is
    dead unless ``run_checks`` also fires)."""
    pattern = _filter_regex("run_checks")
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
    pattern = _filter_regex("run_checks")
    sdk_files = _ls_files("packages/aios-sdk/aios_sdk/_generated/")
    artifact_paths = ["openapi.json", *sdk_files]
    unmatched = [p for p in artifact_paths if not re.search(pattern, p)]

    assert not unmatched, (
        f"{len(unmatched)} generated artifact(s) not matched by the detect-filter regex "
        f"in code-validation.yml.  First 10: {unmatched[:10]}\n"
        f"Pattern: {pattern!r}"
    )


def test_sandbox_image_filter_matches_exactly_its_inputs() -> None:
    """``sandbox_changed`` gates pull-``:smoked``-vs-rebuild for the SANDBOX
    image; it must match exactly that image's build inputs
    (``docker/Dockerfile.sandbox`` + the COPYed ``bin/tool``) and must NOT
    match the browser image's inputs — a browser-only PR pulling the
    pre-built sandbox image is the point of the tightening."""
    pattern = _filter_regex("sandbox_changed")
    for path in ("docker/Dockerfile.sandbox", "bin/tool"):
        assert re.search(pattern, path), f"sandbox input {path!r} escapes the sandbox filter"
    for path in ("docker/Dockerfile.browser", "docker/seccomp-browser.json", ".dockerignore"):
        assert not re.search(pattern, path), (
            f"{path!r} is not a sandbox-image input but matches the sandbox filter — "
            "it would force a pointless sandbox rebuild"
        )


def test_browser_image_filter_matches_exactly_its_inputs() -> None:
    """``browser_changed`` gates pull-``:smoked``-vs-rebuild for the BROWSER
    image; it must match every build input of ``docker/Dockerfile.browser`` —
    the Dockerfile itself, every committed file of the driver package that
    reaches the build context (the directory is COPYed but ``tests/`` is
    ``.dockerignore``d out of it), the protocol module COPYed over the
    package's vendored copy, and ``.dockerignore`` (it shapes that directory
    COPY) — and must match neither sandbox inputs nor the context-excluded
    driver tests (either would force a pointless ~10 min browser rebuild)."""
    pattern = _filter_regex("browser_changed")

    shipped = _ls_files("packages/aios-browser-driver/", ":!packages/aios-browser-driver/tests")
    inputs = [
        "docker/Dockerfile.browser",
        "src/aios/sandbox/browser_protocol.py",
        ".dockerignore",
        *shipped,
    ]
    unmatched = [p for p in inputs if not re.search(pattern, p)]
    assert not unmatched, (
        f"browser-image input(s) escape the browser filter — a stale ``:smoked`` would be "
        f"pulled for a PR that changes them.  First 10: {unmatched[:10]}\nPattern: {pattern!r}"
    )

    # The exclusion is only sound while .dockerignore actually keeps the
    # driver tests out of the build context — pin that line alongside.
    dockerignore = (_REPO_ROOT / ".dockerignore").read_text()
    assert re.search(r"(?m)^packages/aios-browser-driver/tests$", dockerignore), (
        ".dockerignore no longer excludes packages/aios-browser-driver/tests — the "
        "browser_changed filter's tests/ exclusion is now unsound; re-align both"
    )
    driver_tests = _ls_files("packages/aios-browser-driver/tests/")
    non_inputs = ["docker/Dockerfile.sandbox", "bin/tool", *driver_tests]
    over_matched = [p for p in non_inputs if re.search(pattern, p)]
    assert not over_matched, (
        f"path(s) that cannot change the browser image match the browser filter — "
        f"pointless ~10 min rebuilds.  First 10: {over_matched[:10]}\nPattern: {pattern!r}"
    )


def test_image_filter_inputs_all_reach_run_checks() -> None:
    """Every path either image filter fires on must also fire ``run_checks``:
    the pull/build/e2e steps are all ANDed with ``run_checks == 'true'``, so
    an image input outside the outer gate would compute a dead
    ``*_changed=true`` while the whole pipeline takes the docs-only skip."""
    run_checks = _filter_regex("run_checks")
    inputs = [
        "docker/Dockerfile.sandbox",
        "bin/tool",
        "docker/Dockerfile.browser",
        "src/aios/sandbox/browser_protocol.py",
        ".dockerignore",
        *_ls_files("packages/aios-browser-driver/", ":!packages/aios-browser-driver/tests"),
    ]
    escapees = [p for p in inputs if not re.search(run_checks, p)]
    assert not escapees, (
        f"image-filter input(s) do not match run_checks — their *_changed output is dead "
        f"because every consumer step requires run_checks == 'true'.  {escapees}"
    )


def test_build_browser_image_triggers_on_its_inputs() -> None:
    """build-browser-image.yml's ``on.push.paths`` must list every browser
    input (and the negated driver-tests glob, mirroring the CI filter): a
    master push touching an input the trigger misses leaves ``:smoked``
    silently stale — the sandbox precedent is ``bin/tool`` below."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "build-browser-image.yml").read_text()
    for entry in (
        "docker/Dockerfile.browser",
        "docker/seccomp-browser.json",
        "docker/seccomp-sandbox.json",
        ".dockerignore",
        "packages/aios-browser-driver/**",
        '"!packages/aios-browser-driver/tests/**"',
        "src/aios/sandbox/browser_protocol.py",
        "tests/e2e/test_browser_image_contract.py",
        "tests/e2e/test_browser_driver_behavior.py",
        "tests/e2e/browser_image_common.py",
    ):
        assert re.search(rf"(?m)^\s*-\s*{re.escape(entry)}\s*$", workflow), (
            f"{entry!r} is not listed in the on.push.paths trigger of "
            f"build-browser-image.yml; a push touching it would leave :smoked stale"
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
