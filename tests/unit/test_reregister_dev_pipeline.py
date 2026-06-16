"""Unit tests for the IaC-sync v1 re-register script (issue #1179).

The script's load-bearing logic is the pure pipeline: extract the configurable
constants from the LIVE script header (drift-safe), rebuild from the updated
builder, validate, and decide whether to PUT (idempotent). These tests exercise
that logic offline — no network, no live workflow — plus the drift guard against
the workflow YAML and the builder signature.
"""

from __future__ import annotations

import importlib.util
import re
import urllib.error
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "reregister_dev_pipeline.py"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "reregister-dev-pipeline.yml"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reregister_dev_pipeline", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_script_module()


# ─── extract_constants: drift-safe header parse ──────────────────────────────


def test_extract_constants_roundtrips_default_builder() -> None:
    """Constants extracted from a freshly-built script feed the builder back to
    produce a byte-identical script — the round-trip the Action depends on."""
    script = build_dev_pipeline_script()
    constants = rr.extract_constants(script)
    # Every builder kwarg is recovered.
    assert set(constants) == set(rr.CONSTANT_TO_KWARG.values())
    rebuilt = build_dev_pipeline_script(**constants)
    assert rebuilt == script


def test_extract_constants_preserves_non_default_prod_config() -> None:
    """A live deployment with custom agent ids / sentinels / merge tier must
    survive the extract → rebuild round-trip unchanged (drift-safe)."""
    script = build_dev_pipeline_script(
        implement_agent_id="ag_custom_impl",
        review_agent_id="ag_custom_review",
        merge_sentinels=["pytest -q", "ruff check src"],
        auto_merge_max_tier=3,
        merge_method="merge",
        base_branch="main",
    )
    constants = rr.extract_constants(script)
    assert constants["implement_agent_id"] == "ag_custom_impl"
    assert constants["merge_sentinels"] == ["pytest -q", "ruff check src"]
    assert constants["auto_merge_max_tier"] == 3
    assert constants["merge_method"] == "merge"
    assert constants["base_branch"] == "main"
    assert build_dev_pipeline_script(**constants) == script


def test_extract_constants_is_robust_to_header_reordering() -> None:
    """Extraction keys on assignment NAME via ast, not line position, so a
    reordered header still yields the right constants."""
    script = build_dev_pipeline_script(base_branch="trunk")
    lines = script.splitlines()
    # Reverse the first 13 (constant) header lines; the body is unaffected.
    reordered = "\n".join(lines[:13][::-1] + lines[13:])
    constants = rr.extract_constants(reordered)
    assert constants["base_branch"] == "trunk"


def test_extract_constants_missing_constant_is_loud() -> None:
    """A header that dropped a builder constant must fail loudly, not silently
    re-deploy with a builder default."""
    script = build_dev_pipeline_script()
    stripped = "\n".join(ln for ln in script.splitlines() if not ln.startswith("MERGE_METHOD = "))
    with pytest.raises(rr.ReregisterError):
        rr.extract_constants(stripped)


# ─── validate / needs_update ─────────────────────────────────────────────────


def test_validate_script_accepts_real_builder_output() -> None:
    rr.validate_script(build_dev_pipeline_script())


def test_validate_script_rejects_syntax_error() -> None:
    with pytest.raises(rr.ReregisterError):
        rr.validate_script("def broken(:\n    pass\n")


def test_needs_update_idempotent_on_identical() -> None:
    script = build_dev_pipeline_script()
    assert rr.needs_update(script, script) is False


def test_needs_update_true_on_change() -> None:
    a = build_dev_pipeline_script(base_branch="master")
    b = build_dev_pipeline_script(base_branch="main")
    assert rr.needs_update(b, a) is True


def test_regenerate_from_constants_matches_direct_build() -> None:
    constants = rr.extract_constants(build_dev_pipeline_script(merge_method="rebase"))
    assert rr.regenerate_script(constants) == build_dev_pipeline_script(merge_method="rebase")


# ─── builder-signature drift guard ───────────────────────────────────────────


def test_constant_map_covers_every_builder_kwarg() -> None:
    """If a new configurable kwarg is added to ``build_dev_pipeline_script`` it
    MUST be added to CONSTANT_TO_KWARG (and echoed as a header constant), or the
    re-register would drop it and silently re-deploy with a builder default.
    Assert the map equals the builder's keyword-only parameters exactly."""
    import inspect

    sig = inspect.signature(build_dev_pipeline_script)
    kwargs = {
        name for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert set(rr.CONSTANT_TO_KWARG.values()) == kwargs, (
        "CONSTANT_TO_KWARG is out of sync with build_dev_pipeline_script's "
        "keyword-only parameters; update scripts/reregister_dev_pipeline.py "
        "(and the header constants in dev_pipeline.py if a new constant was added)."
    )


# ─── GitHub Action drift guards ──────────────────────────────────────────────


def test_workflow_triggers_only_on_builder_path() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "branches: [master]" in text
    assert "src/aios/workflows/dev_pipeline.py" in text
    # paths-filtered so unrelated master pushes don't fire.
    assert re.search(r"paths:\s*\n\s*-\s*\"src/aios/workflows/dev_pipeline\.py\"", text)


def test_workflow_references_secret_and_env_config() -> None:
    text = _WORKFLOW_PATH.read_text()
    # Key referenced via secrets, never inlined.
    assert "${{ secrets.AIOS_API_KEY }}" in text
    assert "AIOS_URL: https://api.aios.eumemic.ai" in text
    assert "DEV_PIPELINE_WORKFLOW_ID: wf_01KV4YGV4PSGP08TJBAY32J2VK" in text


def test_workflow_runs_fixture_validation_before_reregister() -> None:
    """The fixture-drive validation step must precede the re-register PUT step."""
    text = _WORKFLOW_PATH.read_text()
    validate_at = text.find("test_wf_dev_pipeline_fixture.py")
    reregister_at = text.find("reregister_dev_pipeline.py")
    assert 0 < validate_at < reregister_at


def test_no_hardcoded_api_key() -> None:
    """Defence in depth: no inline key anywhere in the Action or script."""
    for path in (_WORKFLOW_PATH, _SCRIPT_PATH):
        text = path.read_text()
        assert not re.search(r"AIOS_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", text)


# ─── #1183: least-privilege + hardening guards ───────────────────────────────


def test_workflow_declares_least_privilege_permissions() -> None:
    """The job performs NO GitHub writes, so the GITHUB_TOKEN must be read-only
    (#1183 item 1): a compromised freshly-resolved dep must not get a write token
    for free next to the prod-mutating AIOS_API_KEY."""
    text = _WORKFLOW_PATH.read_text()
    assert re.search(r"permissions:\s*\n\s*contents:\s*read", text), (
        "workflow must declare least-privilege `permissions: contents: read`"
    )


def test_api_key_is_env_only_no_cli_flag() -> None:
    """The API key must be read from os.environ only — no --api-key argv path
    (#1183 item 2): an argv flag is a /proc/<pid>/cmdline exposure footgun."""
    text = _SCRIPT_PATH.read_text()
    assert "--api-key" not in text, "the --api-key CLI flag must be removed (env-only)"
    assert 'os.environ.get("AIOS_API_KEY")' in text or 'os.environ["AIOS_API_KEY"]' in text


def test_request_catches_urlerror_cleanly() -> None:
    """A DNS/connection/timeout failure (URLError) must surface as the clean
    ReregisterError path, not an uncaught traceback (#1183 item 4)."""
    with (
        mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("name resolution failed"),
        ),
        pytest.raises(rr.ReregisterError),
    ):
        rr._request("GET", "https://example.invalid/v1/workflows/wf_x", "k")


def test_request_passes_timeout_to_urlopen() -> None:
    """urlopen must be called with a timeout so a hung connection fails fast
    rather than waiting out the GitHub job timeout (#1183 item 5)."""
    captured: dict[str, object] = {}

    class _Resp:
        status = 200

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_urlopen(req: object, *args: object, **kwargs: object) -> _Resp:
        captured.update(kwargs)
        if args:
            captured["positional_timeout"] = args[0]
        return _Resp()

    with mock.patch("urllib.request.urlopen", _fake_urlopen):
        rr._request("GET", "https://example.test/v1/workflows/wf_x", "k")

    timeout = captured.get("timeout", captured.get("positional_timeout"))
    assert isinstance(timeout, (int, float)) and timeout > 0, (
        "urlopen must be called with a positive timeout"
    )


def test_constants_print_is_verbose_gated() -> None:
    """The 'Extracted live constants' dump of prod workflow config must be
    trimmed/gated so it does not unconditionally write agent ids / sentinels /
    tiers / model to CI logs (#1183 item 3)."""
    text = _SCRIPT_PATH.read_text()
    if "Extracted live constants" in text:
        # If the verbose dump remains, it must be gated behind a --verbose flag.
        assert "--verbose" in text or "args.verbose" in text, (
            "the 'Extracted live constants' dump must be --verbose-gated, not "
            "unconditionally printed to CI logs"
        )


def test_concurrency_comment_is_accurate() -> None:
    """The concurrency rationale must not claim stacked merges race the
    optimistic-version PUT (the loser hits 409); the actual mechanism is GitHub
    cancelling the older PENDING run so the tip wins (#1183 item 6)."""
    text = _WORKFLOW_PATH.read_text()
    assert "race the optimistic-version PUT" not in text
    assert "loser hits a 409" not in text
