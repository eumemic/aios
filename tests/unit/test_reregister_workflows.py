"""Unit tests for the IaC-sync v1 re-register script (issue #1179, generalised #1226).

The script's load-bearing logic is the pure, per-target pipeline: extract the
configurable constants from the LIVE script header (drift-safe), rebuild from the
updated builder, validate, and decide whether to PUT (idempotent). These tests
exercise that logic offline — no network, no live workflow — plus the drift guard
against the workflow YAML and each builder signature, AND the #1226 generalisation
to a LIST of workflow targets.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script
from aios.workflows.triage_pipeline import build_triage_pipeline_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "reregister_workflows.py"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "reregister-workflows.yml"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reregister_workflows", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines a @dataclass whose fields reference
    # typing names, and dataclasses resolves them via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rr = _load_script_module()


def _dev_target() -> object:
    return next(t for t in rr._targets() if t.key == "dev-pipeline")


def _triage_target() -> object:
    return next(t for t in rr._targets() if t.key == "triage-pipeline")


# ─── extract_constants: drift-safe header parse ──────────────────────────────


def test_extract_constants_roundtrips_default_builder() -> None:
    """Constants extracted from a freshly-built script feed the builder back to
    produce a byte-identical script — the round-trip the Action depends on."""
    script = build_dev_pipeline_script()
    m = rr.DEV_PIPELINE_CONSTANT_TO_KWARG
    constants = rr.extract_constants(script, m)
    assert set(constants) == set(m.values())
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
    constants = rr.extract_constants(script, rr.DEV_PIPELINE_CONSTANT_TO_KWARG)
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
    reordered = "\n".join(lines[:13][::-1] + lines[13:])
    constants = rr.extract_constants(reordered, rr.DEV_PIPELINE_CONSTANT_TO_KWARG)
    assert constants["base_branch"] == "trunk"


def test_extract_constants_missing_constant_is_loud() -> None:
    """A header that dropped a builder constant must fail loudly, not silently
    re-deploy with a builder default."""
    script = build_dev_pipeline_script()
    stripped = "\n".join(ln for ln in script.splitlines() if not ln.startswith("MERGE_METHOD = "))
    with pytest.raises(rr.ReregisterError):
        rr.extract_constants(stripped, rr.DEV_PIPELINE_CONSTANT_TO_KWARG)


# ─── triage-pipeline target (the #1226 generalisation) ───────────────────────


def test_triage_extract_constants_roundtrips() -> None:
    script = build_triage_pipeline_script()
    m = rr.TRIAGE_PIPELINE_CONSTANT_TO_KWARG
    constants = rr.extract_constants(script, m)
    assert set(constants) == set(m.values())
    assert build_triage_pipeline_script(**constants) == script


def test_triage_extract_preserves_non_default_config() -> None:
    script = build_triage_pipeline_script(
        triage_agent_id="ag_triage_custom", repo="o/r", max_issues_per_run=10
    )
    constants = rr.extract_constants(script, rr.TRIAGE_PIPELINE_CONSTANT_TO_KWARG)
    assert constants["triage_agent_id"] == "ag_triage_custom"
    assert constants["repo"] == "o/r"
    assert constants["max_issues_per_run"] == 10
    assert build_triage_pipeline_script(**constants) == script


def test_targets_registry_has_both_pipelines() -> None:
    keys = {t.key for t in rr._targets()}
    assert keys == {"dev-pipeline", "triage-pipeline"}


def test_targets_have_distinct_id_env_vars() -> None:
    targets = rr._targets()
    envs = [t.id_env for t in targets]
    assert len(envs) == len(set(envs))
    assert _dev_target().id_env == "DEV_PIPELINE_WORKFLOW_ID"
    assert _triage_target().id_env == "TRIAGE_PIPELINE_WORKFLOW_ID"


def test_regenerate_script_uses_the_targets_builder() -> None:
    triage = _triage_target()
    constants = rr.extract_constants(
        build_triage_pipeline_script(max_issues_per_run=3), triage.constant_to_kwarg
    )
    assert rr.regenerate_script(constants, triage.builder) == build_triage_pipeline_script(
        max_issues_per_run=3
    )


# ─── validate / needs_update ─────────────────────────────────────────────────


def test_validate_script_accepts_real_builder_output() -> None:
    rr.validate_script(build_dev_pipeline_script())
    rr.validate_script(build_triage_pipeline_script())


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
    dev = _dev_target()
    constants = rr.extract_constants(
        build_dev_pipeline_script(merge_method="rebase"), dev.constant_to_kwarg
    )
    assert rr.regenerate_script(constants, dev.builder) == build_dev_pipeline_script(
        merge_method="rebase"
    )


# ─── builder-signature drift guards (one per target) ─────────────────────────


def test_dev_constant_map_covers_every_builder_kwarg() -> None:
    sig = inspect.signature(build_dev_pipeline_script)
    kwargs = {
        name for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert set(rr.DEV_PIPELINE_CONSTANT_TO_KWARG.values()) == kwargs, (
        "DEV_PIPELINE_CONSTANT_TO_KWARG is out of sync with build_dev_pipeline_script's "
        "keyword-only parameters; update scripts/reregister_workflows.py."
    )


def test_triage_constant_map_covers_every_builder_kwarg() -> None:
    sig = inspect.signature(build_triage_pipeline_script)
    kwargs = {
        name for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert set(rr.TRIAGE_PIPELINE_CONSTANT_TO_KWARG.values()) == kwargs, (
        "TRIAGE_PIPELINE_CONSTANT_TO_KWARG is out of sync with build_triage_pipeline_script's "
        "keyword-only parameters; update scripts/reregister_workflows.py."
    )


# ─── GitHub Action drift guards ──────────────────────────────────────────────


def test_workflow_triggers_on_both_builder_paths() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "branches: [master]" in text
    # paths-filtered so unrelated master pushes don't fire — both builders listed.
    assert re.search(r"paths:\s*\n\s*-\s*\"src/aios/workflows/dev_pipeline\.py\"", text)
    assert 'src/aios/workflows/triage_pipeline.py' in text


def test_workflow_references_secret_and_env_config() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "${{ secrets.AIOS_API_KEY }}" in text
    assert "AIOS_URL: https://api.aios.eumemic.ai" in text
    assert "DEV_PIPELINE_WORKFLOW_ID: wf_01KV4YGV4PSGP08TJBAY32J2VK" in text
    # the triage target's id env is wired (sourced from a repo var until provisioned)
    assert "TRIAGE_PIPELINE_WORKFLOW_ID" in text


def test_workflow_runs_fixture_validation_before_reregister() -> None:
    """The fixture-drive validation step must precede the re-register PUT step,
    and must validate BOTH pipelines."""
    text = _WORKFLOW_PATH.read_text()
    dev_validate_at = text.find("test_wf_dev_pipeline_fixture.py")
    triage_validate_at = text.find("test_wf_triage_pipeline_fixture.py")
    # match the run-step invocation, not the leading comment that names the script
    reregister_at = text.find("python scripts/reregister_workflows.py")
    assert 0 < dev_validate_at < reregister_at
    assert 0 < triage_validate_at < reregister_at


def test_no_hardcoded_api_key() -> None:
    for path in (_WORKFLOW_PATH, _SCRIPT_PATH):
        text = path.read_text()
        assert not re.search(r"AIOS_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", text)


# ─── #1183: least-privilege + hardening guards (preserved) ───────────────────


def test_workflow_declares_least_privilege_permissions() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert re.search(r"permissions:\s*\n\s*contents:\s*read", text), (
        "workflow must declare least-privilege `permissions: contents: read`"
    )


def test_api_key_is_env_only_no_cli_flag() -> None:
    text = _SCRIPT_PATH.read_text()
    assert "--api-key" not in text, "the --api-key CLI flag must be removed (env-only)"
    assert 'os.environ.get("AIOS_API_KEY")' in text or 'os.environ["AIOS_API_KEY"]' in text


def test_request_catches_urlerror_cleanly() -> None:
    with (
        mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("name resolution failed"),
        ),
        pytest.raises(rr.ReregisterError),
    ):
        rr._request("GET", "https://example.invalid/v1/workflows/wf_x", "k")


def test_request_passes_timeout_to_urlopen() -> None:
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
    text = _SCRIPT_PATH.read_text()
    if "extracted live constants" in text.lower():
        assert "--verbose" in text or "args.verbose" in text, (
            "the extracted-constants dump must be --verbose-gated, not unconditionally printed"
        )


def test_concurrency_comment_is_accurate() -> None:
    text = _WORKFLOW_PATH.read_text()
    assert "race the optimistic-version PUT" not in text
    assert "loser hits a 409" not in text


# ─── main() target loop (the generalisation, exercised offline) ──────────────


def test_main_skips_target_with_unset_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the dev id is set → triage is skipped, dev reconciles. reconcile_target is
    # stubbed so no network is touched.
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://example.test")
    monkeypatch.setenv("DEV_PIPELINE_WORKFLOW_ID", "wf_dev")
    monkeypatch.delenv("TRIAGE_PIPELINE_WORKFLOW_ID", raising=False)
    seen: list[str] = []
    with mock.patch.object(rr, "reconcile_target", lambda t, **k: seen.append(t.key) or False):
        rc = rr.main([])
    assert rc == 0
    assert seen == ["dev-pipeline"]  # triage skipped (no id)


def test_main_reconciles_all_configured_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://example.test")
    monkeypatch.setenv("DEV_PIPELINE_WORKFLOW_ID", "wf_dev")
    monkeypatch.setenv("TRIAGE_PIPELINE_WORKFLOW_ID", "wf_triage")
    seen: list[str] = []
    with mock.patch.object(rr, "reconcile_target", lambda t, **k: seen.append(t.key) or True):
        rc = rr.main([])
    assert rc == 0
    assert set(seen) == {"dev-pipeline", "triage-pipeline"}


def test_main_aborts_when_a_configured_target_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://example.test")
    monkeypatch.setenv("DEV_PIPELINE_WORKFLOW_ID", "wf_dev")
    monkeypatch.setenv("TRIAGE_PIPELINE_WORKFLOW_ID", "wf_triage")

    def _boom(target: object, **kwargs: object) -> bool:
        raise rr.ReregisterError("kaboom")

    with mock.patch.object(rr, "reconcile_target", _boom):
        rc = rr.main([])
    assert rc == 1  # a configured target's failure is a loud, non-zero exit


def test_main_only_filters_to_named_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_API_KEY", "k")
    monkeypatch.setenv("AIOS_URL", "https://example.test")
    monkeypatch.setenv("DEV_PIPELINE_WORKFLOW_ID", "wf_dev")
    monkeypatch.setenv("TRIAGE_PIPELINE_WORKFLOW_ID", "wf_triage")
    seen: list[str] = []
    with mock.patch.object(rr, "reconcile_target", lambda t, **k: seen.append(t.key) or False):
        rc = rr.main(["--only", "triage-pipeline"])
    assert rc == 0
    assert seen == ["triage-pipeline"]


def test_main_missing_api_key_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_API_KEY", raising=False)
    monkeypatch.setenv("AIOS_URL", "https://example.test")
    assert rr.main([]) == 2
