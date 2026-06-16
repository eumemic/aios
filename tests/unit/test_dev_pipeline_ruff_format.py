"""Unit tests for the dev-pipeline ruff-format lint-parity fix (#1182).

PR #1173 parked at the verify gate after the CI-fix loop exhausted all 3 iterations on a
``ruff format --check`` failure: the GitHub CI ``lint`` job runs BOTH ``ruff check`` AND
``ruff format --check``, but the CI-fix path only ran ``ruff check --fix`` (which does NOT
fix formatting) and the merge sentinel only ran ``ruff check`` (so its lint was WEAKER than
CI). A single ``ruff format`` would have resolved it, yet the pipeline could not self-heal.

Two in-script fixes, both exercised here by ``exec``-ing the production script body and
pulling the (otherwise un-importable) helpers / task payload out of a fresh namespace:

1. ``_merge_guard_command`` derives a matching ``ruff format --check <same targets>`` for
   every ``ruff check <targets>`` sentinel, so the merge-guard lint matches the CI lint job
   exactly (one source of lint truth) regardless of the deployed ``MERGE_SENTINELS`` config.
2. The ``fix_ci`` task payload carries an explicit instruction that the CI lint job runs
   ``ruff format --check`` and a format-only failure MUST be fixed by running ``ruff
   format`` (not only ``ruff check --fix``), so a format-only failure is auto-fixable.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script


def _ns(**kwargs: Any) -> dict[str, Any]:
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
        **kwargs,
    )
    namespace: dict[str, Any] = {}
    exec(compile(src, "dev_pipeline_script", "exec"), namespace)
    return namespace


# ─── part 2: merge-sentinel lint parity ──────────────────────────────────────


def test_merge_guard_derives_ruff_format_check_for_ruff_check_sentinel() -> None:
    """A ``ruff check <targets>`` sentinel gets a matching ``ruff format --check
    <targets>`` appended in the merge guard, so the guard's lint matches CI's lint job."""
    ns = _ns(merge_sentinels=["pytest -q", "ruff check src tests"])
    cmd = ns["_merge_guard_command"]("owner/repo", 42)

    # The original check sentinel still runs ...
    assert "ruff check src tests" in cmd
    # ... and a derived format --check over the SAME targets runs too.
    assert "ruff format --check src tests" in cmd


def test_merge_guard_format_check_preserves_full_target_set() -> None:
    """The derived format check carries the EXACT target list of the check sentinel —
    matching CI's wide target set (src tests packages/... connectors/...)."""
    targets = (
        "src tests packages/aios-sdk/aios_sdk connectors/signal/src "
        "connectors/telegram/src connectors/whatsapp/src "
        "packages/aios-connector-http/aios_connector_http"
    )
    ns = _ns(merge_sentinels=[f"ruff check {targets}"])
    cmd = ns["_merge_guard_command"]("owner/repo", 7)
    assert f"ruff format --check {targets}" in cmd


def test_merge_guard_no_double_format_check_when_already_present() -> None:
    """An already format-aware sentinel set is not duplicated: a config that already
    carries ``ruff format --check`` does not get a second derived copy."""
    ns = _ns(merge_sentinels=["ruff check src tests", "ruff format --check src tests"])
    cmd = ns["_merge_guard_command"]("owner/repo", 9)
    assert cmd.count("ruff format --check src tests") == 1


def test_merge_guard_ignores_non_ruff_check_sentinels() -> None:
    """Non-ruff sentinels (pytest, mypy) get no derived format check."""
    ns = _ns(merge_sentinels=["pytest -q", "mypy src"])
    cmd = ns["_merge_guard_command"]("owner/repo", 3)
    assert "ruff format --check" not in cmd


def test_merge_guard_empty_sentinels_safe() -> None:
    """No sentinels => no ruff-format lines, guard still well-formed."""
    ns = _ns(merge_sentinels=[])
    cmd = ns["_merge_guard_command"]("owner/repo", 1)
    assert "ruff format --check" not in cmd
    assert "MERGE_GUARD_OK" in cmd


# ─── part 1: fix_ci runs ruff format ─────────────────────────────────────────


def test_script_fix_ci_instructs_ruff_format() -> None:
    """The production script's ``fix_ci`` dispatch tells the fix agent that the CI lint
    job runs ``ruff format --check`` and a format-only failure must be fixed with
    ``ruff format`` — so the CI-fix loop can self-resolve a format-only failure instead
    of falsely parking at the verify gate."""
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    assert '"task": "fix_ci"' in src
    # The dispatch carries explicit ruff-format guidance (via the named hint constant).
    assert "ruff format" in src
    # And the hint is attached to the fix_ci CI-fix payload, not only review-fix.
    fix_ci_idx = src.index('"task": "fix_ci"')
    window = src[fix_ci_idx : fix_ci_idx + 400]
    assert "FIX_CI_LINT_HINT" in window


def test_fix_ci_lint_hint_constant_present() -> None:
    """The lint-fix hint is a named constant so it is one source of truth and survives
    the reregister round-trip's header extraction unchanged."""
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    assert "FIX_CI_LINT_HINT" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
