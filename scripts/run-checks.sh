#!/bin/bash
# Central check runner for aios
#
# Usage:
#   scripts/run-checks.sh                  # run all checks
#   scripts/run-checks.sh --skip tests     # skip pytest
#   scripts/run-checks.sh --skip mypy      # skip mypy
#   scripts/run-checks.sh --fail-on-autofix # fail if ruff auto-formats (for pre-commit)
#   scripts/run-checks.sh --fail-fast      # stop at first failure
#
# Checks: ruff, mypy, tests (unit only — e2e needs Docker)

set -uo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

# ── Parse flags ────────────────────────────────────────────────────────────────

SKIP=""
FAIL_ON_AUTOFIX=0
FAIL_FAST=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip)       SKIP="$2"; shift 2 ;;
        --fail-on-autofix) FAIL_ON_AUTOFIX=1; shift ;;
        --fail-fast)  FAIL_FAST=1; shift ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

should_skip() { [[ "$SKIP" == *"$1"* ]]; }

OVERALL=0
fail() { OVERALL=1; [[ $FAIL_FAST -eq 1 ]] && exit 1; }

# ── Ruff ───────────────────────────────────────────────────────────────────────

if ! should_skip ruff; then
    echo "── ruff ──"

    if [[ $FAIL_ON_AUTOFIX -eq 1 ]]; then
        # Check + format, then detect if anything changed
        uv run ruff check --fix src tests 2>&1
        uv run ruff format src tests 2>&1
        if [[ -n "$(git diff --name-only)" ]]; then
            echo "ruff auto-fixed files — please re-stage:" >&2
            git diff --name-only >&2
            fail
        fi
    else
        uv run ruff check src tests || fail
        uv run ruff format --check src tests || fail
    fi
fi

# ── Mypy ───────────────────────────────────────────────────────────────────────

if ! should_skip mypy; then
    echo "── mypy ──"
    uv run mypy src || fail
fi

# ── Tests ──────────────────────────────────────────────────────────────────────

if ! should_skip tests; then
    echo "── pytest (unit) ──"
    uv run pytest tests/unit -q || fail
fi

echo ""
if [[ $OVERALL -eq 0 ]]; then
    echo "All checks passed."
else
    echo "Some checks failed." >&2
fi
exit $OVERALL
