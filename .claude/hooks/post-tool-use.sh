#!/bin/bash
# Claude PostToolUse hook — runs quality checks after Write/Edit/MultiEdit
#
# Input: JSON on stdin with tool information
# Output: Errors to stderr on failure, exit code 2 to block

set -uo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Extract file path from JSON
FILE_PATH=$(echo "$JSON" | jq -r '.tool_input.file_path // .tool_input.filePath // ""')

# Only process Python files
if [[ ! "$FILE_PATH" == *.py ]]; then
    exit 0
fi

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

# Create temp files for capturing output
RUFF_OUTPUT=$(mktemp)
MYPY_OUTPUT=$(mktemp)
PYTEST_OUTPUT=$(mktemp)
trap "rm -f $RUFF_OUTPUT $MYPY_OUTPUT $PYTEST_OUTPUT" EXIT

RUFF_FAILED=0
MYPY_FAILED=0
PYTEST_FAILED=0

# Ruff: check the specific file (fast)
if ! uv run ruff check "$FILE_PATH" --output-format concise > "$RUFF_OUTPUT" 2>&1; then
    RUFF_FAILED=1
fi

# Mypy: check src/ for full type context
if ! uv run mypy src --no-error-summary > "$MYPY_OUTPUT" 2>&1; then
    MYPY_FAILED=1
fi

# Pytest: re-run last-failed tests only (instant when nothing is broken).
# Exit code 5 = "no tests collected" which is expected when nothing has failed.
uv run pytest --lf --lfnf=none -x -q --tb=short --no-header > "$PYTEST_OUTPUT" 2>&1
PYTEST_RC=$?
if [ $PYTEST_RC -ne 0 ] && [ $PYTEST_RC -ne 5 ]; then
    PYTEST_FAILED=1
fi

if [ $RUFF_FAILED -ne 0 ] || [ $MYPY_FAILED -ne 0 ] || [ $PYTEST_FAILED -ne 0 ]; then
    [ $RUFF_FAILED -ne 0 ] && [ -s "$RUFF_OUTPUT" ] && cat "$RUFF_OUTPUT" >&2
    [ $MYPY_FAILED -ne 0 ] && [ -s "$MYPY_OUTPUT" ] && cat "$MYPY_OUTPUT" >&2
    [ $PYTEST_FAILED -ne 0 ] && [ -s "$PYTEST_OUTPUT" ] && cat "$PYTEST_OUTPUT" >&2
    exit 2
fi

exit 0
