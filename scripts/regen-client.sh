#!/usr/bin/env bash
# Regenerate the typed Python SDK at src/aios/sdk/_generated/ from the
# canonical OpenAPI spec at openapi.json.
#
# Usage:
#   ./scripts/regen-client.sh
#
# Run this after any change to FastAPI route signatures, response_models,
# operation_ids, or response shapes. Commit both openapi.json and the
# regenerated _generated/ together.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# 1. Refresh the spec snapshot (the SDK is downstream of openapi.json).
bash scripts/regen-openapi.sh

# 2. Generate into a tmp dir, then sync into src/aios/sdk/_generated/.
#    --meta none drops the standalone-package wrapper (no pyproject/README);
#    relative imports inside the generated code work no matter what name
#    we mount it under.
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

uv run openapi-python-client generate \
  --path openapi.json \
  --output-path "$WORK" \
  --meta none \
  --overwrite

mkdir -p src/aios/sdk/_generated
rsync -a --delete --exclude=".ruff_cache" "$WORK/" src/aios/sdk/_generated/

py_count=$(find src/aios/sdk/_generated -name '*.py' | wc -l | awk '{print $1}')
echo "regenerated src/aios/sdk/_generated/ ($py_count py files)"
