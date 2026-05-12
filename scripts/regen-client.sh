#!/usr/bin/env bash
# Regenerate the typed Python SDK at packages/aios-sdk/aios_sdk/_generated/
# from the canonical OpenAPI spec at openapi.json.
#
# Usage:
#   ./scripts/regen-client.sh
#
# Run this after any change to FastAPI route signatures, response_models,
# operation_ids, or response shapes. Commit both openapi.json and the
# regenerated _generated/ together.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

bash scripts/regen-openapi.sh

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# --meta none drops the standalone-package wrapper (no pyproject/README);
# relative imports inside the generated code work no matter what name we
# mount it under.
uv run openapi-python-client generate \
  --path openapi.json \
  --output-path "$WORK" \
  --meta none \
  --overwrite

OUT="packages/aios-sdk/aios_sdk/_generated"
mkdir -p "$OUT"
rsync -a --delete --exclude=".ruff_cache" "$WORK/" "$OUT/"

py_count=$(find "$OUT" -name '*.py' | wc -l | awk '{print $1}')
echo "regenerated $OUT/ ($py_count py files)"
