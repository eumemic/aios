#!/usr/bin/env bash
# Boot the aios FastAPI app in-process and dump its OpenAPI spec to
# openapi.json at repo root. Used as the source-of-truth input for any
# future codegen (typed Python SDK, MCP server, etc.) and as the
# committed contract whose drift CI guards via
# tests/unit/test_openapi_snapshot.py.
#
# No live database or worker is needed — app.openapi() only walks route
# metadata. The placeholder env vars below satisfy aios.config import-time
# validation; nothing actually connects.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

AIOS_API_KEY=x \
AIOS_VAULT_KEY=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA= \
AIOS_DB_URL=postgresql://x@localhost/x \
uv run python -c "
import json
from aios.api.app import create_app
print(json.dumps(create_app().openapi(), indent=2))
" > openapi.json

echo "wrote $(wc -c < openapi.json | awk '{print $1}') bytes to $(pwd)/openapi.json"
