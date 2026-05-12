# aios-sdk

Typed Python SDK for the aios management plane.

The bulk of this package is auto-generated from the committed OpenAPI
spec at `openapi.json` (repo root) via `scripts/regen-client.sh`. The
generated client lives at `aios_sdk._generated`; this package's
top-level re-exports the curated public surface plus hand-written
helpers (SSE streaming, env-resolved client factory).

## Quick start

```python
from aios_sdk import Client, client_from_env, stream_session
from aios_sdk._generated.api.agents import list_agents

client = client_from_env()
response = list_agents.sync_detailed(client=client)

with stream_session(client, session_id) as events:
    for msg in events:
        print(msg.event, msg.data)
```

## Drift guarding

Two snapshots guard the spec → client pipeline:

- `tests/unit/test_openapi_snapshot.py` asserts the committed
  `openapi.json` matches what `aios.api.create_app().openapi()`
  produces. If a route changes, re-run `scripts/regen-openapi.sh`.
- `tests/unit/test_client_snapshot.py` asserts the committed
  `_generated/` tree matches what `openapi-python-client` produces
  from `openapi.json`. If the spec changes, re-run
  `scripts/regen-client.sh`.
