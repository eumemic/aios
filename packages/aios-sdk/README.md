# aios-sdk

Typed Python SDK for the aios management plane.

The bulk of this package is auto-generated from the committed OpenAPI
spec at `openapi.json` (repo root) via `scripts/regen-client.sh`. The
generated client lives at `aios_sdk._generated`; this package's
top-level re-exports the curated public surface plus hand-written
helpers (SSE streaming, env-resolved client factory).

## Install

Releases are attached to [GitHub Releases](https://github.com/eumemic/aios/releases?q=aios-sdk)
as built wheels. There is no PyPI package; pin to a tagged version:

```bash
uv add "https://github.com/eumemic/aios/releases/download/aios-sdk-v0.1.0/aios_sdk-0.1.0-py3-none-any.whl"
```

The git-tag form is available as a source-pin fallback:

```bash
uv add "aios-sdk @ git+https://github.com/eumemic/aios.git@aios-sdk-v0.1.0#subdirectory=packages/aios-sdk"
```

The SDK encodes whatever `openapi.json` said when its tag was cut, so the
deployed aios runtime must be at least as recent as the SDK version on the
caller side. In practice: tag SDK releases from commits that have already
been promoted to `:stable` (see [Releasing](#releasing) below).

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

## Releasing

1. After the aios runtime has been promoted to `:stable` via
   `promote-api.yml` / `promote-worker.yml`, pick a new SDK version in
   `packages/aios-sdk/pyproject.toml` (`project.version`). 0.x semver:
   breaking changes are allowed at minor bumps.
2. Add an entry to `packages/aios-sdk/CHANGELOG.md` under
   `## X.Y.Z — YYYY-MM-DD`.
3. Open a PR with both changes; merge to master.
4. Tag the merge commit on master as `aios-sdk-vX.Y.Z` and push the tag:
   ```bash
   git tag aios-sdk-vX.Y.Z
   git push origin aios-sdk-vX.Y.Z
   ```
5. The release workflow (`.github/workflows/release-aios-sdk.yml`)
   builds the wheel + sdist and attaches them to a GitHub Release of
   the same name.

The version in `pyproject.toml` must match the tag suffix; the workflow
asserts this and fails the release on mismatch.
