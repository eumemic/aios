# Workspace-Root Validation — Rollout Guide

> **This PR is detection and guarding only.** It does not correct the
> production API/worker `AIOS_WORKSPACE_ROOT` / volume-mount mismatch.
> Rollout must first align the effective workspace root and volume mount
> across both processes.

## What this PR does

1. **Startup validation** (`workspace_root_startup.py`): On boot, each
   process (API and worker) scans every live session row and verifies its
   `workspace_volume_path` resolves within the process's configured
   `AIOS_WORKSPACE_ROOT`. A mismatch fails startup with full diagnostics.

2. **Filesystem probe** (`production_watchdogs.py`): An optional periodic
   watchdog provisions a configured standing session's real sandbox and
   verifies workspace write/read capability, plus optional configured
   repository and memory sentinels.

3. **Resource safety**: The API lifespan closes the connection pool on any
   pre-yield startup exception (validation failure, crypto init failure, etc.).

## Pre-rollout checklist

- [ ] **Align `AIOS_WORKSPACE_ROOT`**: Confirm that both the API and worker
      processes use the same absolute path (e.g., `/srv/aios/workspaces`).
      The validation intentionally fails startup if any live session row
      is outside the configured root.

- [ ] **Align volume mounts**: The container/host volume mount backing
      `AIOS_WORKSPACE_ROOT` must be identical for both processes. A
      mismatch (e.g., API sees `/srv/aios/workspaces` but mounts a
      different host directory than the worker) defeats the validation
      even though both processes pass it.

- [ ] **Review scan budget**: The startup scan has a configurable deadline
      (`AIOS_WORKSPACE_SCAN_TIMEOUT_SECONDS`, default 30s) and per-page
      query timeout (`AIOS_WORKSPACE_SCAN_QUERY_TIMEOUT_SECONDS`, default
      10s). High-cardinality deployments (>30k live sessions) may need
      a higher scan budget.

## Optional filesystem probe

Disabled by default. To enable:

```bash
# On the worker only:
AIOS_STANDING_SESSION_FILESYSTEM_PROBE_SESSION_ID=sess_01KVBPGT6VNJFE3JXVMJD1BKNJ

# Optional tuning:
AIOS_STANDING_SESSION_FILESYSTEM_PROBE_INTERVAL_SECONDS=300   # default
AIOS_STANDING_SESSION_FILESYSTEM_PROBE_TIMEOUT_SECONDS=120    # default

# Optional capability sentinels (skip if not configured):
AIOS_STANDING_SESSION_FILESYSTEM_PROBE_REPO_SENTINEL=.git/HEAD
AIOS_STANDING_SESSION_FILESYSTEM_PROBE_MEMORY_SENTINEL=/mnt/memory/my-store/MEMORY.md
```

The probe exercises:
- **Core** (always): workspace scratch write, readback verification, cleanup
- **Repository** (only when `REPO_SENTINEL` is set): reads the configured
  file path (handles both normal `.git/HEAD` and git-worktree `.git` files)
- **Memory** (only when `MEMORY_SENTINEL` is set): reads the configured
  memory mount file

## Rollback

Revert commits on this branch. No database migration or persistent-volume
change is involved.

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `AIOS_WORKSPACE_SCAN_TIMEOUT_SECONDS` | 30 | Overall wall-clock budget for the startup scan |
| `AIOS_WORKSPACE_SCAN_QUERY_TIMEOUT_SECONDS` | 10 | Per-page DB query timeout |
| `AIOS_STANDING_SESSION_FILESYSTEM_PROBE_SESSION_ID` | *(unset)* | Session ID to probe; disabled when unset |
| `AIOS_STANDING_SESSION_FILESYSTEM_PROBE_INTERVAL_SECONDS` | 300 | Probe interval |
| `AIOS_STANDING_SESSION_FILESYSTEM_PROBE_TIMEOUT_SECONDS` | 120 | Overall probe deadline |
| `AIOS_STANDING_SESSION_FILESYSTEM_PROBE_REPO_SENTINEL` | *(unset)* | Sandbox-relative repo sentinel path |
| `AIOS_STANDING_SESSION_FILESYSTEM_PROBE_MEMORY_SENTINEL` | *(unset)* | Sandbox-absolute memory sentinel path |
