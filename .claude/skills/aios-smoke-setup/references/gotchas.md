# Smoke-setup gotchas

The eight failure modes that cost ~5–15 min each during the openclaw-parity PR smoke session.  Symptom → root cause → one-liner fix.

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 1 | `worker.duplicate_instance_refused` | A sibling worktree's `aios worker` holds the supervisor advisory lock on the shared `aios` Postgres database. | Use a dedicated DB per worktree (`AIOS_DB_URL=…/aios_smoke_<branch>`).  Don't kill the sibling's worker — it might be a managed monitor session another Claude Code instance is driving. |
| 2 | `[Errno 48] address already in use` on `:8090` | Sibling worktree's api on the default port. | Pick a different port (`AIOS_API_PORT=8091`).  Probe with `lsof -iTCP:<port> -sTCP:LISTEN` first. |
| 3 | `procrastinate.exceptions.ConnectorException` after a successful alembic migration | `alembic upgrade head` creates aios tables but **not** the procrastinate schema. | Use `uv run aios migrate` instead — it does both. CLAUDE.md is misleading on this point. |
| 4 | `telegram.error.Conflict: terminated by other getUpdates request` in worker.log | Bot token is being polled by an unknown remote (fly.io, render, coworker's box, stale local process you can't grep for). | Pre-flight with `curl …/getUpdates?timeout=2` before starting the worker.  If `error_code: 409`, the user must rotate the token (Telegram invalidates the old one) or stop the rogue poller. |
| 5 | `RuntimeError: connector 'signal' listed in connectors_enabled but no aios.connectors entry point` | Source `.env` has `AIOS_CONNECTORS_ENABLED=signal,telegram` but this worktree's venv only installed telegram. | Narrow `AIOS_CONNECTORS_ENABLED` to the connectors actually present (`uv pip list \| grep aios-`). |
| 6 | Bot replies never arrive; session stuck in `status=idle stop_reason={"type":"requires_action"}` after first `telegram_send` | `AIOS_DEFAULT_MCP_PERMISSION_POLICY` defaults to `always_ask`; every MCP tool call parks for human confirmation. | Add `AIOS_DEFAULT_MCP_PERMISSION_POLICY=always_allow` to `.env` for unattended smoke runs.  In production, leave it default and configure per-toolset on the agent. |
| 7 | `connections attach` returns `connector 'X' is None; snapshot unavailable` even though `aios connectors list` shows status=running | Pre-existing bug in `_assert_account_in_snapshot` reading the wrong envelope key (singular `connector` instead of plural `connectors`). | Fixed in #242.  If you hit this on a branch off pre-`bff6660` master, cherry-pick the fix or use `--phase post-attach` to skip the validation. |
| 8 | `aios sessions create` rejects with "either provide --agent and --environment-id" | Subcommand needs **both** flags; `--agent` alone isn't enough. | `aios sessions create --agent <agent_id> --environment-id <env_id>`.  See also: `envs` (plural) not `environments`; `--session-id` not `--session` on connections attach.  See `references/missing-affordances.md` for the wishlist of CLI naming consistency. |

## Fresh-DB & headless bring-up (added after the #685 verify session)

These bite a **cold start** — a brand-new per-branch DB and/or a
no-connector runtime for `/verify`-ing a harness change.  `setup.sh` now
handles all five, but they're documented here because a manual bring-up
(or a future refactor of the script) will re-hit them.

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 9 | `aios migrate` dies with `pydantic … AIOS_WORKSPACE_ROOT must be an absolute path; got 'workspaces'` | Source `.env` ships `AIOS_WORKSPACE_ROOT=./workspaces` (relative).  Pydantic rejects it regardless of CWD; api and worker would also resolve a relative path against diverging CWDs. | `setup.sh` `write_env` now overrides it to `$WORKTREE/workspaces` (absolute).  Manual: set an absolute path. |
| 10 | Every CLI/API call returns `401 invalid api key` on a fresh DB, even with the `.env` `AIOS_API_KEY` | Auth is **DB-backed** (`account_keys` table via `lookup_account_by_key_hash`); the `.env` `AIOS_API_KEY` is *not* what the server validates.  A fresh DB has zero keys. | `setup.sh` now runs `bootstrap_root` after migrate and writes the minted plaintext key back into `.env`.  Manual: `uv run python -c "...bootstrap_root(pool, display_name='x')..."` then use the returned key.  (The HTTP `/v1/accounts/bootstrap` route also works but is gated on `AIOS_BOOTSTRAP_TOKEN`.) |
| 11 | Fixtures land on the wrong runtime / `connection refused` from the CLI during `build_fixtures` | `write_env` overrode `AIOS_API_PORT` but not `AIOS_URL`; the CLI targets `AIOS_URL`, so it hit whatever the source `.env` pointed at (often `:8090` — a sibling worktree's runtime). | `setup.sh` now also pins `AIOS_URL=http://127.0.0.1:$PORT`.  Manual: export `AIOS_URL` to match the chosen port. |
| 12 | `docker exec aios-pg …: No such container` | `ensure_db` hardcoded `aios-pg`; compose now names it `aios-postgres-1`. | `setup.sh` `pg_container()` auto-detects the container publishing `:5433`, falling back to `aios-postgres-1`. |
| 13 | api/worker spawned by `setup.sh` (or by `nohup … &` in a Bash tool call) die the moment the call returns — runtime gone before you can drive it | When **Claude** runs a command via the Bash tool, the harness reaps the call's process group on completion; `nohup` only ignores SIGHUP, not the SIGTERM/SIGKILL that reaping sends.  (A human running `setup.sh` in a real terminal is fine — children reparent to init.) | For Claude-driven sessions: `setup.sh --phase prep`, then start **api and worker as two separate `run_in_background` Bash calls**, then `setup.sh --phase fixtures`.  See SKILL.md "Headless bring-up when Claude is driving". |

## Less-frequent pitfalls

- **Pre-commit hook gates on the whole working tree**, not just staged files.  Splitting two logical changes into separate commits requires `git stash` of the file you don't want gated.
- **Tool confirmation manual unblock**: if you find yourself stuck with `requires_action` and need to release a single tool call without restarting, POST to `/v1/sessions/<id>/tool-confirmations` with `{"tool_call_id": "...", "result": "allow"}` (note: `result`, not `decision`; both `allow` and `deny` accepted; `deny_message` optional).
- **`-f json` is a global flag**, must come before the subcommand chain: `aios -f json sessions get <id>` ✓, `aios sessions -f json get <id>` ✗.
- **`message.video_note` vs `message.video`**: regular videos (paperclip → photo/video, file like `IMG_3802.MOV`) hit the `video` slot.  Round circular video notes (long-press the mic icon to switch to camera mode on mobile) hit `video_note` and only ship from mobile clients.
