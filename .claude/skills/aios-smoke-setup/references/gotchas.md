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

## Less-frequent pitfalls

- **Pre-commit hook gates on the whole working tree**, not just staged files.  Splitting two logical changes into separate commits requires `git stash` of the file you don't want gated.
- **Tool confirmation manual unblock**: if you find yourself stuck with `requires_action` and need to release a single tool call without restarting, POST to `/v1/sessions/<id>/tool-confirmations` with `{"tool_call_id": "...", "result": "allow"}` (note: `result`, not `decision`; both `allow` and `deny` accepted; `deny_message` optional).
- **`-f json` is a global flag**, must come before the subcommand chain: `aios -f json sessions get <id>` ✓, `aios sessions -f json get <id>` ✗.
- **`message.video_note` vs `message.video`**: regular videos (paperclip → photo/video, file like `IMG_3802.MOV`) hit the `video` slot.  Round circular video notes (long-press the mic icon to switch to camera mode on mobile) hit `video_note` and only ship from mobile clients.
