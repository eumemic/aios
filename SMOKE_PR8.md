# PR 8 capstone smoke — findings

Live smoke of the `refactor/328-pr8-cleanup` branch plus the `refactor/328-fixup-smoke` follow-up branch.  Telegram + Signal exercised against real bot accounts, with the smoke session bound to two connectors at once (telegram + signal).  Two-bot QA group brought online to verify bidirectional capability rendering against an independent peer bot.

## Environment

- **First (botched) run**: api/worker against the shared local dev DB at `localhost:5433/aios` — sourced from `/Users/tom/code/aios/.env` directly.  Persisted real connection records (EumemicBot 8622121495, Ultron signal 909) and produced ~380 events of live Telegram traffic in an existing prod-shadow session before being torn down.  Cleanup: archived the stray smoke connection in the local DB, archived Tom's binding modifications, revoked smoke runtime tokens, kept the model-switch + session events as-is.
- **Second (correct) run**: per-worktree dev instance via `aios dev bootstrap`, separate DB (`aios_dev_worktree_effervescent_tickling_cook_9myls3_995bf272`) on the same postgres host, fresh agent + session + connections.  Initial agent was Haiku 4.5; switched to Sonnet 4.6 for the bidirectional matrix run (Sonnet produces cleaner one-shot tool calls and less monologue drift).
- Connectors run locally, not in Docker.  `signal-cli` daemon spawned from `~/.local/share/signal-cli` (shared with other local instances by design).

## Capability matrix — bidirectional

Verified end-to-end with an independent peer bot ("Jarvis") on a 3-party Signal group ("QA"):

| Capability | Send (bot → human/peer) | Receive (human/peer → bot) |
|---|---|---|
| Text message | ✅ | ✅ |
| Emoji reaction | ✅ | ✅ |
| Quote-reply / threading | ✅ | ✅ |
| @mention | ✅ (pings recipient) | ⚠ delivered, but mention metadata flattened to `@<name>` text (see #5) |
| Attachment | ❌ blocked upstream (#7: ARM64 sandbox) | ✅ metadata + path delivered; ❌ content access (sandbox) |
| Voice note (audio) | ❌ blocked upstream (sandbox) | ✅ metadata + path delivered; ❌ playback / transcription (sandbox) |
| Sound effect (audio) | ❌ blocked upstream (sandbox) | ✅ metadata + path delivered; ❌ playback (sandbox) |
| Audio transcription | n/a — not in connector tool spec | n/a |
| Read receipts (sender side) | ✅ promptly via #2 fix | n/a (we don't act on signal-cli receipts) |

Send-side ❌s and the receive-side "❌ content access" entries all trace to **#7** (the ARM64 sandbox image gap), not connector code.

Telegram-side: text inbound + text outbound on `@EumemicBot`, plus switch_channel between signal-DM + signal-group + telegram-DM focal channels in a single session.  Three bindings on one session record (signal connection × 1 + telegram connection × 1), all routing to the same model context.

## Bugs found

### Fixed in `refactor/328-fixup-smoke`

**1. `connection_id` exposed as a required arg in model-facing tool schemas** — `cc92e35`

`packages/aios-connector-http/aios_connector_http/schema.py` stripped `account` and `chat_id` from the JSON schema but **not** `connection_id`.  Model sees `connection_id` as a required keyword-only param it must supply; runner injects it correctly from the SSE payload but only when not already passed, so the model's guess overrides reality and the per-conn-state lookup `KeyError`s with the bad value.  Symptom on smoke: `telegram_send({"connection_id": "", "text": "..."})` and `telegram_send({"connection_id": "8622121495/1595907265", "text": "..."})` both returned `{"error": "''"}` / `{"error": "'8622121495/1595907265'"}` until the model gave up and tried without `connection_id`.

Fix: add `connection_id` to the strip set alongside `account` + `chat_id`, rename `_FOCAL_INJECTABLE` → `_INJECTED_PARAMS` to reflect that `connection_id` comes from the SSE payload (not the focal channel), refresh module docstring, lock with a `test_connection_id_is_excluded_from_schema` test.

**2. Signal delivery receipts batch unpredictably; sender sees inconsistent checkmarks** — `6dfc978`

signal-cli's daemon-mode automatic delivery receipts (the 2nd grey checkmark) batch on internal flush triggers we don't control (new inbound traffic, reconnects, timers).  Some sender-side messages got 2 checks immediately; some got 1 then promoted to 2 minutes later; some sat at 1 for the duration of the smoke.  Not a delivery failure (every message reached the bot) — purely sender UX.

Fix: after `emit_inbound` returns 201 ("the agent has seen it" in the session log), fire an explicit `sendReceipt --type read` via the daemon RPC targeting the original sender.  Best-effort: a receipt failure is logged + swallowed since the inbound is already persisted.  Semantics: ✅ on the sender side now means "the agent's session log has the message", not "signal-cli's flush scheduler got around to it".  Locked with an e2e assertion that `sendReceipt` fires per inbound with the right account/recipient/targetTimestamp.

**3. Empty-content inbound 422s and crashes the connector container** — `326b612`

Reaction-only, attachment-only, and group-update envelopes have no text body.  `build_content_text(msg)` correctly produces `""`; the connector POSTs `content=""` as a multipart form part; **the api 422s it as a missing required field** because FastAPI's multipart `Form` parser treats empty values as `input=null` (missing).  The 422 propagates through `serve_connection` → TaskGroup, taking down the entire container along with every other connection it serves.  Both crashes in this smoke had `content_len=0` as the last `connector.inbound` log line before exit.

Root cause analysis was painful for a second reason: `emit_inbound` called `response.raise_for_status()` and let the `HTTPStatusError` propagate verbatim, discarding the response body — which is exactly where the FastAPI validation diagnostic lives.  Operator saw `HTTPStatusError: 422 ...` with no hint at the offending field.

Two-part fix:

- **api**: `src/aios/api/routers/connectors.py` — `content: Annotated[str, Form()] = ""` (default empty so attachment-only / reaction-passthrough envelopes are a first-class shape).  Locked with `test_empty_content_inbound_accepted`.
- **connector**: `packages/aios-connector-http/aios_connector_http/runner.py:emit_inbound` — log `connector.inbound.failed` with `status_code` + `body` (capped at 2000 chars) before raising.  Locked with `test_emit_inbound_logs_response_body_on_error`.

Verified by replaying the previously-crashing payload via curl (now returns 201) AND by Jarvis exercising the full bidirectional matrix above — reaction inbounds, attachment inbounds, voice notes, sound effects all flow through cleanly.  Container stayed up across the entire smoke after the fix.

### Open — observed but not yet fixed

**4. `emit_inbound` propagating raise still tears down the container on non-2xx**

Even with #3, the connector still treats any `emit_inbound` failure as fatal: 4xx propagates → `serve_connection` raises → TaskGroup tears down → process exit.  That's correct for genuine server outages and for unrecoverable contract violations (5xx, 401/403), but for routine 4xx ("the api rejected one specific payload"), drop-and-continue makes more sense.  One bad envelope shouldn't kill every other connection the container is serving.

Recommended: in `_handle_envelope`, catch `httpx.HTTPStatusError` for 4xx specifically, log + `return`; let 5xx and connection errors propagate.

**5. Inbound `@mention` metadata is collapsed into plain text**

`connectors/signal/src/aios_signal/parse.py:_substitute_mentions` replaces the Unicode placeholder (`￼`) with `@<display_name>` inline in the message body.  The model sees a string indistinguishable from "the sender typed my name as text" — there's no signal that the sender's client encoded a structured mention targeting the bot's own UUID.

Recommended: extend `build_metadata` to carry a `mentions: list[{"uuid", "name"}]` entry on the inbound event, plus a derived `self_mentioned: bool` when one of the entries matches the bot's UUID.  The model can then choose to react/respond differently when actually pinged vs. just named.

**6. Group `signal_send` returns `{"status": "ok"}`; DM returns `{"sent_at_ms": ...}`**

Inconsistent result shape between group and DM outbound.  `_extract_timestamp(result)` in `connectors/signal/src/aios_signal/connector.py` returns `None` for group sends, so the tool result falls back to `{"status": "ok"}`.  Minor — the model can edit/react/delete a previous message by `target_timestamp_ms`, but only knows the timestamp for DM sends.  Group outbound editing/reacting/deleting from the bot's own messages is therefore harder.

Recommended: check what signal-cli returns for group `send` calls and surface that timestamp (signal-cli's `sendMessage` should return a per-recipient envelope with a timestamp even for group sends).

### Pre-existing, not stack-related

**7. ARM64 sandbox image missing** (`ghcr.io/eumemic/aios-sandbox:latest` has no `linux/arm64/v8` manifest)

Blocks every sandbox-dependent capability on Apple Silicon developer machines.  `bash`, `read`, `write`, `edit`, `grep`, `glob` tools fail with `SandboxBackendError`.  Cascades through the smoke: SmokeBot couldn't create attachment files, generate audio, read inbound attachment contents, transcribe voice notes, or play back sound effects.  Every ❌ in the matrix above traces back here.

Fix scope: rebuild `docker/Dockerfile.sandbox` as multi-arch via `docker buildx build --platform linux/amd64,linux/arm64 --push`.

**8. Stale `mcp_toolset` config on existing pre-PR-5 agents**

Agents like `eumemic-bot` still carry an `mcp_toolset` tools-list entry with `mcp_server_name='telegram'`.  Inert at runtime (skipped by `to_openai_tools`, not in `agent.mcp_servers`), but presents as "the agent has 7 tools" in operator-facing reads.  Not a bug — orphaned data from before PR 5's MCP-out, no data migration was written to scrub.

**9. Polluted session history → tool-name hallucination**

The pre-PR-5 `eumemic-bot` session at `sess_01KQXTMDFR8NPCE24ZBEQE9W81` had ~1200 events of `mcp__telegram__telegram_send(...)` calls in its history.  Post-PR-5 the actual exposed tool is plain `telegram_send`, but the model pattern-matched off history and called `mcp__telegram__telegram_send(...)`, which routed to the MCP dispatcher and errored `"MCP server 'telegram' not found"`.  Real architectural cost of the monotonic-context invariant.

**10. `signal-cli` daemon child not in connector's process group**

Encountered mid-smoke: `pkill -f aios_signal` killed the Python connector but left the spawned `signal-cli daemon` JVM running with the TCP port + SQLite lock on `~/.local/share/signal-cli/data` still held.  Multiple restarts accumulated three orphaned daemons fighting for the lock.  Symptom: new connector's daemon couldn't fully claim the receive websocket — the oldest live daemon kept consuming inbounds.

Recommended: connector should put `signal-cli daemon` in its own process group (start_new_session=True on subprocess) so a single SIGTERM to the connector cleans up the daemon tree, OR explicitly terminate the subprocess on `SignalConnector.teardown()` and confirm via wait_for / timeout.

### Verification

After the three fixes in `refactor/328-fixup-smoke`, the previously-crashing scenarios were re-exercised end-to-end:

- Reaction-only inbound (empty content + reaction metadata) — ✅ no crash, model sees the reaction context
- Attachment-only inbound with text caption present — ✅ delivered, metadata + path visible to model
- Voice note inbound (audio attachment, empty caption) — ✅ delivered
- Sound effect inbound (small audio attachment) — ✅ delivered
- All four scenarios run in sequence against the same connector container without restart — container stays alive

## Architecture observations / footguns

### Smoke isolation gaps

- **Same-host shared DB by default**: sourcing `/Users/tom/code/aios/.env` from a worktree picks up the shared dev DB.  Easy to pollute with smoke traffic.  `aios dev bootstrap` provisions a per-worktree DB on the same postgres host + writes a worktree-local `.env` — needs to be run *before* the first source of any `.env`.
- **`~/.local/share/signal-cli` is shared globally**: only one process at a time can hold a registered phone's device key.  Worktree-DB isolation doesn't isolate the signal-cli device.  Smoke against signal requires either a number with no other live aios instance claiming it, OR a separate config dir.
- **Telegram bot tokens in shared `.env`**: `getUpdates` is exclusive per token.  Any worktree's telegram connector polling the same token competes with prod / other worktrees.

### Runtime container can't scope discovery

`aios_signal` (and `aios_telegram`) discover connections via `/v1/connectors/connections` filtered only by connector *type*.  No way to subset by label, tag, or explicit allowlist.  Means: smoking ONE signal connection requires either (a) archiving every other active signal connection in the DB, or (b) running in an isolated DB.

### Verbose internal monologue from smaller models

Both qwen3.6-flash and Haiku 4.5 frequently emitted `INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER` content instead of either calling a connector send tool or returning silently (the latter being the "right" choice per the focal paradigm).  Sonnet 4.6 was noticeably cleaner — fewer wakes ended in monologue-only output, more wakes responded directly with `signal_send` when a reply was actually warranted.  Not a bug per se — bare assistant text is correctly never delivered — but a measurable model-capability gradient.

## Recommendations / follow-ups

| # | Action | Scope | Where |
|---|---|---|---|
| 4 | Container resilience: drop-and-continue on 4xx in `_handle_envelope` | one commit | runner.py |
| 5 | Expose mention metadata + `self_mentioned: bool` to the model | small feature | signal connector `build_metadata` + parse |
| 6 | Surface group-send timestamp | small fix | signal `signal_send` result shape |
| 7 | Build multi-arch sandbox image (linux/amd64 + linux/arm64) | infra | `docker/Dockerfile.sandbox` + buildx |
| 8 | Scrub orphaned `mcp_toolset` entries on agents missing matching `mcp_servers` | data migration or CLI | one-off |
| 10 | Process-group ownership of `signal-cli daemon` | small fix | signal `daemon.py:SignalDaemon.__aenter__` |
| (env) | `aios dev bootstrap` discoverability / harder-to-source-wrong-.env | DX | dev CLI doc |
| (env) | Runtime token connection-subset scope | feature | runtime tokens + discovery filter |

Items 4, 5, 6, 7, 10 are stack-level and could fold into the fixup PR.  Item 7 (sandbox image) is highest priority given how many ❌s in the matrix collapse to it.
