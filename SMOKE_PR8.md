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

**4. `emit_inbound` propagating raise tore down the container on non-2xx** — `315c969`

Even with #3, the connector previously treated any `emit_inbound` failure as fatal: 4xx propagated → `serve_connection` raised → TaskGroup tore down → process exit.  That's correct for genuine server outages and unrecoverable contract violations (5xx, 401/403), but for routine 4xx ("the api rejected one specific payload"), drop-and-continue is the right posture — one bad envelope shouldn't kill every other connection the container is serving.

Fix: in `_handle_envelope`, catch `httpx.HTTPStatusError` and classify by status.  4xx except 401/403 → log + drop the envelope, keep serving.  401/403 (token revoked / runtime misconfigured) and 5xx (server outage or bug) still propagate so the operator sees red on real problems.  Locked with five tests covering 400/422/401/503 + the happy path (read receipt fires on 2xx).

**5. Inbound `@mention` metadata was collapsed into plain text** — `67a14e0`

`connectors/signal/src/aios_signal/parse.py:_substitute_mentions` replaces the Unicode placeholder (`￼`) with `@<display_name>` inline in the message body.  The model saw a string indistinguishable from "the sender typed my name as text" — no signal that the sender's client encoded a structured mention targeting the bot's own UUID.  Group chats in particular use mentions to summon a response; that signal was getting lost in the substring.

Fix: structured `Mention(uuid, name)` dataclass on `InboundMessage`, populated from the raw envelope alongside the placeholder substitution.  `build_metadata` emits a `mentions: [{uuid, name}]` list and a derived `self_mentioned: bool` (True when one of the entries matches `bot_uuid`).  Agents that want different behavior for "tagged" vs "named-in-passing" inbounds read `metadata.self_mentioned` instead of grepping content.  Locked with four `build_metadata` tests + a parse test that asserts the raw envelope mention array survives intact through `parse_envelope`.

**Follow-on caught in live smoke** — `cfc464e`: extracting mention metadata to the inbound event was only half the job.  The harness's `_format_channel_header` rendered a whitelist of metadata fields into the visible `content` (channel, sender, timestamp, edited, reaction, reply_to, sticker_emoji) but **didn't include `mentions` or `self_mentioned`**.  So even with the connector fix the model never saw the structured signal — it only saw the placeholder-substituted `@<name>` text in the body, exactly the pre-fix shape.  Direct harness invocation against the persisted event confirmed: the metadata was there, the renderer was dropping it.  Fix: extend `_format_channel_header` to emit `self_mentioned=true` on the main header line and one `[mention: name=... · uuid=...]` line per entry so the model has both the boolean signal and the uuids for outbound mention encoding.  Locked with three new context tests.

**6. Group `signal_send` returned `{"status": "ok"}`; DM returned `{"sent_at_ms": ...}`** — `a5db125`

Investigation against the live daemon (signal-cli 0.14.2) revealed that `send` RPC for groups returns literal `null` regardless of params — no nested timestamp to fish out.  The smoke doc's original recommendation ("check what signal-cli returns for group send calls and surface that timestamp") rested on a premise that turned out to be false.

However, the timestamp *does* arrive on the receive stream as a self-echo envelope (`sourceUuid == bot_uuid` + `dataMessage.groupInfo` + `dataMessage.timestamp`).  Direct probe confirmed the shape; parse_envelope already drops self-messages to `None`, which is the existing hook we extend.

Fix: `signal_send` for groups pre-registers an `asyncio.Future` in a per-`(phone, chat_id)` FIFO before issuing the RPC, then waits up to 2s for the matching echo.  `_maybe_resolve_self_echo` runs in `_handle_envelope`'s `msg is None` branch, pops the head future, sets the timestamp.  On timeout we degrade to `{"status": "ok"}` so a slow network doesn't hang the tool call.  DMs untouched — signal-cli already returns the timestamp inline.  Locked with seven tests covering match / prune / non-bot-sender / DM-skip + signal_send end-to-end (echo → sent_at_ms, timeout → status:ok, DM skips registration entirely).

**Three follow-on bugs caught in live smoke after the first deploy:**

- **`9620a74`** — Self-echo for group sends actually lands on the **receiving peers' account streams**, not the bot's own account stream.  When signal-cli serves multiple registered phones (e.g. a co-located bot like Ultron in the QA group), the dataMessage echo for SmokeBot's send appears on Ultron's `+19092871349` stream — never on SmokeBot's `+16575274288` stream.  The original fix's `_handle_envelope` only watched the bot's own queue, so the echo never resolved, every group send hit the 2s timeout, and the result fell back to `{"status": "ok"}` 100% of the time.  Moved the resolver to `_inbound_dispatcher` and key match by `sourceUuid` against any known `bot_uuid` rather than per-account routing.  Live-verified: `signal_send` → `{"sent_at_ms": 1778711928198}` on the very next group send.
- **`c7d824c`** — Edit echoes use a DIFFERENT envelope shape: `envelope.editMessage.dataMessage` (nested one level deeper) with the new edit's timestamp at the envelope root.  The matcher only checked `envelope.dataMessage`; chained-edit timestamps (model edits its own message and wants the new timestamp for further edits) silently fell back to `{"status": "ok"}`.  Accept both shapes.
- **`4fe0ef5`** — When `daemon.rpc.call("send", ...)` raises (e.g. signal-cli's `InvalidSessionException` if a group member has no protocol session yet), the pre-registered echo future was orphaned at the head of the deque.  Drain-stale only pops `done()` futures; an orphan sits indefinitely and the NEXT successful send's echo resolves the orphan with the wrong message's timestamp.  Wrap in `try/finally` cancelling the future on the way out so drain-stale prunes it before the next send.

### Open — surfaced post-fixup

**12. Per-connection `serve_connection` failure crashes the entire connector container**

Adjacent to #4's failure-isolation theme but on the bring-up path rather than the inbound path.  When the discovery SSE emits `added` for a connection whose `serve_connection` raises — most commonly when signal-cli has no `accounts.json` entry for the phone, so `daemon.verify_phone` raises `BotAccountNotFoundError` — the exception propagates out of the spawned task into the runner's `TaskGroup`, which cancels every sibling worker (every healthy connection of this connector type) and exits the container.

Concretely: an operator types a typo in a new signal connection's `phone` secret, or attaches a connection for a phone they haven't yet `signal-cli register`-ed.  The discovery loop sees the new row, spawns `serve_connection(connection_id, secrets)`, the task raises before it ever drains its inbound queue, and the TaskGroup tears down the whole process.  Every other connection in that container loses its in-flight inbound queue.

This was theoretical during PR 8 smoke (we registered both phones before attaching) but surfaced in the post-fix architecture pass: it's the same "one bad envelope shouldn't kill every other connection" principle as #4, applied to connection lifecycle instead of inbound dispatch.

Recommended scope:

- Wrap each per-connection task body in a `try/except` that catches non-`CancelledError` exceptions, logs `signal.connection.bringup_failed` with the connection_id + error class + traceback, and ends the task cleanly.
- Optionally: POST a `connector_error` event back to aios so the api can mark the connection's binding as errored, surfacing the bad connection to operators via the existing connection-status reads instead of via container restart logs.
- Keep `CancelledError` propagating so `_on_connection_removed` continues to work (the runner cancels the task on `removed` events).

The same wrap applies to `telegram_connector.serve_connection` — a bad bot token currently has the same blast radius.

**13. SDK lifecycle hardening: duplication between signal + telegram that should live on `HttpConnector`**

Surfaced when reading the post-fixup connectors side-by-side.  Three of the smoke fixes (#4 4xx drop-and-continue, #10 SIGTERM trap, the recommended #12 per-connection failure isolation) landed in signal-specific code paths, which means **telegram inherits none of the hardening**.  Same footguns, same blast radius — `pkill -f aios_telegram` leaves the PTB updater's `getUpdates` long-poll holding the bot token against Telegram's API until the dead poll times out, and a 422 on any single inbound envelope still tears down the whole telegram container.

Per-connector duplication that should be hoisted to the SDK base:

| Item | Currently | Should be |
|---|---|---|
| SIGINT / SIGTERM trap + cancel-on-stop helper | `aios_signal/__main__.py` (telegram is unhardened) | `HttpConnector.run_until_stopped()` or a `runner.serve(connector)` wrapper; both `__main__.py` files collapse to one line |
| 4xx drop-and-continue on `emit_inbound` | `aios_signal/connector.py:_handle_envelope` (telegram is unhardened) | `HttpConnector.emit_inbound` directly, with a `raise_on_4xx=True` escape hatch for callers that genuinely want fatal-on-4xx |
| Per-connection `serve_connection` failure isolation (#12) | nowhere yet — both connectors crash the container on a bad bring-up | `HttpConnector._on_connection_added`'s task wrapper catches non-`CancelledError`, logs structured failure, optionally POSTs a `connector_error` back to aios |
| `_conn_state` dict + finally-pop boilerplate; focal channel `f"{connector}/{account}/{chat_id}"` string | reimplemented inline in every connector | generic state slot via `set_connection_state` / `get_connection_state`; `self.focal_channel(account, chat_id)` helper |

Correctly per-connector (do NOT hoist): `event_id` construction (platform-specific identity tuples), `build_metadata` content (signal has mentions / quote / reaction; telegram has message_id / old_emojis / new_emojis), inbound parsing, long-lived per-platform plumbing (signal-cli daemon vs PTB Application), tool method implementations.

After the refactor: telegram inherits #4 / #10 / #12 hardening for free; the next connector author gets the safety rails by default; each connector shrinks by ~30-50 LOC.

Note: changing `emit_inbound`'s default to swallow 4xx is a behavior change.  Either (a) signal removes its wrap when the SDK gets one, or (b) the SDK adds the wrap as opt-in (`raise_on_4xx=False` default, but with a strict mode for callers that depend on the raise).  Sign-off needed on the default before landing.

**14. Signal account registration via the aios API (close the SSH-required gap)**

Surfaced when reasoning about operator UX for adding new signal/telegram accounts.  Today's aios api is the bookkeeping side: `POST /v1/connections`, `PUT .../secrets`, `attach`, `configure-per-chat`, `bind-chat` — all dynamic, all no-restart for adding connections of an **already-registered** platform account.  The connector runtime's discovery SSE picks up new connection rows without a container bounce.

What's gated behind SSH today: signal-cli's `register` / `verify` / `submitRateLimitChallenge` / `updateProfile` JSON-RPC methods.  signal-cli exposes them on the running daemon, but the connector doesn't route them through; operators have to SSH into the host and shell into `signal-cli` directly to register a new phone.  Once registration completes, `accounts.json` updates and `verify_phone` re-reads it fresh per call — so the running daemon picks up the new account without restart.  The api gap is the missing piece, not the runtime.

Telegram has a different shape: bot creation via @BotFather is upstream-manual (Telegram doesn't expose programmatic bot creation), but token-in-hand → connection-attached is already a single `POST /v1/connections` away.  So telegram's part of this finding is "already done modulo the irreducible BotFather step".

Recommended scope for signal:

| Piece | Size |
|---|---|
| Three api routes (`/v1/connectors/signal/register`, `/verify`, optional `/profile`) | ~120 LOC + tests |
| Management-call SSE event kind (sibling of `tool_call` on the existing per-type call stream) | ~80 LOC + tests |
| SDK base: management-call dispatcher routing the new event to per-connector handlers | ~40 LOC + tests |
| Captcha-handoff response shape (api returns signalcaptchas.org URL; operator solves; reposts token) | ~30 LOC + tests |
| CLI convenience (`aios signal register +<phone>` / `aios signal verify <code>`) | ~60 LOC |

Architecture choice: reuse the existing tool-call SSE pattern (api emits `management_call` events on the connector-type stream; connector handles via a sibling of `_tool_loop`; result POSTs back through the existing tool-result route, keyed by `call_id` and unblocked via the same one-shot future pattern requires-action custom tools already use).  Fits the current architecture without adding a new connector-side HTTP listener.

End-to-end signal flow then becomes:

```
POST /v1/connectors/signal/register   { phone, captcha_token? }
→ if captcha needed: 200 { captcha_url }; operator solves + reposts with token
→ 200 { verification_required: true }

POST /v1/connectors/signal/verify     { phone, code }
→ 200 { account: { phone, uuid } }    // accounts.json now has the entry

POST /v1/connections                  { connector: "signal", account: <phone>, secrets: { phone } }
→ 201                                   // discovery SSE fires "added"

POST /v1/connections/{id}/attach      { session_id }
→ 201                                   // binding row inserted, connector serves
```

No SSH, no restart, no shelling into the host.

Risks worth flagging:

- **Captcha rate-limiting** — Signal aggressively requires captcha on programmatic registration; the route shape needs to surface the captcha URL cleanly rather than returning opaque "Captcha required" errors.
- **Phone-number reclamation** — registering a phone already in use elsewhere boots the other device.  Worth a confirmation flag.
- **Secrets shape** — `secrets={phone}` is just an addressing label; the real cryptographic material (libsignal protocol keys) lives in signal-cli's `accounts.json`, not in aios.

**15. Peer inbound edits silently dropped** — `cfc464e`

When a chat partner (not the bot) edits a prior message, signal-cli emits an envelope with `envelope.editMessage.dataMessage` rather than the top-level `envelope.dataMessage` shape `parse_envelope` was matching.  Edits returned `None` from parse → the inbound dispatcher dropped them → **the bot never saw a peer's edit**, so the bot's view of the conversation silently diverged from the chat client's view.

Caught live when Tom edited "@+16575274288 still?" to "@+16575274288 still? - EDIT can you see this?" and asked SmokeBot.  Before the fix the bot would have been answering off the pre-edit content.

Fix: `parse_envelope` falls back to `envelope.editMessage.dataMessage` and sets `edited=True` + `edit_target_timestamp_ms` on the resulting `InboundMessage`.  `build_metadata` propagates these to event metadata; the harness's existing `edited=true` rendering picks it up and now also surfaces `edit_target_timestamp_ms`.  Locked with a parse test against the captured envelope shape.

**16. Telegram outbound `.gif` rendered as a static first frame** — `4f409bd`

Telegram's Bot API treats a `.gif` passed to `sendPhoto` as a static image and only renders the first frame.  `sendAnimation` is the first-class animated-image surface.

`_classify` had `.gif` in `_PHOTO_EXTS` → routed to `bot.send_photo` → Tom saw a static image where a 30-frame Mandelbrot zoom was supposed to be.  The bot worked around it client-side by converting to MP4 via moviepy (installed on demand via `pip install`), which worked but lost the obvious-thing-should-just-work property.

Fix: extract `.gif` from `_PHOTO_EXTS` into a new `_ANIMATION_EXTS`; `_classify` returns `"animation"`; dispatch table maps `"animation"` to `bot.send_animation`.  Multi-attachment media-group path leaves `.gif` falling back to `InputMediaDocument` since Telegram media groups don't have an `InputMediaAnimation` type (platform constraint, not a fixable connector issue).  Locked with a new `_classify` assertion.

**Note**: #16 + the `9617c84` follow-on to #11 together close the full attachment-rendering chain: `.gif` now routes via `send_animation` AND has `Content-Type: image/gif`, so animated GIFs play inline in Telegram.

### Open — observed but not yet fixed

**17. Telegram `parse_mode="html"` runs Markdown→HTML converter; name collides with Bot API semantics** → [#351](https://github.com/eumemic/aios/issues/351)

The connector's `parse_mode: Literal["plain", "html"]` parameter does NOT mean "I'm writing HTML" (Telegram Bot API semantics) — it means "I'm writing markdown, you run it through `markdown_to_telegram_html` before sending".  The docstring describes the actual behavior, but the parameter NAME is the load-bearing signal; smart-enough models infer the standard semantics, write `<a href="...">`, and get literal-text fallout in chat.

Surfaced live when another agent (Ultron) tried `parse_mode="html"` with raw `<a href>` tags and saw the text render as literal characters.  Recommended rename: `parse_mode: Literal["plain", "markdown", "html"]` where `"markdown"` is the current converter behavior and `"html"` becomes true HTML pass-through.

**18. Tool-call dispatched before `serve_connection` registers state → stringified KeyError** → [#352](https://github.com/eumemic/aios/issues/352)

The dispatch SSE and the connection-discovery SSE are independent streams.  Backfill on the tool-call stream can fire while `_on_connection_added` is still running.  When the tool method does `self._conn_state[connection_id]`, it raises `KeyError(connection_id)` and the SDK base stringifies it to `{"error": "'conn_01...'"}` — incomprehensible from the model's POV.

Observed during a telegram restart: connector came up at `17:48:40`, tool-call dispatched at `17:48:43`, got the bare-ID KeyError; retry 1s later succeeded.  Same root cause family as #346 (per-connection state race), different symptom path.

**19. Harness sweep re-wakes session forever on persistent model timeouts** → [#353](https://github.com/eumemic/aios/issues/353)

When wake-step consistently exceeds the 5-min `step_timeout` (e.g. agent on a slow local model with too-large a context window), the sweep keeps re-firing the same wake on every cycle with no backoff or budget.  Each retry pegs the local GPU for 5 minutes and produces nothing.

Observed during the ollama-27B experiment: wake_session[331, 332, 362, 375, 379, 402, 423, ...] all on the same session, alternating between 41s/142s/300s.  Tom's laptop got pegged before we caught it; archive on the session was the only way to break the loop.  Recommended fix: retry budget on consecutive timeouts + promote to `errored` status after N to make the sweep skip the session.

### Pre-existing, not stack-related

**7. ARM64 sandbox image missing** (`ghcr.io/eumemic/aios-sandbox:latest` has no `linux/arm64/v8` manifest)

Blocks every sandbox-dependent capability on Apple Silicon developer machines.  `bash`, `read`, `write`, `edit`, `grep`, `glob` tools fail with `SandboxBackendError`.  Cascades through the smoke: SmokeBot couldn't create attachment files, generate audio, read inbound attachment contents, transcribe voice notes, or play back sound effects.  Every ❌ in the matrix above traces back here.

Fix scope: rebuild `docker/Dockerfile.sandbox` as multi-arch via `docker buildx build --platform linux/amd64,linux/arm64 --push`.

**8. Stale `mcp_toolset` config on existing pre-PR-5 agents**

Agents like `eumemic-bot` still carry an `mcp_toolset` tools-list entry with `mcp_server_name='telegram'`.  Inert at runtime (skipped by `to_openai_tools`, not in `agent.mcp_servers`), but presents as "the agent has 7 tools" in operator-facing reads.  Not a bug — orphaned data from before PR 5's MCP-out, no data migration was written to scrub.

**9. Polluted session history → tool-name hallucination**

The pre-PR-5 `eumemic-bot` session at `sess_01KQXTMDFR8NPCE24ZBEQE9W81` had ~1200 events of `mcp__telegram__telegram_send(...)` calls in its history.  Post-PR-5 the actual exposed tool is plain `telegram_send`, but the model pattern-matched off history and called `mcp__telegram__telegram_send(...)`, which routed to the MCP dispatcher and errored `"MCP server 'telegram' not found"`.  Real architectural cost of the monotonic-context invariant.

**10. `signal-cli` daemon child orphaned on connector kill** — `5f6a0e5`

`pkill -f aios_signal` killed the Python connector but left the spawned `signal-cli daemon` JVM running with the TCP port + SQLite lock on `~/.local/share/signal-cli/data` still held.  Multiple restarts accumulated three orphaned daemons fighting for the lock; the oldest live daemon kept consuming inbounds.

Two correlated causes:
1. `asyncio.run` installs a SIGINT handler by default but not SIGTERM, so the default SIGTERM action (kill the runtime) ran before `teardown` could fire.
2. `pkill -f aios_signal` matches only the Python cmdline, never the JVM cmdline; the JVM survived as an orphan.

Fix: `__main__.py` now traps both SIGINT and SIGTERM via `loop.add_signal_handler`, flipping a stop event that cancels the connector task — the `try/finally` in `HttpConnector.run` then calls `teardown` which SIGTERMs the daemon subprocess and waits with a grace period.  `daemon.py` spawns signal-cli with `start_new_session=True` so a foreground-terminal Ctrl-C no longer reaches the daemon via the controlling terminal.  Locked with three tests: `_serve` cancels the connector on stop, `_serve` returns cleanly if the connector exits first, `_spawn_subprocess` passes `start_new_session=True` to the loop.

**11. Telegram outbound attachments crashed on `pathlib.Path` JSON serialization** — `02b4ac5`

python-telegram-bot's HTTPX request layer JSON-serializes the request body; passing a raw `pathlib.Path` as a `photo` / `document` / etc. kwarg falls into the "unknown object" branch and raises `TypeError: Object of type PosixPath is not JSON serializable`.  Surfaced live in PR 8 smoke when SmokeBot tried to send an outbound image attachment.

Fix: `_read_for_upload(host_path)` reads bytes up-front; `_send_single_media` and `_build_media_group` pass the bytes (not Path) to PTB so the upload path is multipart, not JSON-encoded.  Locked with a regression test asserting every media kwarg handed to PTB is bytes, never a `Path`.

**Follow-on caught in live smoke** — `9617c84`: bytes survived the JSON-serialize layer, but PTB then wraps raw bytes in an `InputFile` with a default filename of `"application.octet-stream"`.  Telegram derives `Content-Type` from the InputFile filename's extension; with the default, every attachment landed as `Content-Type: application/octet-stream` — Tom saw a 713 KB "application.octet-stream" download icon even for animated GIFs that had been correctly routed (post-#16) to `send_animation`.  Wrap the bytes in `InputFile(bytes, filename=host_path.name)` so PTB's multipart writer attaches the source filename, Telegram reads the extension, and the attachment renders in its native player (image preview / inline animation / audio scrubber / etc.).

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

### Bot ↔ bot invisibility in Telegram groups

Observed when SmokeBot and another bot (MetalsAndAIBot / Ultron) were both members of the "Metals and AI" Telegram group: SmokeBot received messages from human members (Tom, David) but NOT from MetalsAndAIBot, and vice versa.  Confirmed via direct test — SmokeBot saw 4 `@mention` strings in a message that tagged 4 entities (`@MetalsAndAIBot @eumemic @EumemicBot @daviduser`) but no inbound stream from MetalsAndAIBot's posts.

Telegram-platform-level constraint: bots can't read other bots' messages in groups, even with privacy mode disabled.  Nothing aios can route around.  Worth noting because the smoke matrix's "bot-to-bot in group" cell is permanently ❌ on Telegram (whereas on Signal it works fine — Signal has no bot-vs-user concept).

### Ollama 27B local model: latency profile vs harness step_timeout

Observed during a brief experiment switching the agent's `model` to `ollama_chat/huihui_ai/qwen3.5-abliterated:27b-Claude` (locally hosted Qwen 3.5 27B Q4_K_M) on an M4 Pro / 48GB Mac:

- **Throughput**: ~18.7 tok/s prompt eval, ~8.6 tok/s generation eval (vs typical M4 Pro range of 15-20 tok/s for 27B Q4_K_M — somewhat below ceiling but functional).
- **Thinking-mode**: Default ON for this Qwen variant.  With thinking enabled, all output budget gets spent in `<thinking>` tokens and the visible `response` field stays empty — directly observed: 10-token probe returned `response=''` with all 10 tokens in `thinking=`.  Disabled via `/no_think` system-prompt prefix + ollama `think: false` option (latter passes through LiteLLM via the agent's `litellm_extra`).
- **Step budget**: The harness's 5-minute `step_timeout` is too short for this combination of (27B model + 30k context window + un-disabled thinking).  Hit reliably during smoke; the sweep then re-fires the timed-out wake forever (see #19).

Not a bug — agent-level configuration choice.  Documented here as a capacity / compat note for anyone wiring a similar local model.

## Recommendations / follow-ups

| # | Action | Scope | Status |
|---|---|---|---|
| 4 | Container resilience: drop-and-continue on 4xx | one commit | ✅ `315c969` |
| 5 | Expose mention metadata + `self_mentioned: bool` | small feature | ✅ `67a14e0` |
| 6 | Surface group-send timestamp via self-echo correlation | small feature | ✅ `a5db125` |
| 7 | Build multi-arch sandbox image (linux/amd64 + linux/arm64) | infra | ✅ workflow `456cf58` (image republishes on next master merge) |
| 8 | Scrub orphaned `mcp_toolset` entries on pre-PR-5 agents | data migration or CLI | open (separate one-off) |
| 10 | Daemon process-group ownership + SIGTERM trap | small fix | ✅ `5f6a0e5` |
| 11 | Telegram attachment shape (Path → bytes → InputFile + filename for Content-Type) | small fix | ✅ `02b4ac5` + `9617c84` |
| 12 | Per-connection `serve_connection` failure isolation (same shape as #4 but for bring-up) | small fix | open → [#346](https://github.com/eumemic/aios/issues/346) |
| 13 | SDK lifecycle hardening — hoist #4 / #10 / #12 + state/focal-channel boilerplate onto `HttpConnector` so telegram inherits the safety rails | medium refactor | open → [#347](https://github.com/eumemic/aios/issues/347) |
| 14 | Signal account registration via the aios API — three routes + management-call SSE so new phones can be onboarded without SSH or restart | medium feature | open → [#348](https://github.com/eumemic/aios/issues/348) |
| 15 | Peer inbound edits silently dropped — accept `envelope.editMessage.dataMessage` shape | small fix | ✅ `cfc464e` |
| 16 | Telegram outbound `.gif` → `send_animation` (not `send_photo`) | small fix | ✅ `4f409bd` |
| 17 | Telegram `parse_mode="html"` parameter name collides with Bot API semantics | small fix | open → [#351](https://github.com/eumemic/aios/issues/351) |
| 18 | Tool-call dispatched before `serve_connection` registers state → stringified KeyError | small fix | open → [#352](https://github.com/eumemic/aios/issues/352) |
| 19 | Harness sweep retries persistent-timeout sessions forever; no backoff or cap | medium fix | open → [#353](https://github.com/eumemic/aios/issues/353) |
| (env) | `aios dev bootstrap` discoverability / harder-to-source-wrong-.env | DX | open → [#349](https://github.com/eumemic/aios/issues/349) |
| (env) | Runtime token connection-subset scope | feature | open → [#350](https://github.com/eumemic/aios/issues/350) |

All ✅ items landed in the `refactor/328-fixup-smoke` follow-up branch (PR #344).  #7's workflow change publishes the multi-arch image on the next push to master.  #8 is pre-existing data hygiene → [#345](https://github.com/eumemic/aios/issues/345).  #9 is the architectural cost of the monotonic-context invariant and not a fixable bug.  #12 → [#346](https://github.com/eumemic/aios/issues/346), #13 → [#347](https://github.com/eumemic/aios/issues/347), #14 → [#348](https://github.com/eumemic/aios/issues/348), #17 → [#351](https://github.com/eumemic/aios/issues/351), #18 → [#352](https://github.com/eumemic/aios/issues/352), and #19 → [#353](https://github.com/eumemic/aios/issues/353) are all open follow-ups with their own targeted PRs.
