---
name: aios-live-monitor
description: This skill should be used when the user asks to "watch the live aios session", "tail factchecker", "set up live monitoring on aios", "monitor the chat", "tail worker errors", "narrate aios session activity", or runs a smoke test that requires persistent observation of a running aios api+worker pair. Provides a chat-stream filter, error-grep pattern, and pre-flight probe to arm Monitor-tool tasks safely (always with --after-seq to avoid backfill blowing up the context window).
---

# aios live monitoring

Persistent observation of a running aios session (api + worker pair) for live-testing, smoke runs, and AFK narration. Two streams are usually armed in parallel: a per-session **chat monitor** (one line per turn) and a worker-log **error filter** (silent-failure detector).

## When to use

- Smoke-testing a PR stack against real sessions before merging
- Watching a fix land in a running session (did the connector behavior change?)
- AFK monitoring with proactive narration (per saved feedback memory: post a one-liner per turn)
- Diagnosing a stuck session, hang, or unexpected silence

Skip the chat monitor for one-off snapshots — `aios sessions events` is faster.

## The fast path

From the running aios worktree (the one whose `.env` points at the api), arm both monitors via the bundled preflight script:

```bash
.claude/skills/aios-live-monitor/scripts/preflight.sh <session_id>
```

It checks api reachability, reads `last_event_seq` directly via `aios sessions get` (single RTT), and prints two ready-to-paste commands — one for the chat monitor, one for the error filter. Wrap each in a `Monitor(... persistent=True, timeout_ms=3600000 ...)` call.

## The --after-seq rule (critical)

`aios sessions stream <id>` SSE-backfills the **entire** session before live frames. For a 50k-event session that becomes ~50k notifications — each one a `<task-notification>` injected into context. Cost: ~200k tokens of pure noise.

**Always** pass `--after-seq <current_tail>` so the stream starts at the live edge with no backfill. The preflight script computes this automatically. If a monitor accidentally fires without `--after-seq`, stop it immediately with `TaskStop` and re-arm with the correct seq.

## Chat monitor invocation shape

```
Monitor(
  description="<session> chat from seq <N>",
  command="cd <worktree> && set -a && source .env && set +a && \\
    uv run aios sessions stream <session_id> --after-seq <N> --raw 2>&1 | \\
    python3 .claude/skills/aios-live-monitor/scripts/chat_filter.py",
  timeout_ms=3600000,
  persistent=True,
)
```

The filter at `scripts/chat_filter.py` formats one line per message-kind event (user / assistant text / tool call / tool result / monologue). It reads from stdin only — invoking it with positional args by mistake exits silently with no output.

## Error filter invocation shape

```
Monitor(
  description="worker errors",
  command="tail -f <worktree>/.logs/worker.log | grep --line-buffered -E 'error|ERROR|exception|Exception|traceback|Traceback|RuntimeError|KeyError|FAILED|step\\.\\w+_failed|HTTPStatusError|BadRequestError|raise|provider_error|mark_read_failed|msgstore_put_failed' | grep --line-buffered -v heartbeat",
  timeout_ms=3600000,
  persistent=True,
)
```

`--line-buffered` is essential — without it `grep` batches and event timing is lost.

### Broad alternation: silence is not success

The alternation above looks redundant on purpose.  A narrow filter that
only matches `error|exception|Traceback` will MISS real failures whose log
line uses none of those tokens — observed during the WhatsApp PR-5 smoke
where a model-side LiteLLM 400 surfaced as `step.litellm_failed` and the
narrow grep never fired, so the session sat silent and the operator
correctly asked: *"hmm, no response... are you monitoring error
channels?"*  Lesson: if the user reports nothing happening, the bug is
probably in the filter, not the system.  Broaden the alternation and
re-arm.  Specific signatures to include:

* `step\.\w+_failed` — harness-level failures (litellm_failed,
  tool_dispatch_failed, etc.)
* `HTTPStatusError`, `BadRequestError`, `ProviderException` — LLM /
  external-API rejections
* `RuntimeError`, `KeyError`, `AttributeError`, `TypeError` — Python
  internal failures the harness re-raises
* connector-specific `*_failed` events (`mark_read_failed`,
  `msgstore_put_failed`, `media_download_failed`)

The `grep -v heartbeat` tail filter cuts noise from periodic
`heartbeat.touch_failed` warnings on macOS (the worker can't `touch
/var/run/aios-worker-alive` without root — benign).

## CLI snapshots vs. SSE stream

Use the SSE monitor for *live* narration; use the CLI for *static*
look-back at past events.  The CLI's `--after-seq N` is an **exclusive
lower bound** — events with `seq > N` only.

```bash
# Everything: --after-seq 0 (default) returns all events
uv run aios sessions events <session_id>

# Just messages (user / assistant / tool / tool_result)
uv run aios sessions events <session_id> --kind message --all

# Events created since N (exclusive); pass N-1 to include event N
uv run aios sessions events <session_id> --after-seq <N-1>

# JSON for programmatic inspection
uv run aios -f json sessions events <session_id> --kind message --all
```

**Footgun:** if `aios sessions get` shows `last_event_seq=230` AND
`aios sessions events --after-seq 230` returns nothing, it means seq=230
is the actual tail and the session is quiet — *not* that the events
weren't persisted.  Don't drop to `docker exec psql` to "investigate" —
adjust the `--after-seq` argument.  (Observed during WhatsApp PR-5
smoke.)

## Restart cadence

Per saved project feedback, every commit on the smoke branch auto-triggers a stop+restart on the new HEAD. Protocol:

1. `pkill -f "aios api"; pkill -f "aios worker"`
2. Restart api+worker in the smoke worktree, redirect to `.logs/`
3. Wait ~5s, confirm `signal.account.ready` and `connector.running` in worker.log
4. **Stop the existing chat monitor with `TaskStop`** — its SSE stream points at a dead API
5. Re-run `preflight.sh` to get the new tail seq
6. Re-arm with the new `--after-seq`

## Daemon-side mysteries: the instrument-rebuild-restart-repro loop

When smoke surfaces an inbound that the model never perceives (or a
similar mystery that lives inside the Go connector daemon rather than
the harness), static reading hits a wall fast — the WhatsApp protocol,
whatsmeow's event types, and the daemon's translator have too many
shapes to enumerate.  The reliable pattern is to add a one-shot
diagnostic log, rebuild, restart, drive a controlled repro, read the
log, then remove the log.

For the WhatsApp daemon specifically:

```go
// In connectors/whatsapp/daemon/internal/wameow/inbound.go, at the
// point where you suspect data is being dropped:
c.log.Info("wameow.<mystery_name>",
    "id", string(e.Info.ID),
    "msg_struct", fmt.Sprintf("%+v", e.Message),
)
```

Then rebuild via the standard cross-compile:

```bash
docker run --rm \
  -v "$PWD/connectors/whatsapp/daemon:/src" \
  -v aios-whatsapp-gocache:/go \
  -v /tmp/whatsapp-build:/out \
  -w /src -e GOPATH=/go -e GOCACHE=/go/build-cache \
  -e GOOS=darwin -e GOARCH=arm64 -e CGO_ENABLED=0 \
  golang:1.25 go build -o /out/whatsapp-daemon ./cmd/whatsapp-daemon
```

Restart just the connector (api/worker stay up — they don't hold the
diagnostic):

```bash
pkill -f "uv run python -m aios_whatsapp" && sleep 3 && \
  AIOS_URL=http://127.0.0.1:$AIOS_API_PORT \
  AIOS_RUNTIME_TOKEN=$RUNTIME_TOKEN \
  AIOS_WHATSAPP_DAEMON_BIN=/tmp/whatsapp-build/whatsapp-daemon \
  AIOS_WHATSAPP_DATA_DIR=/tmp/aios-whatsapp-smoke-data \
  nohup uv run python -m aios_whatsapp >> .logs/connector.log 2>&1 &
```

Drive the repro on the user's phone.  The diagnostic line shows up in
`.logs/connector.log` as a `whatsapp.daemon.stream` entry containing
the JSON the daemon emitted.  Use `%+v` formatting in `fmt.Sprintf` for
protobuf structs — the field names print verbatim so you can spot
which sub-field actually holds the payload.

Remove the log after fixing.  Don't leave info-level diagnostics in the
production hot path; they're discoverable in git history if needed.

**Use this loop for**: inbound message types the parser drops as
"no_signal", reactions / edits / revokes that don't reach the model,
attachment paths that fail to materialise, any "is whatsmeow sending
something I'm not handling?" question.  This loop produced #111's root
cause (SecretEncryptedMessage_MESSAGE_EDIT) in ~30 minutes of total
turnaround vs. open-ended reading of whatsmeow source.

## Narration etiquette during AFK monitoring

When the user is AFK (e.g., gave a "Dev, ..." instruction via Signal), per saved feedback:

- One-liner per turn: who spoke, what tool fired, what landed
- Don't narrate every internal monologue — only flag when the model is engaging or something looks off
- Flag real errors from the error filter promptly; don't auto-act on benign false positives (see references for the known set)

## Additional resources

- **`scripts/preflight.sh`** — reachability + tail-seq probe; emits ready-to-paste Monitor commands
- **`scripts/chat_filter.py`** — stdin-only formatting filter for `aios sessions stream --raw`
- **`references/patterns.md`** — chat-line shapes, error-filter false positives to ignore, gap-bridging probes after restart, customization conventions for the filter
