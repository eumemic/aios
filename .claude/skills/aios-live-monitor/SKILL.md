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
  command="tail -f <worktree>/.logs/worker.log | grep --line-buffered -E 'error|ERROR|exception|Exception|traceback|Traceback|RuntimeError|KeyError|FAILED'",
  timeout_ms=3600000,
  persistent=True,
)
```

`--line-buffered` is essential — without it `grep` batches and event timing is lost.

## Restart cadence

Per saved project feedback, every commit on the smoke branch auto-triggers a stop+restart on the new HEAD. Protocol:

1. `pkill -f "aios api"; pkill -f "aios worker"`
2. Restart api+worker in the smoke worktree, redirect to `.logs/`
3. Wait ~5s, confirm `signal.account.ready` and `connector.running` in worker.log
4. **Stop the existing chat monitor with `TaskStop`** — its SSE stream points at a dead API
5. Re-run `preflight.sh` to get the new tail seq
6. Re-arm with the new `--after-seq`

## Narration etiquette during AFK monitoring

When the user is AFK (e.g., gave a "Dev, ..." instruction via Signal), per saved feedback:

- One-liner per turn: who spoke, what tool fired, what landed
- Don't narrate every internal monologue — only flag when the model is engaging or something looks off
- Flag real errors from the error filter promptly; don't auto-act on benign false positives (see references for the known set)

## Additional resources

- **`scripts/preflight.sh`** — reachability + tail-seq probe; emits ready-to-paste Monitor commands
- **`scripts/chat_filter.py`** — stdin-only formatting filter for `aios sessions stream --raw`
- **`references/patterns.md`** — chat-line shapes, error-filter false positives to ignore, gap-bridging probes after restart, customization conventions for the filter
