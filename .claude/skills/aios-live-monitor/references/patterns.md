# aios live-monitor patterns

Detail not covered in SKILL.md. Read this when needed.

## Chat-filter output format

The filter at `scripts/chat_filter.py` emits one line per message-kind event:

| event | line shape |
|---|---|
| `role: user` (text) | `[<seq>] USER<sender@channel> <text preview, 200 chars>` |
| `role: user` (multipart) | `[<seq>] USER<sender@channel> [<part-types>] <first text part, 200>` |
| `role: user` w/ attachments | additional `      ATT: <fn> <ct> <sz>B in_sandbox=<path>` rows beneath |
| `role: assistant` text (visible) | `[<seq>] ASSISTANT <text preview, 200>` |
| `role: assistant` text (`INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER` prefix) | `[<seq>] ASSISTANT [monologue: <preview, 140>]` |
| `role: assistant` tool call | `[<seq>] ASSISTANT call <name>(<k=v, k=v, ...>)` (first 3 args, values ≤80 each) |
| `role: tool` (text result) | `[<seq>] TOOL <name> -> <preview, 200>` |
| `role: tool` (multipart) | `[<seq>] TOOL <name> -> [<part-types>]` |

Spans, lifecycle, and metadata frames are dropped. Stdin is parsed line-by-line; the filter unwraps the SSE envelope (`{"event":"event","data":"<JSON of the event>"}`) once before parsing the inner event.

## Error-filter false positives to ignore

The grep pattern `error|ERROR|exception|Exception|traceback|Traceback|RuntimeError|KeyError|FAILED` is intentionally broad. Known benign matches (investigate only if pattern looks new):

- `signal.daemon.stderr` lines containing `NotRegisteredException` for unregistered phones in `MultiAccountManager` — signal-cli's normal startup chatter on multi-account mode.
- `mcp_tool.completed` info lines that happen to contain `is_error: false`.
- The literal string `Exception` appearing inside JSON-encoded inbound envelopes (signal-cli serializes typing/receipt envelopes containing class names).
- `mcp_pool.close_entry_failed` warnings during graceful shutdown — cleanup noise, not a crash trigger.

If a real error fires, look for surrounding context — the grep buffers per-line, so the line above/below isn't visible in the notification. Use `Read` on `.logs/worker.log` around the timestamp.

## Why the chat monitor sometimes ends silently

The Monitor stream completes (sends a "stream ended" notification) when:

- The api process dies — SSE connection closed by server
- The script gets EOF on stdin — usually because invoked without the upstream `aios sessions stream` pipe (e.g., positional args mistake)
- `aios sessions stream` itself exits — the session was deleted, or the server shut down between events

The chat filter never crashes on bad input — bad lines are silently dropped via `try: json.loads except JSONDecodeError: continue`. So a "stream ended" with no events almost always means stdin was empty or the api was unreachable.

## Bridging the gap after a restart

After step 5 of the restart cadence, the new chat monitor starts at the new tail seq — events between old and new are not backfilled. To inspect what happened during downtime + replay:

```bash
uv run aios --format json sessions events <id> --after-seq <old_tail> --limit 200 \
  | python3 -c "
import json, sys
for e in json.load(sys.stdin)['data']:
    if e.get('kind') != 'message': continue
    d = e['data']; r = d['role']
    md = d.get('metadata') or {}
    s = md.get('sender') or 'asst'
    print(f'[{e[\"seq\"]}] {r} {s}: {str(d.get(\"content\",\"\"))[:200]}')"
```

Pages by setting `--after-seq` to the last seq of the previous batch.

## Customizing the chat filter

The filter is intentionally simple (98 lines, no deps beyond stdlib). To extend (e.g., decode tool-call args differently, surface span events, change preview width), edit a fork in the smoke worktree under `/tmp/` first, validate against a live stream, then update the in-skill copy.

Output conventions to preserve:

- Lines start with `[<seq>] <ROLE>` so the user can grep notifications.
- Truncate previews to `[:200]` (or `[:140]` for monologues) — wider notifications wrap badly in chat.
- `print(..., flush=True)` so the runtime's 200ms batcher sees lines as they arrive.

## Cost rationale for the --after-seq rule

`aios sessions stream <id>` SSE-backfills via `LISTEN/NOTIFY` plus an initial replay of the events table from the requested seq. Without `--after-seq`, the replay covers the whole session. Each backfilled event becomes a Monitor stdout line, then a `<task-notification>` injected into the assistant's context.

For the long-running factchecker session (51k+ events), that's ~200k tokens of pure noise injected on a single arm. Use the preflight script.
