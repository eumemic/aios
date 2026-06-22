# Run & session observability: polling contract and event schemas

Reference for anyone watching a run or session from the outside — an operator
tailing with `aios`, an SDK consumer, or an agent awaiting a sub-run as an MCP
tool. It pins down three contracts surfaced by the #1068 dogfood (#1140).

## (a) Terminal state is `status` / `outcome`, never `state`

A run's lifecycle field is **`status`**. There is **no `state` field** anywhere
on a run row (`WfRun`) or on the unified awaiter's `AwaitResponse`.

A watcher that keys on `.state` reads `None` forever — even after `output` is
already populated — because the field simply does not exist in the JSON. This is
exactly the #1068 symptom: the awaiter "returned `state:None` while `output` was
already populated."

What to poll:

| Surface | Field to poll | Terminal when |
| --- | --- | --- |
| `GET /v1/runs/{id}` (`WfRun`) | `status` | `status ∈ {completed, errored, cancelled}` |
| `GET /v1/tasks/{task_id}/await` (`AwaitResponse`) | `outcome` | `outcome` is non-null (`ok` / `errored` / `cancelled`) |

`GET /v1/tasks/{task_id}/await` is the one await-a-completion primitive
(over both a run and a session servicer): it blocks until the task is
terminal or `timeout` seconds elapse. On timeout it returns `outcome=null` (still
pending); call again to keep blocking. The completion payload is `{outcome:"ok",
result}` (success) or `{outcome:"errored"|"cancelled", error}` (failure). The
`task_id` is the servicer id; the response carries no `state`, `done`, or
`run_status` field — `outcome` alone is the terminal discriminant.

> Sessions, by contrast, expose `status` (`SessionStatus`) and are not
> persisted the way runs are — but the same rule holds: poll `status` (or, for a
> task, `outcome`), there is no `state`.

## (b) An empty event page is not a "session/run reset"

`GET /v1/sessions/{id}/events` and `GET /v1/runs/{id}/events` can return an
empty `items` list **transiently** — notably on back-to-back polls under load,
which then recover on the next read. An empty page only means *no events match
this page right now*; it is **never** a signal that the log was cleared or the
session/run was reset.

Causes of a legitimately-empty page:

- A forward read whose `after_seq` is already at (or past) the current tail —
  i.e. "nothing new yet."
- Back-to-back polls racing the writer: the second read can momentarily observe
  no new rows before the next event commits and notifies.
- A filtered read (`?kind=`, `?error_only=`) with no matching rows in this
  window.

Correct consumer behavior: **page by `seq`** (carry the last `seq` you saw and
read forward from it), and treat an empty page as "poll again," not as a reset.
Do not infer log truncation, session loss, or run cancellation from emptiness —
those are reported by `status` (see (a)), never by an empty list. For push-based
consumption prefer the SSE `/stream` endpoint, which is backed by Postgres
`LISTEN`/`NOTIFY`.

## (c) Two event schemas coexist — don't assume one shape

Run events and session events are **different shapes**. A consumer that watches
both surfaces must branch on which endpoint produced the event.

### Run events — `{type, payload, seq}`

`GET /v1/runs/{id}/events` returns `WfRunEvent` rows (the run's append-only
journal):

```json
{ "type": "run_started | call_started | call_result | annotation | run_completed",
  "payload": { "...": "..." },
  "seq": 1 }
```

`type` is the run journal's frame type. An `annotation` payload is a journaled
progress marker, `{"kind": "log" | "phase", "text": ...}` — note this inner
`kind` is *not* the session-event `kind` below.

### Session events — `{kind, data}`

`GET /v1/sessions/{id}/events` returns `Event` rows (the session log). These are
the **child-session** events produced when a run spawns agent sessions:

```json
{ "kind": "message | lifecycle | span | interrupt",
  "data": { "...": "..." },
  "seq": 1 }
```

- `kind == "span"` — observability markers around model/tool calls.
- `kind == "lifecycle"` — session state transitions (turn started/ended, status
  changes, stop_reason).
- `kind == "message"` — a chat-completions message dict; `data.role` ∈
  `user | assistant | tool`.

### At a glance

| | Run event | Session event |
| --- | --- | --- |
| Endpoint | `/v1/runs/{id}/events` | `/v1/sessions/{id}/events` |
| Model | `WfRunEvent` | `Event` |
| Discriminator | `type` | `kind` |
| Payload field | `payload` | `data` |
| Values | run_started / call_started / call_result / annotation / run_completed | message / lifecycle / span / interrupt |

Source of truth: `aios.models.workflows.WfRunEvent` and
`aios.models.events.Event`.

## (d) Per-run cost / token / wall-clock usage on the read path (#1324)

`GET /v1/runs/{id}` (`WfRun`) and `GET /v1/runs` (`list_runs`) carry a `usage`
object — the run's realized spend, the machine-observer's cost-substrate. It is
summed over the run's direct child sessions via the same `run_children_usage`
source the in-script `budget()` builtin consumes, so the run's `budget_usd`
*ceiling* and its `usage.cost_microusd` *spend* are both legible from the read
path (not buried in a builtin). On `list_runs` the whole page is enriched in one
batched aggregate — no per-run fan-out.

`usage` fields (`WfRunUsage`):

| Field | Meaning |
| --- | --- |
| `cost_microusd` | Summed child-session cost (µUSD). A childless run sums to a real `0`. |
| `input_tokens` / `output_tokens` | Summed child-session tokens. |
| `cache_read_input_tokens` / `cache_creation_input_tokens` | Summed child-session cache tokens. |
| `iteration_count` | The run's wake/step count. **Always `null` today** — the host keeps no per-run iteration counter on any substrate; reserved for when one lands. |
| `wall_clock_ms` | `updated_at − created_at` (ms) for a **terminal** run; `null` while the run is still live (its `updated_at` is a moving "last touched" stamp, not an end). |

**Fail-loud absence.** Every field is `int | null`, and a value that cannot be
determined is returned as **explicit `null`**, never a silent `0` or an omitted
key (cf. the `vault_ids:null` read-path disease). An observer reads `null` as
*cannot-determine* and fails loud; it reads `0` as a real observed zero. Do not
conflate the two.
