# Design contract — `triggers` slice 2 (#819)

**Status: FROZEN (rev 2).** Contract for the `workflow` action kind, the `run_completion` source
kind, and the `trigger_runs` audit table — built as specified; the implementation landed on
`feat/triggers-slice2`. Produced by a research pass over the live code (every load-bearing claim
grounded `file:line`), adversarially verified per question, and cross-checked for coherence;
empirical probes were executed against a throwaway Postgres 16 (hostile-row CHECK execution in
both directions, jsonb numeric-expansion round-trip, byte-identity of predicate constants,
bare-DROP-COLUMN auto-drop behavior). The §11 sign-offs are RESOLVED (see the table there); rev-2
deltas from rev 1, all build-time:

- **The migration is 0085** (`down_revision="0084"` — master took 0084 for the session snapshot
  pointer while this was in design). Every "0084" in the body reads as 0085.
- **Resolution #2 (IN):** the migration carries `triggers_run_completion_no_next_fire CHECK
  (source <> 'run_completion' OR next_fire IS NULL)` — the §6 residual promoted to a constraint.
- **The 0083-style diagnostic validating SELECT is INCLUDED** in the migration (shares the
  predicate constants; expected empty — no backfill) per the build directive's same-craft rule.
- **Seam A:** the shared write-path validator's account-scoped session read is its FIRST,
  UNCONDITIONAL statement (the §7 sketch's ordering understated §4's cross-tenant fix, which must
  cover ALL action kinds incl. slice-1 `sandbox_command`); the §7 sketch below is corrected.
- **Seam B:** `list_trigger_runs` keys off the denormalized `(account_id, owner_session_id,
  trigger_name)` — already the §2 stance, now also the route's contract (history survives trigger
  deletion; name reuse merges incarnations, disambiguated by `trigger_id` on the rows).
- **Resolution #7 (tombstones):** one-shot timer skips write a `skipped` audit row (§2's sign-off
  option (b)).

Prime directive carried over from slice 1: **additive future** — zero slice-1 row rewrites; a
future operator-owned slice (nullable `owner_session_id`) stays additive. Second directive:
**zero slice-1 behavioral regression** — `scheduler.py` and the claim/MIN queries take a zero-line
diff; the tick defer tail (per-trigger `queueing_lock`, task-id-only payload) stays frozen per D5.

---

## 0. Envelope summary

```jsonc
// NEW action kind (wire shape; environment_id deliberately NOT a field — §7):
{
  "kind": "workflow",
  "workflow_id": "wf_01JX…",
  "workflow_version": null,        // null = float (run current version at fire);
                                   // int = drift-assert pin (must == current at write; fire errors on drift)
  "input_template": {"report": "daily"},   // arbitrary JSON incl. null; ≤16_384 serialized bytes (write-path)
  "vault_ids": ["vlt_01JX…"]       // pinned list; subset-of-owner enforced at EVERY fire by create_run
}

// NEW source kind:
{
  "kind": "run_completion",
  "workflow_id": "wf_01JX…",                          // the watched workflow (account-scoped)
  "statuses": ["completed", "errored", "cancelled"]   // default all three, materialized at write
}
```

```text
NEW table   trigger_runs   — one row per fire; the per-event claim/dedup carrier (D3)
NEW column  triggers.environment_id  — nullable FK → environments(id); set iff action kind = 'workflow'
NEW kwarg   harness.run_trigger(trigger_id, trigger_run_id=None)  — the §1.2 additive per-event context
NEW index   triggers_run_completion_watch  — partial expression index for completion matching
```

Fire-path topology: `_complete_run`/`_cancel_run` (the sole `run_completed` writers) match watching
triggers and insert one `pending` `trigger_runs` row per match **inside the completion
transaction**; after commit, one `harness.run_trigger` job per fire is deferred with
`queueing_lock=f"trigger_run:{id}"`; the runner claims the row `pending→running` and executes; the
30s periodic sweep re-defers stale `pending` rows (lost defers) — exactly-once dispatch, at-most-once
execution after claim.

---

## 1. The reactive `run_completion` fire path (Q1)

### Recommendation

Dispatch **inline in the run-completion transaction**. `_complete_run` and `_cancel_run`
(`src/aios/workflows/step.py:621-677`) are the only writers of `run_completed` + `set_run_terminal`
(verified by grep; the cancel API is signal-driven and the flip lands in the step under
`lock=run_id`), so the completion txn is the single consistency point: "the run completed" and
"these fires are owed" commit atomically. After commit, defer **one job per fire** carrying the
`trigger_run_id` kwarg (§1.2's reserved additive per-event context). NOTIFY+listener lost (no
global completion channel exists — `append_run_event` notifies only the per-run
`wf_run_events_{run_id}`, `db/queries/workflows.py:727` — and pg_notify is fire-and-forget across
restarts, so recovering lost events would need exactly the durable dispatch record the inline
design writes anyway). Scheduler extension lost (the scheduler is timer-shaped; event matching in
it means polling — a parallel matcher, against the reuse principle).

**Exactly-once dispatch.** The fire-insert is gated on `append_run_event` actually inserting:
the journal memo (`UNIQUE NULLS NOT DISTINCT (run_id, call_key, type)`, migration 0064:96-100)
guarantees exactly one `run_completed` insert per run EVER commits — under procrastinate dual
execution the loser's append returns `None` and dispatches nothing. Crash before commit rolls
everything back (stalled jobs are marked failed, never re-run; the needs-step sweep excludes
terminal runs; `run_workflow_step`'s entry guard no-ops on terminal). Crash **after** commit but
before the defer leaves a durable `pending` row that the existing 30s periodic sweep
(`harness/worker.py:387-415`) re-defers — the repo's standard
commit-intent/defer-best-effort/sweep-backstop idiom (`wf_run_signals` precedent,
`db/queries/workflows.py:616-618`). The defer carries
`queueing_lock=f"trigger_run:{trigger_run_id}"` — keyed per EVENT, never the bare trigger id
(§1.2-compliant; matches every sibling defer chokepoint: queueing_lock dedups todo-state, the row
claim guards doing-state, exactly as `wake_session`/`wake_workflow` pair them).

**At-most-once after claim.** The runner claims the row `pending→running`; a worker crash mid-action
leaves an observable `running` row and the fire is NOT retried (re-firing could double-launch a
run — the one-shot delete-before-fire trade, but observable instead of silent). The sweep re-defers
**`pending` only**.

### Concrete shape

Matching + fire-intent insert (one query function — Q4's authority framing merged in;
`db/queries/__init__.py`, triggers section):

```python
class TriggerFireRef(NamedTuple):
    trigger_run_id: str
    trigger_id: str

async def insert_run_completion_fires(
    conn, *, account_id: str, workflow_id: str, run_id: str, status: str
) -> list[TriggerFireRef]:
    """MUST run inside the run-completion transaction. The ``t.account_id = $1``
    conjunct is the tenant boundary (write-path validation is UX; this is
    enforcement): a trigger is only ever handed run data its owner could already
    read via the account-scoped ``await_run``/``get_run`` — watching = reading,
    no new read authority."""
    rows = await conn.fetch(
        """
        SELECT t.id, t.account_id, t.owner_session_id, t.name
        FROM triggers t
        JOIN sessions s ON s.id = t.owner_session_id
        WHERE t.source = 'run_completion'
          AND t.account_id = $1
          AND t.source_spec ->> 'workflow_id' = $2
          AND t.source_spec -> 'statuses' ? $3
          AND t.enabled
          AND s.archived_at IS NULL
        ORDER BY t.created_at
        """,
        account_id, workflow_id, status,
    )
    event = json.dumps({"run_id": run_id, "workflow_id": workflow_id, "status": status})
    refs: list[TriggerFireRef] = []
    for r in rows:  # bounded by the per-account enabled-trigger cap
        trigger_run_id = make_id(TRIGGER_RUN)
        await conn.execute(
            "INSERT INTO trigger_runs (id, trigger_id, account_id, owner_session_id,"
            " trigger_name, trigger_context, event, status)"
            " VALUES ($1, $2, $3, $4, $5, 'run_completion', $6::jsonb, 'pending')",
            trigger_run_id, r["id"], r["account_id"], r["owner_session_id"], r["name"], event,
        )
        refs.append(TriggerFireRef(trigger_run_id, r["id"]))
    return refs
```

Dispatch hook (`workflows/step.py` — identical tail in `_cancel_run` with `status="cancelled"`;
`_complete_run` passes `"errored" if is_error else "completed"`):

```python
fires: list[db_queries.TriggerFireRef] = []
async with conn.transaction():
    inserted = await wf_queries.append_run_event(..., type="run_completed", payload=payload)
    await wf_queries.set_run_terminal(...)               # unchanged
    if inserted is not None:                             # exactly-once gate (journal memo)
        fires = await db_queries.insert_run_completion_fires(
            conn, account_id=run.account_id, workflow_id=run.workflow_id,
            run_id=run.id, status=...,
        )
# Post-commit: intents are durable; defer is best-effort (loss → sweep re-defers 'pending').
for fire in fires:
    try:
        await defer_trigger_fire(fire.trigger_id, fire.trigger_run_id)
    except Exception:
        log.exception("trigger.fire_defer_failed", trigger_run_id=fire.trigger_run_id)
```

Defer helper (`services/wake.py`, beside `defer_wake`/`defer_run_wake`) + task kwarg
(`harness/tasks.py`):

```python
async def defer_trigger_fire(trigger_id: str, trigger_run_id: str) -> None:
    """One job PER event fire (§1.2). queueing_lock keys the FIRE (trigger_run id),
    never the bare trigger id — distinct completions never coalesce; a sweep
    re-defer racing a queued job dedups. Swallows AlreadyEnqueued."""
    from aios.harness.procrastinate_app import app
    try:
        await app.configure_task(
            "harness.run_trigger", queueing_lock=f"trigger_run:{trigger_run_id}"
        ).defer_async(trigger_id=trigger_id, trigger_run_id=trigger_run_id)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("trigger.fire_already_enqueued", trigger_run_id=trigger_run_id)

@app.task(name="harness.run_trigger", queue="sessions", retry=False, pass_context=False)
async def run_trigger(trigger_id: str, trigger_run_id: str | None = None) -> None:
    await run_trigger_step(trigger_id, trigger_run_id=trigger_run_id)
```

Runner lifecycle (`harness/trigger_runner.py`) — **the lifecycle arm derives from the fire's
ORIGIN, never the reloaded row's `source`** (the row is user-mutable between match and fire; a
mid-flight `run_completion → one_shot` source replacement must not route an event fire into the
one-shot DELETE arm and destroy the user's fresh trigger):

```python
async def run_trigger_step(trigger_id: str, trigger_run_id: str | None = None) -> None:
    started_at = datetime.now(UTC)        # THE fire timestamp: claim, envelope, record — one value
    event: dict[str, Any] | None = None
    if trigger_run_id is not None:
        # Event fire: claim the carrier row INSTEAD of the tick's running_since claim.
        async with pool.acquire() as conn:
            event = await queries.claim_trigger_run(conn, trigger_run_id, started_at=started_at)
        if event is None:
            log.info("trigger.fire_already_claimed", trigger_run_id=trigger_run_id)
            return
    is_event_fire = trigger_run_id is not None
    try:
        trigger = await queries.unscoped_get_trigger_row(conn, trigger_id)
    except NotFoundError:
        if is_event_fire:   # trigger deleted between match and fire — finalize the claim row
            ... finalize_trigger_run(status="skipped", error_summary="trigger deleted") ...
        return
    is_one_shot = (not is_event_fire) and trigger.source == "one_shot"
    # skip_archived / skip_disabled: event fires take the record arm + finalize the
    # carrier row as 'skipped' — NEVER the one-shot delete, NEVER an unfinished row
    # (an unfinished claim row would be re-deferred by the sweep forever).
    ...
    status, error_summary, result_id = await <action executor>(trigger, action, event=event, started_at=started_at)
    if is_one_shot:
        ...existing one-shot tail; plus the timer audit insert (§2)...
    else:
        async with pool.acquire() as conn, conn.transaction():
            failures = await queries.record_trigger_fire(conn, trigger_id, status=status, fired_at=started_at)
            if is_event_fire:
                await queries.finalize_trigger_run(conn, trigger_run_id, status=status,
                                                   error_summary=error_summary, result_id=result_id)
            else:
                await queries.record_trigger_run(...)    # timer audit row (§2)
            if failures is not None and failures == MAX_CONSECUTIVE_FAILURES:
                await queries.disable_trigger(conn, trigger_id)
        # auto-disable surfacing after commit — unchanged
```

`record_trigger_fire` becomes atomic — fixes the read-modify-write race at
`trigger_runner.py:113`, which becomes *reachable* once two fires of one trigger run concurrently
(event fires have no `running_since`/queueing-lock serialization). Independently derived three
times in the research pass; behavior-identical for single-flight cron:

```sql
UPDATE triggers
SET running_since = NULL, last_fire_at = $1, last_fire_status = $2,
    consecutive_failures = CASE
        WHEN $2 = 'ok' THEN 0
        WHEN $2 = 'skipped' THEN consecutive_failures
        ELSE consecutive_failures + 1
    END,
    updated_at = $1
WHERE id = $3
RETURNING consecutive_failures
```

Returns `int | None` — `None` = row deleted mid-fire (benign race; the caller's disable check is
None-tolerant, and for event fires `finalize_trigger_run` in the same txn still completes the
audit). The disable + surfaced message gate on `returned == MAX_CONSECUTIVE_FAILURES` **exactly**
(not `>=`), so in-flight straggler fires that increment past 5 after disable don't re-spam the
owner.

Sweep backstop (chained into `_periodic_sweep`, `harness/worker.py:387-415`):

```python
refs = await queries.list_pending_trigger_run_refs(pool, older_than_seconds=60.0)
for ref in refs:
    await defer_trigger_fire(ref.trigger_id, ref.trigger_run_id)   # claim makes re-defer safe
# observability for the at-most-once tail (not a retry — deliberate):
n = await queries.count_stuck_running_trigger_runs(pool, older_than_seconds=7200.0)
if n:
    log.warning("trigger.fires_stuck_running", count=n)
```

### Source member + service deltas

```python
RunTerminalStatus = Literal["completed", "errored", "cancelled"]

class RunCompletionSource(BaseModel):
    """Reactive source: fires once per terminal completion of any run of the
    watched workflow whose status is in ``statuses``. No ``next_fire`` — never
    scheduled by the tick; dispatched from the run-completion transaction."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["run_completion"] = "run_completion"
    workflow_id: str
    statuses: list[RunTerminalStatus] = Field(
        default_factory=lambda: ["completed", "errored", "cancelled"], min_length=1,
    )

class RunCompletionSourceReplace(RunCompletionSource):
    """Update-side variant (§2.2 Replace rule): ``statuses`` REQUIRED, so a partial
    source on update 422s instead of silently resetting a narrowed filter to all-three.
    First SOURCE member with a defaulted field → first TriggerSourceReplace union."""
    statuses: list[RunTerminalStatus] = Field(min_length=1)

TriggerSource = Annotated[CronSource | OneShotSource | RunCompletionSource, Field(discriminator="kind")]
TriggerSourceReplace = Annotated[
    CronSource | OneShotSource | RunCompletionSourceReplace, Field(discriminator="kind")
]
# TriggerUpdate.source: TriggerSourceReplace | None  (read path stays on the create-side union)
```

`compute_initial_next_fire` returns `datetime | None` — `None` for `RunCompletionSource`
(unschedulable by the tick BY PREDICATE, the §3 invariant doing the job it was reserved for).
`update_trigger`'s Ellipsis sentinel already accepts explicit `None`. The §2.4 matrix holds:
re-enable/source-replace recomputes `next_fire → None`; the past-`fire_at` rejection stays
`isinstance(merged_source, OneShotSource)`-gated. Write path validates the watched workflow exists
account-scoped (the silent-dead-watch analog of the §7 occurrence check — fire-time never surfaces
a never-matching watch). Statuses semantics: fire on **all** terminal statuses by default
(`_cancel_run` appends the same `run_completed` bookend; failure-only alerting is the motivating
reactive case, so the filter is matching parametrization — the source's job, like cron's
`schedule` — not a behavior flag).

**Watch shape: `workflow_id`, never `run_id`.** A single-run watch is one-shot-shaped (dead row
after one fire), races run completion (the run may be terminal before the trigger commits), and
the need is already served by `await_run`/`GET /runs/{id}/wait`. The §1.2 coalescing concern only
exists for a many-completions watch — the workflow_id shape. No `workflow_version` on the source
spec: the §1.1 version-pin reservation binds kinds that RESOLVE a versioned resource for
execution (action kinds); a watch consumes events and executes nothing, and `wf_runs` stores no
launching workflow version to even match. No FK column for the watched id: the §1.1 FK exception
covers DELETABLE resources; workflows have no delete path.

### Tradeoffs, residual risk

- **Cron tick: zero regression.** `run_completion` rows carry `next_fire NULL`; the claim
  (`queries/__init__.py:8092`) and MIN (`:8364`) predicates exclude them BY PREDICATE. The
  `triggers_notify` INSERT arm fires once on row creation → one harmless MIN recompute. The
  runner-clear NOTIFY edge never fires for event fires (`running_since` stays NULL).
- **Completion-txn coupling (named accepted cost):** the matcher SELECT + inserts run inside every
  run-completion txn (one partial-index probe for accounts with zero watchers). If the insert ever
  raises persistently (e.g. a migration bug), the completion txn aborts and the run cannot reach
  terminal until fixed — a trigger-subsystem defect blocking workflow terminal transitions. Loud
  and fail-hard-consistent; the alternative (insert outside the txn) loses atomicity and was
  rejected.
- **Wake amplification posture change (document):** a `run_completion × wake_owner` trigger
  converts watched completions into owner-session model wakes at completion speed — the per-account
  enabled-trigger cap no longer bounds wake *rate* the way cron granularity did. Bounded by the
  per-account outstanding-run cap upstream; named, not mitigated.
- **Boundary-time matching:** a trigger created/enabled concurrently with a completion txn may or
  may not see that completion (READ COMMITTED snapshot order). Inherent to event-time semantics;
  no ordering promised.

---

## 2. `trigger_runs` — schema, scope, retention (Q2)

### Recommendation

One table, two writer modes. **Event fires:** the row is inserted `pending` in the completion txn
(the dispatch carrier — D3's "natural per-event claim/dedup carrier"), claimed `running` by the
runner, finalized terminal. **Timer fires (cron/one_shot):** the row is written ONCE at fire
completion with a terminal status — cron inside `record_trigger_fire`'s txn (echo cache and audit
can never disagree), one-shot standalone post-action. Timer rows are NEVER written at
claim/dispatch time: the tick tail is contractually frozen (D5 — task-id-only payload, per-trigger
queueing_lock), and a claim-txn row would orphan as `pending` on every silently-dropped
`AlreadyEnqueued` coalesce, which the sweep would then re-defer into a **spurious off-schedule cron
fire**. Scope: **every source's executed fires** (the audit is the only persistent record a
one-shot ever fired — its trigger row deletes pre-fire and takes `last_fire_*` with it).

The prompt's "deliberately NO status column" sketch was evaluated and **rejected**: the
claim-carrier role requires distinguishing `pending` (lost defer → sweep re-defers) from `running`
(crashed mid-fire → observable, NOT retried) from terminal — a `finished_at IS NULL` trinary
cannot express that without converting recovery to at-least-once (duplicate run launches). With a
status column, success == `status = 'ok'`; `timeout`/`skipped` keep their existing
`TriggerFireStatus` vocabulary; `error_summary` carries the detail (no separate error jsonb — the
status IS the kind).

### Concrete shape (migration 0085, same file as §6)

```sql
CREATE TABLE trigger_runs (
    id               text PRIMARY KEY,             -- 'trun_' + ULID  (ids.py: TRIGGER_RUN = "trun")
    trigger_id       text NOT NULL,                -- PLAIN id, deliberately NO FK (D3): one-shot rows
                                                   -- DELETE-BEFORE-FIRE and sessions CASCADE-delete
                                                   -- triggers — an FK would eat or block the audit.
    account_id       text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                                                   -- nothing else reaps trigger_runs (0061/0064/0073 precedent)
    owner_session_id text NOT NULL,                -- denormalized; slice 4 drops NOT NULL (catalog-only)
    trigger_name     text NOT NULL,                -- denormalized: the audit outlives the trigger row
    trigger_context  text NOT NULL
        CHECK (trigger_context IN ('cron', 'one_shot', 'run_completion')),
                                                   -- 'manual' stays vocabulary-only until fire-now wires
                                                   -- (contract stance: reserved values join no CHECK)
    event            jsonb,                        -- {run_id, workflow_id, status}; NULL for timer fires
    status           text NOT NULL
        CHECK (status IN ('pending', 'running', 'ok', 'error', 'timeout', 'skipped')),
    result_id        text,                         -- prefixed id of the created resource (wfr_… today;
                                                   -- sess_… for future spawn_session); NULL when the
                                                   -- fire produced no resource or failed. No FK
                                                   -- (audit-outlives-referent, same stance as trigger_id).
    error_summary    text,
    created_at       timestamptz NOT NULL DEFAULT now(),  -- fire-INTENT creation (match txn for events;
                                                          -- insert time for timer rows)
    started_at       timestamptz,                  -- runner claim; one value with the envelope's
                                                   -- fired_at and record_trigger_fire's stamp
    finished_at      timestamptz
);
-- the listing index (list_trigger_runs keys on the DENORMALIZED columns)
CREATE INDEX trigger_runs_by_owner_name
    ON trigger_runs (account_id, owner_session_id, trigger_name, created_at DESC);
-- the retention prune's time-range scan (the events-table BRIN precedent)
CREATE INDEX trigger_runs_created_brin ON trigger_runs USING BRIN (created_at);
-- sweep scan set: pending = lost defer (re-defer); running = crashed mid-fire (count + log only)
CREATE INDEX trigger_runs_unfinished ON trigger_runs (created_at)
    WHERE status IN ('pending', 'running');

-- the completion matcher's index:
CREATE INDEX triggers_run_completion_watch
    ON triggers (account_id, (source_spec ->> 'workflow_id'))
    WHERE source = 'run_completion';
```

Query layer: `claim_trigger_run(conn, id, *, started_at) -> dict | None` (`UPDATE … SET
status='running', started_at=$2 WHERE id=$1 AND status='pending' RETURNING event`; `None` =
duplicate job, caller exits); `finalize_trigger_run(conn, id, *, status, error_summary, result_id)`
(`… SET status=$2, …, finished_at=now() WHERE id=$1 AND status='running'` — first-writer-wins);
`record_trigger_run(...)` (timer path: single INSERT with terminal status, `started_at` = the
runner's fire timestamp, `finished_at = now()`); `list_trigger_runs(conn, trigger_id, *,
account_id, limit)` (account-scoped read); `list_pending_trigger_run_refs`,
`count_stuck_running_trigger_runs` (sweep); `prune_trigger_runs(conn, *, retention_days) -> int`.

**One fire timestamp rule:** the runner mints `started_at` once per fire; it is the claim's
`started_at`, the composed envelope's `fired_at` (§3), and `record_trigger_fire`'s stamp — audit
and delivered input agree by construction.

**Skip semantics:** TIMER skips keep slice-1 behavior (cron records `last_fire_status='skipped'`,
no audit row — the action never executed and no row pre-exists). EVENT skips MUST finalize their
pre-existing claim row (`status='skipped'`, `error_summary` = `"owner session archived" |
"trigger disabled" | "trigger deleted"`) — an unfinished claim row would be sweep-re-deferred until
retention pruned it 30 days later. One-shot timer skips currently leave zero record anywhere (row
deleted, no echo, no audit) — closing that with a tombstone row is sign-off item §11.7.

**Retention:** `Settings.trigger_runs_retention_days` (default 30), pruned from the periodic sweep
(own `pool.acquire`, returns deleted count for the log line). Time-based, never count-capped: a
count-cap could evict a *young* `run_completion` claim row inside the recovery horizon, and under
procrastinate dual execution of `_complete_run` the evicted claim is exactly what makes the
replayed hook's insert succeed → duplicate fire. 30 days ≫ the hours-scale recovery horizon makes
that structurally unreachable. Worst-case growth: `triggers_per_account_max(100) × 1440/day × 30d
≈ 4.3M rows/account` (~1 GB order) — bounded; the prune is index-ranged and usually empty.

`last_fire_at`/`last_fire_status`/`consecutive_failures` **stay on the trigger row** — the O(1)
echo cache (`_row_to_trigger_echo` builds straight off the row; deriving from `trigger_runs` would
add an aggregate join per session GET) and the auto-disable counter. Cache vs history: different
access patterns, both load-bearing, not redundancy. Under concurrent event fires the
`last_fire_*` pair is last-writer-wins (the counter is atomic per §1); accepted — the truthful
per-fire record is `trigger_runs`.

Read surface (slice-2 deliverable, no question owned it): `GET
/v1/sessions/{session_id}/triggers/{name}/runs → ListResponse[TriggerRunEcho]` (account-scoped,
`limit` param, `ORDER BY created_at DESC`) + `aios sessions triggers runs <session> <name>` CLI.
Agent tool surface: NOT exposed this slice (failure surfacing via the synthetic messages + echo
already serves the model; an audit-pagination tool is porcelain that can land additively).

```python
class TriggerRunEcho(BaseModel):
    id: str
    trigger_id: str
    trigger_context: str                       # 'cron' | 'one_shot' | 'run_completion'
    event: dict[str, Any] | None               # {run_id, workflow_id, status} for run_completion
    status: str                                # pending|running|ok|error|timeout|skipped (open str on read)
    result_id: str | None
    error_summary: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
```

Residual: a fire older than retention is unobservable again (accepted; per-deploy knob). A
one-shot fire that crashes after its action but before the audit INSERT leaves zero record — the
at-most-once spirit, unfixable without journaling-before-action, which D5 forbids for timer fires.
`fired_at`-family column comments must state the per-path meaning (`created_at` = intent,
`started_at` = execution) so operators sort/prune by the right one.

---

## 3. `input_template` → run input: deterministic envelope (Q3)

### Recommendation

**No substitution language.** The fired run's input is ALWAYS

```json
{"trigger": <composed firing context>, "input": <input_template verbatim>}
```

composed by one pure function in the runner — the single composition point for every source kind.
String-placeholder substitution (`{{run.output}}`) lost: it is a mini-language (escaping,
whole-value-vs-interpolation, type fidelity through string positions), and every one of those is a
fire-time render-failure mode on a source with **no next slot** — precisely the §1.2-forbidden
lossy path. Conditional reserved-key injection lost: it mutates the author's shape only when the
template is an object and clobbers/aliases a template that already carries `"trigger"`. Under the
envelope the composition is pure dict-building + `json.dumps` at insert — a hostile completing-run
output is nested data that cannot escape, and a template carrying its own `trigger` key nests
harmlessly under `"input"`.

**The completing run's output rides BY VALUE.** By-reference is dead on arrival: a workflow
script's tool surface is exactly `RUN_TOOLS = {web_search, web_fetch, http_request}`
(`workflows/run_tools.py:50`) and the host exposes only gate/agent/tool/parallel/pipeline/log — no
get-run capability exists, and reaching one via an `agent()` child needs a model wake, defeating
"deterministic, no model wake". Re-embedding adds no new unboundedness: no launch path caps run
input today (`WfRunCreate.input: Any`, `insert_wf_run` json.dumps unbounded), the chain
self-limits at the 64 MiB protocol frame guard, and output does not compound (run B's output is
whatever its script returns, not its input).

### Concrete shape

```python
def compose_workflow_run_input(
    *, trigger_id: str, trigger_name: str, source: str, fired_at: datetime,
    input_template: Any,
    completed_run: WfRun | None = None,            # run_completion only
    completed_error: dict[str, Any] | None = None, # run_completed payload's {'kind': …}, errored only
) -> dict[str, Any]:
    trigger: dict[str, Any] = {
        "id": trigger_id, "name": trigger_name,
        "source": source, "fired_at": fired_at.isoformat(),
    }
    if completed_run is not None:
        trigger["run"] = {
            "id": completed_run.id,
            "workflow_id": completed_run.workflow_id,
            "status": completed_run.status,        # completed | errored | cancelled
            "output": completed_run.output,        # row value: non-null only on completed
            "error": completed_error,              # {'kind': …} | None — mirrors WfRunWaitResponse
        }
    return {"trigger": trigger, "input": input_template}
```

The watched run is read **at fire time, account-scoped**: `get_wf_run(conn, event["run_id"],
account_id=trigger.account_id)` — NEVER the unscoped step-getter, so a mismatched/foreign run id
fails NotFound instead of embedding another tenant's output (the trigger row carries `account_id`
denormalized for exactly this). Safe because terminal statuses are monotonic and runs are never
deleted (archival is reserved terminal-only). On `errored`, `completed_error` comes from
`get_run_completed_event(conn, run_id).payload.get("error")` — byte-consistent with `await_run`.
Target workflows read it natively: `async def main(input): ctx = input["trigger"]; payload =
input["input"]`.

Composed-input examples: cron/one_shot/manual fires carry `trigger` without `run`; a
run_completion fire of an errored watch carries `run.status="errored"`, `run.output=null`,
`run.error={"kind": "script_error"}`.

**Size bound — WRITE MODELS ONLY (empirically proven necessary).** A serialized-byte bound on the
union member is NOT jsonb-round-trip stable: Postgres normalizes numbers through `numeric` —
`1e+308` is 6 chars at write and 309 digits back from jsonb, so a ~1.5 KB legally-written template
can read back >75 KB and a member-level validator would make the row unreadable. Blast radius:
`TRIGGER_ACTION_ADAPTER` runs inside `fetch_and_claim_due_triggers`' claim transaction
(`queries/__init__.py:8142`) — one poisoned row aborts the whole claim batch every tick and halts
EVERY trigger on the deployment (§2.2 is scheduler-load-bearing, not just API-load-bearing). The
bound therefore lives next to `_validate_cron_write_path` on `TriggerCreate` + `TriggerUpdate`
(automatically covering all three write paths incl. `SessionCreate.triggers`):

```python
MAX_INPUT_TEMPLATE_BYTES = 16_384   # sibling of MAX_COMMAND_CHARS / MAX_WAKE_CONTENT_CHARS

# WRITE-PATH ONLY — serialized-byte bounds are not jsonb-round-trip stable
# (numeric normalization can expand a written template ~50x); never on read
# models or the query-layer TypeAdapters.
n = len(json.dumps(action.input_template,
                   separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
```

`allow_nan=False` is load-bearing: it measures exactly what jsonb will accept, turning a
NaN/Infinity template into a 422 instead of a 500-at-INSERT.

**Deliberately NOT validated:** the template against the target workflow's `input_schema` —
`input_schema` is enforced on NO launch path today (operator HTTP, agent builtin, internal
`create_run` all pass input raw; verified), the workflow is mutable after write (false
confidence), and under the envelope the template isn't the run input anyway. No size cap on the
embedded `trigger.run.output` — it would fail fires for data every other launch path accepts
uncapped (the 64 MiB frame guard bounds the pathological end; a near-cap output errs the fired
run's first step loudly).

Residual: **envelope adoption burden** — a workflow authored for direct launch and later wired to
a trigger receives the envelope, not its bare input; misuse is a loud script KeyError, and the
tool-schema description must state the envelope explicitly (runtime vocabulary). The errored
watch's error MESSAGE (`error_repr`) is deliberately not carried (parity with `WfRunWaitResponse`
— only `{kind}`); a richer failure-handling field is a future additive key. Delivered type
fidelity is jsonb-grade, not byte-grade (same as every existing launch path; the composition adds
no new loss).

---

## 4. Auth + attenuation (Q4)

### Recommendation

**(a) Account-scoped watch.** Run-read authority is already account-scoped: `await_run` /
`get_run` / `list_runs` / `list_run_events` check only `account_id` (`tools/
workflow_management.py:179-190`, `services/workflows.py:230-317`); the sole launcher-scoped op is
the `cancel_run` MUTATION. An owner-launched-only watch would invent a stricter read class than
every existing read and structurally could never match operator-launched runs
(`launcher_session_id NULL`) — the deployment-reaction use case. Watching = reading; the matcher's
`t.account_id = $1` equality (run's own row value, never caller input) creates no new authority.
Enforcement points: the matching query's account equality (the boundary) + write-time
account-scoped existence of the watched `workflow_id` (correctness, not authority — the only
possible surface for a silently-dead watch; workflows are undeletable so it cannot go stale).

**(b) Owner authority at every fire.** The fire calls the UNMODIFIED `create_run` with
`launcher_session_id = trigger.owner_session_id`. Per fire, inside one txn (`workflows/
service.py:88-175`): environment re-resolved account-scoped (existence + tenancy — NOT liveness;
an archived env still resolves, matching session-create and operator launches); workflow
re-resolved and CURRENT script/surface snapshotted; the snapshot clamped (silent meet) to the
owner's CURRENT agent surface, read on the same conn (#835 consistency point); `action.vault_ids`
re-checked ⊆ the owner's CURRENT session vaults (loud ForbiddenError) + existence/tenancy in
`set_run_vaults`; depth cap on the threaded parent; per-launcher + per-account outstanding caps
under the advisory lock. Frozen at trigger-create: the references only (`workflow_id`, pin,
`environment_id` column, `vault_ids` list, `input_template`, `owner_session_id` anchor).
Authority is time-of-use — a trigger is not a frozen capability (an operator broadening the
owner's agent broadens future fires' clamp ceiling; documented). `launcher_session_id=None`
(operator-shaped fires) rejected outright: it would skip clamp, vault subset, AND the launcher cap
— privilege escalation from a session-ownable row. Archived-owner: the runner's source/action-
agnostic skip (`trigger_runner.py:73-89`) is the lifecycle gate; the claim→fire TOCTOU window is
the existing slice-1 width (the late-launched run is harmless — clamped to the pre-archive
surface). Slice-1 deputy precedent holds: operator-created sandbox_command triggers already
execute with the owner session's sandbox authority.

**(c) Vaults: pinned list in the action jsonb; fire-time is the single authority enforcement
point.** No write-time subset or existence check by default (sign-off §11.1): the subset check is
authority and goes stale in both directions (vault detached later = false confidence; granted
later = false denial); vaults ARE deletable, so the fire-time NotFound/Forbidden is reachable and
loud (→ counter → auto-disable + surfaced message — "can the model handle this failure itself?"
yes).

**(d) Loop bound: thread `parent_run_id`.** For `run_completion` fires:
`parent_run_id = the completing run's id` — same-account by construction (the matching query's
account equality discharges `create_run`'s trusted-same-account obligation; update the comment at
`service.py:107-113` to name this second trusted setter). For cron/one_shot fires:
`parent_run_id = the owner session's own parent_run_id` (byte-for-byte what the `create_run`
builtin threads, `workflow_management.py:174`) — `None` for normal sessions (root run); for a
workflow-child owner this closes the depth-laundering bypass (a past-`fire_at` one-shot is
`create_run` with a 0s delay). Result: `run_ancestor_depth` (recursive CTE, account-scoped per
hop, immutable chain) + `WORKFLOW_RUN_MAX_DEPTH=10` bounds EVERY trigger cycle
correct-by-construction — the depth-11 fire errors **before any run row exists**, so no
`run_completed` event ever re-arms the chain. This matters because the outstanding-run caps are
explicitly concurrency bounds, not rate bounds ("a sequential launch loop is unbounded by design",
`config.py:345-355`) — and a trigger loop, unlike an agent loop, burns no tokens and has no one in
it. Alternatives lost: per-trigger rate limiter (new mechanism, tunable, converts hard bound to
slow loop); exclude-self-launched-runs from matching (misses 2-trigger cycles, forbids intentional
bounded chaining); generation counter in context (duplicates lineage).

Cross-cutting fix surfaced by this analysis (broken-windows, same PR): **the HTTP trigger write
path never validates that `{session_id}` belongs to the caller's account** —
`services/triggers.add_trigger` inserts with the caller's `account_id` and the path's
`owner_session_id` unchecked (single-column FK, 0058). For the workflow action this fails closed
at fire (scoped `get_session_bare` → NotFound), but slice-1 `sandbox_command` provisions the raw
owner session's sandbox unscoped — cross-tenant code execution if multi-tenancy goes live. Slice 2
rewrites this path anyway: add an account-scoped session existence check in the shared write-path
helper (§7).

Residual: depth-10 caps legitimate linear trigger pipelines (>9 chained stages from one root
errors; shared ceiling with `agent()` recursion). A cycle re-armed by fresh operator roots wastes
~9 runs per root before the depth error and never auto-disables (the ok-fires reset the counter) —
termination per chain guaranteed, waste observable in `trigger_runs`. Horizontal amplification (N
watchers × cycles) is throttled in-flight by the synchronous account cap. Slice-4 additivity
obligations recorded: the matcher is a FOURTH `JOIN sessions … archived_at IS NULL` trigger query
(extend the contract's LEFT-JOIN list); the TriggerRow `session_parent_run_id` projection (as built: the timer lineage rides the row's own sessions JOIN) needs the owner-present
guard; owner-NULL fires degrade to `launcher_session_id=None` = operator authority via `create_run`'s
existing path — correct for operator-owned triggers, and the slice-4 CHECK swap is what stands
between a nullable-owner migration and accidentally operator-privileged fires.

---

## 5. Overlap / concurrency policy (Q5)

### Recommendation

**Overlap.** Every fire creates a run; the run subsystem's existing contractual bounds are the
policy: per-launcher 20 / per-account 100 outstanding, serialized COUNT+INSERT under the
per-account advisory lock (so "N^k breadth before caps" is structurally impossible — excess
launches raise `RateLimitedError` synchronously). A cap breach at fire is recorded as the fire's
**`error`** — never `skipped` — feeding the atomic counter and the 5-strike auto-disable with the
existing surfaced message (which already interpolates the actionable RateLimitedError text:
"…wait for runs you launched to finish (await_run) or cancel one…"). Cap-breach-as-skip rejected:
it would let a misconfigured 1-min-cron × 30-min-workflow trigger drop every fire forever,
silently, at steady state — the precise silent-drop class the principles ban — and would overload
`skipped`, which today means "deliberately not attempted per row lifecycle" and never advances the
counter.

**Skip-if-active** rejected on three independent grounds: (1) the "prior run from this trigger"
linkage needs either a racy join (two near-simultaneous fires both see none and both launch — a
contractual skip needs a per-trigger advisory lock: new serialization machinery) or a `trigger_id`
column on `wf_runs` (couples the run table to a deletable referent); (2) for event sources a
skipped fire never recurs — §1.2's silently-lossy coalescing; (3) as a knob on the workflow kind
it is a flag, which the growth rule forbids. **Queue** rejected: a durable pending-fire queue +
completion-watching dispatcher is a second scheduler; for cron it converts a too-fast cadence into
unbounded queue growth instead of a loud cap error.

`'ok'` means **the run was created** (launch semantics); the run's own outcome is its own audit
trail, reachable via `trigger_runs.result_id` — and observing run outcomes is exactly what a
`run_completion` watcher is for. A 1-min cron launching 10-min workflows steady-states at ~10
outstanding; launching 30-min workflows ramps to the cap at ~minute 20, errors fires 21-25, and
auto-disables at ~25 with the surfaced message — that ramp-then-disable IS the intended fail-hard
backpressure (recovery: cancel runs or fix cadence, re-enable).

Non-workflow actions under event fires need zero new machinery: two distinct cron triggers on one
session already exec concurrently today (`run_trigger` has no decorator lock; `registry.exec`
takes no per-session lock — the asyncio.Lock serializes provisioning only; each docker exec is an
independent subprocess), and `wake_owner` appends serialize at the session-row lock (gapless-seq
invariant).

Shared-budget residual (accepted, one principal one budget): trigger fires and the owner's own
`create_run` tool calls share the per-launcher cap of 20 — a chatty trigger can starve interactive
launches and vice versa; the error text names the remedy. Stuck (suspended-forever) runs pin cap
slots and can 5-strike a healthy trigger; recovery is `cancel_run`.

---

## 6. Additive CHECK extension + migration 0085 (Q6)

### Recommendation

Append one new CASE branch per predicate immediately before `ELSE false`; every existing branch
stays **byte-identical** (verified programmatically against the live 0083 constants, not by eye).
Plain `DROP CONSTRAINT` + `ADD CONSTRAINT` per the 0083 precedent — one read-only validation scan
under the migration's single transaction, zero row rewrites (`NOT VALID + VALIDATE` buys nothing:
alembic holds the lock to commit either way, and the table is small by construction). NO
validating SELECT — there is no backfill; old rows pass the superset CASE by construction, and
`ADD CONSTRAINT` itself hard-fails with the constraint name if the impossible happens.

```python
# migrations/versions/0085_triggers_slice2_workflow_action.py — module constants.
# The *_0083 constants are the verbatim previous predicates, embedded for
# downgrade() (synthetic-module loading forbids cross-migration imports; the
# 0083 _OLD_NOTIFY_FN pattern). No @dataclass anywhere.

SOURCE_SPEC_PREDICATE = """COALESCE((
    CASE source
        WHEN 'cron' THEN
            jsonb_typeof(source_spec -> 'schedule') = 'string'
            AND NOT (source_spec ? 'fire_at')
        WHEN 'one_shot' THEN
            jsonb_typeof(source_spec -> 'fire_at') = 'string'
            AND NOT (source_spec ? 'schedule')
        WHEN 'run_completion' THEN
            jsonb_typeof(source_spec -> 'workflow_id') = 'string'
            AND jsonb_typeof(source_spec -> 'statuses') = 'array'
            AND NOT (source_spec ? 'schedule')
            AND NOT (source_spec ? 'fire_at')
        ELSE false
    END
), false)"""

ACTION_PREDICATE = """COALESCE((
    CASE action ->> 'kind'
        WHEN 'sandbox_command' THEN
            jsonb_typeof(action -> 'command') = 'string'
            AND jsonb_typeof(action -> 'timeout_seconds') = 'number'
            AND jsonb_typeof(action -> 'max_output_bytes') = 'number'
            AND NOT (action ? 'content')
        WHEN 'wake_owner' THEN
            jsonb_typeof(action -> 'content') = 'string'
            AND NOT (action ? 'command')
        WHEN 'workflow' THEN
            jsonb_typeof(action -> 'workflow_id') = 'string'
            AND (action ? 'workflow_version')
            AND jsonb_typeof(action -> 'workflow_version') IN ('number', 'null')
            AND (action ? 'input_template')
            AND jsonb_typeof(action -> 'vault_ids') = 'array'
            AND NOT (action ? 'environment_id')
            AND NOT (action ? 'command')
            AND NOT (action ? 'content')
        ELSE false
    END
), false)"""

# Contract §1.1's per-kind presence/absence conjunct for the FK column, relocated
# into a SIBLING constraint (same migration, zero rewrites, both directions) so the
# action predicate's existing branches stay byte-identical. Boolean equality:
# workflow rows MUST carry environment_id; every other kind MUST NOT. (A missing
# 'kind' makes this NULL → CHECK-satisfied; triggers_action_shape's ELSE false
# already rejects that row.) Slice-4 spawn_session swaps this to
# ((action ->> 'kind') IN ('workflow','spawn_session')) = (environment_id IS NOT NULL).
ENVIRONMENT_ID_IFF_WORKFLOW_PREDICATE = (
    "(action ->> 'kind' = 'workflow') = (environment_id IS NOT NULL)"
)
```

Branch-content notes: `workflow_version` is the §1.1 required-but-nullable idiom verbatim
(false-dominance makes the `?` guard order-independent: SQL `AND` needs no short-circuit —
`false AND NULL = false`). `input_template` is presence-only (`?`) because every JSON type incl.
json `null` is legal (`WfRun.input: Any` — typing it `object` would make the trigger path narrower
than the launch path it reuses) and `jsonb_typeof` never returns SQL NULL for a present key.
`NOT (action ? 'environment_id')` is hygiene (single storage location — the column). `vault_ids`
must be an array (materialized `[]`). Existing branches deliberately do NOT gain
`NOT (action ? 'workflow_id')` etc. — byte-identical mandate + contractually open key set;
`extra="forbid"` write models make stray keys unreachable (deliberate asymmetry, noted in the 0085
docstring). `statuses` joins the run_completion CHECK because it ships in the kind's FIRST shape
(the retro-add rule bars only post-first-ship fields).

Every hostile probe executed on PG16: workflow+NULL env rejected (iff), sandbox_command+env-set
rejected (iff, reverse direction), run_completion+stray fire_at rejected, absent/string
`workflow_version` rejected, absent `input_template` rejected, absent `workflow_id` rejected via
the COALESCE NULL-collapse, object `vault_ids` rejected, json-null watch `workflow_id` rejected,
kind-less action rejected by `ELSE false`; legal float-pin/int-pin rows insert; slice-1 rows
untouched.

```python
def upgrade() -> None:
    # Nullable FK column (0067 wf_runs precedent: bare REFERENCES, no ON DELETE, no
    # index — environments are archive-only today; the RESTRICT is latent unless a
    # hard-delete path ever ships). ADD COLUMN, no default: catalog-only, zero rewrites.
    op.execute("ALTER TABLE triggers ADD COLUMN environment_id text REFERENCES environments(id)")

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute(f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape CHECK ({SOURCE_SPEC_PREDICATE})")
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape CHECK ({ACTION_PREDICATE})")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_environment_id_iff_workflow "
        f"CHECK ({ENVIRONMENT_ID_IFF_WORKFLOW_PREDICATE})"
    )
    # + trigger_runs CREATE TABLE + indexes (§2) + triggers_run_completion_watch (§2)

def downgrade() -> None:
    # Fail hard on any slice-2 row — unrepresentable under the 0083 predicates and
    # not reconstructible (0083's wake_owner stance). Under the iff constraint,
    # environment_id NOT NULL ⇒ kind='workflow', so this one count subsumes the column.
    n = op.get_bind().execute(sa.text(
        "SELECT count(*) FROM triggers "
        "WHERE source = 'run_completion' OR action ->> 'kind' = 'workflow'"
    )).scalar()
    if n:
        raise RuntimeError(f"cannot downgrade: {n} run_completion/workflow trigger rows")
    # Explicit for symmetry with upgrade(); Postgres would auto-drop this table
    # constraint with the column (table constraints involving a dropped column are
    # dropped automatically — empirically verified; CASCADE is only for outside deps).
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_environment_id_iff_workflow")
    op.execute("ALTER TABLE triggers DROP COLUMN environment_id")
    # restore the embedded 0083 predicates; DROP TABLE trigger_runs; drop the watch index
```

**NOTIFY function: NO rewrite.** The 0083 `notify_scheduled_tasks_due()` UPDATE gate compares
`source`/`source_spec`/`enabled`/the running_since-clear edge value-generically; new kind values
flow through unchanged. A `run_completion` INSERT causes one harmless MIN recompute;
`environment_id`-only updates don't notify (action-material read at fire time, like the
deliberately-ungated action edits).

Residual (sign-off §11.2): the CHECKs deliberately do not prevent a `run_completion` row from
carrying a non-NULL `next_fire` — only the service layer keeps it NULL. The failure mode of a bug
there is a **hot re-claim runaway** (the claim's non-cron arm never advances `next_fire`, the
runner-clear edge re-NOTIFYs, ok-fires never trip auto-disable — a tick-speed loop), not one
mis-fire. The optional guard `CHECK (source <> 'run_completion' OR next_fire IS NULL)` is
belt-and-suspenders requiring explicit sign-off, with that severity framing.

---

## 7. `workflow` action shape + version pin (Q7)

### Recommendation

```python
class WorkflowAction(BaseModel):
    """Launch a run of ``workflow_id`` at fire time — deterministic, no model wake.

    The run's input is ALWAYS the envelope ``{"trigger": <firing context>,
    "input": <input_template verbatim>}`` — no placeholder substitution.
    ``environment_id`` is deliberately NOT a field: the run binds to the owner
    session's environment, resolved at write time into the first-class
    ``triggers.environment_id`` column. ``workflow_version``: null = run the
    workflow's CURRENT version at each fire (float); an integer is a drift
    assertion — it must equal the workflow's current version at write, and a
    fire whose workflow has since been edited records an error instead of
    running the unreviewed script (workflows have no version-history table; a
    pin cannot resolve an old script, only refuse a new one)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["workflow"] = "workflow"
    workflow_id: str = Field(min_length=1)
    workflow_version: int | None = Field(default=None, ge=1)   # int bounds round-trip jsonb exactly
    input_template: Any = None        # structure-only here; byte bound on the WRITE models (§3)
    vault_ids: list[str] = Field(default_factory=list)

class WorkflowActionReplace(WorkflowAction):
    """Update-side variant (§2.2): optional-at-create fields REQUIRED, so a partial
    action 422s instead of silently flipping a pin to float, nulling the template,
    or dropping vault bindings."""
    workflow_version: int | None = Field(ge=1)   # required; explicit null = explicit float
    input_template: Any                           # required; explicit null = explicit no-payload
    vault_ids: list[str]                          # required; explicit [] = explicit none

TriggerAction = Annotated[
    SandboxCommandAction | WakeOwnerAction | WorkflowAction, Field(discriminator="kind")
]
TriggerActionReplace = Annotated[
    SandboxCommandActionReplace | WakeOwnerAction | WorkflowActionReplace, Field(discriminator="kind")
]
```

**No `environment_id` on the wire — resolved from the owner session at write.** This deviates from
the early lean (in-action wire field) because the code forbids it: `trigger_create_handler`
validates the SAME `TriggerCreate` model the operator API uses (`tools/trigger_create.py:184`), so
any field on the union member is agent-reachable — and the agent-facing `create_run` builtin
deliberately refuses caller-chosen environments ("a caller-chosen env id would be a cross-tenant
attack surface"; F2, `workflow_management.py:71-79`). An in-action `environment_id` would be a
trigger-shaped F2 bypass (one-shot a workflow into a permissive same-account env). Nothing is lost:
sessions' `environment_id` is immutable (verified: no `UPDATE sessions` statement touches it), so
write-time freeze == fire-time resolution. The future operator-owned slice adds the wire field
additively (it maps to a column, exempt from the jsonb CHECK retro-add rule). Consequently the
read path needs NO reinjection — the echo's workflow member is the stored jsonb verbatim;
`_row_to_trigger_echo` is unchanged.

**Pin = fail-on-drift (the only implementable semantic).** Provably no version-resolution path
exists (versions bump in place; "There is no version-snapshot table"). Enforced twice at the same
consistency points: at WRITE, `workflow_version` must equal the workflow's current version
("resolve-latest-at-write" — the write resolves what the pin freezes; mismatch → 409
`ConflictError`, mirroring `update_workflow`'s optimistic token); at FIRE, via a new **additive
`expected_version: int | None = None` kwarg on `create_run`**, checked right after its in-txn
`get_workflow` — the same consistency point as the script snapshot, so no check-to-snapshot TOCTOU
(a runner-side pre-check would race). Existing callers untouched. **Default = float (null).**
Anthropic's probed deployments default to pin-and-freeze, but they have version snapshots — a
pinned deployment keeps *running* the old version. aios pin can only refuse, so auto-pin-current
would turn every `update_workflow` into a 5-fire auto-disable of its triggers. Float's exposure is
bounded by the per-fire protections that already exist (launch-time script snapshot, surface clamp
to the owner's current agent, vault subset) — what floats is script content under already-clamped
authority, the same exposure as the owner calling `create_run` directly. If a `workflow_versions`
table ever lands, upgrading int-pin from refuse-on-drift to run-old-version is a behavior change
for existing pinned rows — call it out then.

**Shared write-path validation helper** (`services/triggers.py`) — called from ALL THREE write
paths: `add_trigger`, `update_trigger`, AND `create_session`'s trigger-attach loop
(`services/sessions.py:262-275` calls `queries.add_trigger` directly today and would otherwise
skip validation or 500 on the CHECK):

```python
async def validate_trigger_spec(conn, source, action, *, session_id, account_id) -> str | None:
    """Returns the resolved environment_id (workflow action) or None. Shared by the
    POST/PUT trigger paths and SessionCreate.triggers attachment."""
    # Seam A: FIRST and UNCONDITIONAL — the cross-tenant-attach fix (§4) must
    # cover ALL action kinds (for slice-1 sandbox_command a foreign session is
    # cross-tenant code execution at fire time). 404s like every scoped read.
    session = await queries.get_session_bare(conn, session_id, account_id=account_id)
    if isinstance(source, RunCompletionSource):
        # silent-dead-watch guard (§1): account-scoped existence; cannot go stale (no delete path)
        await wf_queries.get_workflow(conn, source.workflow_id, account_id=account_id)
    if not isinstance(action, WorkflowAction):
        return None
    workflow = await wf_queries.get_workflow(conn, action.workflow_id, account_id=account_id)
    if action.workflow_version is not None and action.workflow_version != workflow.version:
        raise ConflictError(
            f"workflow_version pin {action.workflow_version} does not match current "
            f"version {workflow.version}; re-read the workflow and pin the version you reviewed",
            detail={"pinned": action.workflow_version, "current": workflow.version},
        )
    return session.environment_id   # immutable ⇒ freeze == fire-time resolve
    # NO vault existence/subset check (fire-time create_run is the single authority
    # enforcement point — resolution #1).
```

`queries.add_trigger` gains `environment_id` as a **required** kwarg (no default — mypy forces
every present and future call site through the decision); `queries.update_trigger` gains
`environment_id: str | None | EllipsisType = ...` (the `next_fire` sentinel precedent) and the
service recomputes it **whenever `action` is provided** (workflow → resolved env; other kinds →
`None`) so the jsonb and the column flip in ONE UPDATE — the iff CHECK catches forgotten kind
conversions loudly, but a same-kind workflow→workflow update that changed nothing visible would
silently keep a stale column without this pairing (the one silent-corruption path in the design;
pinned as an obligation + test). `TriggerRow` gains `environment_id: str | None`;
`unscoped_get_trigger_row` / `fetch_and_claim_due_triggers` project it. **`clone_session`'s
trigger INSERT adds the column** (`s.environment_id` in both the column list and the SELECT) —
without it, cloning a session owning a workflow-action trigger aborts the entire clone on the iff
CHECK, byte-for-byte the §6.1 clone-crash class slice 1 just fixed; verbatim copy is correct
(the clone's session row copies the parent's `environment_id`, so the trigger-env == owner-env
invariant holds). The blanket `ForeignKeyViolationError → "session not found"` mapping in
`add_trigger` gets a comment noting the second FK (unreachable today — env pre-validated in-txn,
no env delete path — but a future env-delete must not inherit a lying error).

**The merged fire branch** (all four research sketches unified):

```python
async def _run_workflow(
    trigger: queries.TriggerRow, action: WorkflowAction,
    *, event: dict[str, Any] | None, started_at: datetime,
) -> tuple[TriggerFireStatus, str | None, str | None]:
    """Returns (status, error_summary, result_id). 'ok' = the run was CREATED;
    statuses ok/error (timeout N/A — the launch is one DB transaction; the run
    executes asynchronously and its outcome is not this fire's outcome)."""
    pool = runtime.require_pool()
    assert trigger.environment_id is not None   # backed by the iff CHECK
    try:
        completed_run = completed_error = None
        parent_run_id: str | None
        if event is not None:                   # run_completion fire
            async with pool.acquire() as conn:
                # account-scoped — NEVER the unscoped step-getter (§3)
                completed_run = await wf_queries.get_wf_run(
                    conn, event["run_id"], account_id=trigger.account_id)
                if completed_run.status == "errored":
                    ev = await wf_queries.get_run_completed_event(conn, completed_run.id)
                    completed_error = ev.payload.get("error") if ev else None
            parent_run_id = completed_run.id    # depth cap bounds reactive cascades (§4d)
        else:                                   # cron / one_shot fire
            parent_run_id = trigger.session_parent_run_id  # owner lineage (§4d),
            # projected onto TriggerRow off its sessions JOIN (immutable)
        composed = compose_workflow_run_input(
            trigger_id=trigger.id, trigger_name=trigger.name, source=trigger.source,
            fired_at=started_at, input_template=action.input_template,
            completed_run=completed_run, completed_error=completed_error)
        run = await wf_service.create_run(
            pool, account_id=trigger.account_id,
            workflow_id=action.workflow_id,
            environment_id=trigger.environment_id,
            input=composed,
            vault_ids=action.vault_ids,
            launcher_session_id=trigger.owner_session_id,   # ALL owner authority flows from this
            parent_run_id=parent_run_id,
            expected_version=action.workflow_version,        # None = float; int = drift assertion
        )
        return "ok", None, run.id
    except Exception as e:
        # ConflictError (pin drift) / ForbiddenError (vault breach) / RateLimitedError
        # (outstanding caps) / WorkflowRunDepthExceededError / NotFoundError (a vault in
        # action.vault_ids was deleted, or the owner session deleted mid-fire — envs and
        # workflows have no delete path) — all loud, all feed the counter + auto-disable.
        log.exception("trigger.workflow_error", trigger_id=trigger.id, name=trigger.name)
        return "error", f"run launch failed: {type(e).__name__}: {e!s:.200}", None
```

One-shot failure surfacing widens to the workflow action with a NEW action-aware marker —
`[Trigger '<name>' failed to launch its workflow run: <detail>]` — not the byte-frozen
`[Scheduled wake '<name>' failed to deliver: …]` string (which binds backfilled rows only; new
kind, new runtime-truthful string). All three `_run_*` helpers normalize to the
`(status, error_summary, result_id)` shape (`result_id=None` for sandbox/wake).

**Tool surface:** the `trigger_create`/`trigger_update` `oneOf` gains the workflow branch
(`required: [kind, workflow_id]` on create; `workflow_version`/`input_template`/`vault_ids` ALL
required on update). Descriptions must state, in runtime vocabulary: the envelope shape (§3), pin
= drift-assert ("an edited workflow makes pinned fires error — re-pin after review"), "the run
launches into your session's environment", and 'ok' = launched-not-succeeded. No `environment_id`
anywhere on the tool surface. openapi + SDK regen before push (`./scripts/regen-client.sh`); the
discriminated `oneOf` + `discriminator.mapping` already round-trips the generator (slice-1
precedent — third member is the same path).

---

## 8. Cross-cutting obligations (no question owned these; all are slice-2 deliverables)

1. **trigger_runs read surface** — route + CLI per §2; without it "observable" means hand-SQL.
2. **Unified write-path validation** (§7 helper) across POST/PUT/SessionCreate-attach — including
   the run_completion watched-workflow check (a typo'd watch in a session-create body must 404,
   not sit silently dead), and the cross-tenant-attach fix (§4).
3. **Deploy gates (eumemic-ops):** `aios migrate` then **lockstep api+worker promote BEFORE any
   client writes slice-2 rows**. Skew failure modes: an old worker CLAIMING a new-kind row aborts
   the whole claim batch inside `fetch_and_claim_due_triggers` → scheduler hot-loop → ALL triggers
   stall (the §3 claim-batch-poisoning blast radius as a deploy hazard); an old API 500s session
   GETs on new-kind rows; old workers never dispatch run_completion fires (silent missed events
   during skew). Rollback below slice 2 is forbidden once rows exist. Add
   `AIOS_TRIGGER_RUNS_RETENTION_DAYS` to the ops grep list.
4. **Vocabulary fix (same PR):** the cap RateLimitedError strings say "active-timer cap"
   (`services/triggers.py`, `services/sessions.py`) — the caps now also gate event watches; say
   "active-trigger cap". (The caps themselves apply to new kinds automatically —
   `count_account_triggers` is source-agnostic.)
5. **Event context for non-workflow actions (decided):** `run_completion × wake_owner` delivers
   `content` VERBATIM (no event interpolation) and `× sandbox_command` gets no event — the
   workflow action is the event-consuming action; a wake is a "something completed, go look" ping
   and the model can list runs. Documented blindness, not a gap; an event-aware wake is a future
   new kind if ever wanted.
6. **Doc/comment obligations:** the `parent_run_id` second-trusted-setter comment
   (`workflows/service.py:107-113`) + `list_wf_runs`/`WfRun.parent_run_id` docstrings (children
   now include trigger-reaction runs); `hard_delete_account`'s stale "FKs all use RESTRICT"
   docstring (false since 0061/0064/0073; trigger_runs adds another CASCADE table) — broken
   window, fix in the same PR; the 0085 docstring anchors the iff constraint to contract §1.1
   (without it the relocation reads as a dropped obligation); §1.1's version-pin reservation is
   hereby read as binding **action kinds** (executed references), not source watches; slice-4
   obligations extended per §4.

---

## 9. What does NOT change in slice 1

- **`harness/scheduler.py`: zero-line diff.** The claim and MIN queries are untouched —
  `run_completion` rows are excluded BY the existing `next_fire IS NOT NULL` predicates, exactly
  as §3 of the slice-1 contract reserved.
- **The tick defer tail is frozen** (D5): per-trigger `queueing_lock=f"trigger:{id}"`,
  task-id-only payload, `running_since` claim — cron/one_shot fires keep it verbatim. The
  `trigger_run_id` kwarg and carrier row are event-fire-only.
- **The NOTIFY function and channel** (`notify_scheduled_tasks_due` / `aios_scheduled_tasks_due`):
  byte-untouched; value-generic gates absorb the new kinds.
- **Existing CHECK branch texts** (`cron`/`one_shot`/`sandbox_command`/`wake_owner`):
  byte-identical inside the new predicate constants (machine-verified). Zero row rewrites
  anywhere; `ADD COLUMN` is catalog-only.
- **One-shot delete-before-fire, cron record-after, auto-disable at 5, skip arms, failure-marker
  strings for existing kinds:** semantics unchanged. (`consecutive_failures` moves to a SQL CASE —
  behavior-identical for single-flight cron, coherent under event-fire concurrency.)
- **`last_fire_*` echo columns and `TriggerEcho` shape:** kept; unions gain members, no other
  change; no read-path reinjection (the workflow member carries no `environment_id`).
- **Per-session (32) and per-account trigger caps:** apply to the new kinds unchanged.
- **`sandbox_command`/`wake_owner` execution, `wake_session`/`wake_self`/`schedule_wake` tools,
  the workflow subsystem's run lifecycle/journal/locks:** untouched (the completion hook adds one
  matcher SELECT + bounded inserts inside the existing terminal txn; `create_run` gains one
  default-`None` kwarg).

---

## 10. Test obligations

(a) **THE coalescing regression** (the central §1.2 obligation): two completions of one watched
workflow → exactly two `trigger_runs` rows + two launched runs, never one. (b) Concurrent-fire
counter atomicity: two parallel error fires → `consecutive_failures` exactly 2; disable fires at
exactly 5, surfaced once. (c) CHECK must-reject e2e probes for both new kinds (the §6 hostile-row
list, extending `tests/e2e/test_triggers.py:841-885`). (d) jsonb round-trip read-acceptance (the
§2.2 twin of the rare-cron test): a maximal `1e+308`-float template writes, round-trips, the
action adapter still validates the row, and `fetch_and_claim_due_triggers` still claims the batch.
(e) Clone regression extension: clone a session owning a workflow-action trigger (env column
copies; clone succeeds) and a run_completion trigger. (f) Event-skip finalization:
archived/disabled/deleted-trigger event fires finalize the claim row `skipped` and the sweep
terminates. (g) Duplicate-job no-op via the `pending→running` claim. (h) `update_trigger` kind
conversions flip the `environment_id` column both directions in one UPDATE; same-kind
workflow→workflow updates re-resolve it. (i) 0085 downgrade refusal probe. (j) Session-create
trigger attachment validates identically to POST /triggers (bad watch 404s; pin drift 409s).
(k) Mid-flight source replacement: an in-flight event fire of a trigger converted to one_shot does
NOT delete the row (origin-derived lifecycle arm). (l) Pin drift: edit the workflow, pinned
trigger's next fire records `error` with the ConflictError summary.

---

## 11. Sign-offs — RESOLVED (the build directive's final decisions)

| # | Question | Resolution | As built |
|---|---|---|---|
| 1 | Write-time vault-subset check | **OUT** | fire-time `create_run` is the single authority point; the §7 helper does no vault reads |
| 2 | `next_fire IS NULL` guard CHECK | **IN** | `triggers_run_completion_no_next_fire` in migration 0085 — DB-enforces the §3 invariant; the bug it guards is a tick-speed hot re-claim runaway |
| 3 | `trigger_runs.event` shape CHECK | **OUT** | the writer is unit/e2e-tested instead (the coalescing regression asserts the exact event keys) |
| 4 | Lazy finalizer for stuck `running` rows | **OUT** | the sweep ships the warning log only (`trigger.fires_stuck_running`); deliberately never retried |
| 5 | Retention | **30 days**, time-based | `Settings.trigger_runs_retention_days` / `AIOS_TRIGGER_RUNS_RETENTION_DAYS`; never count-capped |
| 6 | Self-fire | **ALLOW**, depth-capped | parent_run_id threading + the depth-cycle termination test (10 runs, fire #10 errors, chain dead) |
| 7 | One-shot timer-skip | **tombstone** | `_skip_claimed_fire`'s one-shot arm writes a `skipped` audit row — the zero-record hole is closed |
| 8 | `trun` prefix + runs route | **as written** | `GET /v1/sessions/{id}/triggers/{name}/runs`, `operation_id=list_trigger_runs` |
