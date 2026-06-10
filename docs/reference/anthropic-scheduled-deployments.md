# Anthropic Managed Agents — Scheduled Deployments (captured wire semantics)

> **Purpose.** Empirically-captured semantics of Anthropic's *Scheduled Deployments* feature
> (shipped 2026-06-10), to inform the aios **`triggers`** primitive — the unified scheduled-task/cron
> surface (issues `#818` slice 1, `#819` slice 2), which subsumes the older `#467` cron-umbrella +
> `#636` `scheduled_tasks` framing into one entity with `source` + `action` unions. Upstream-reference
> material, **not** aios behaviour.
>
> **Method.** (1) Raw HTTP (`curl`+`jq`) against `https://api.anthropic.com`, headers `x-api-key`,
> `anthropic-version: 2023-06-01`, `anthropic-beta: managed-agents-2026-04-01`, probed **2026-06-10**
> end-to-end then cleaned up. (2) Cross-verified against the official docs page + all 5 Stainless
> SDKs (Python/TS/Go/Java/C#) via a read-only verification pass.
>
> **Confidence legend:** ✅ wire-probed · 🧰 SDK-canonical (5 SDKs agree, byte-identical) ·
> 📄 docs-only · ⚠️ resolved a docs contradiction · ❓ open (follow-up probe in §14).

---

## 0. Mental model

A **deployment** is a control-plane sibling of a session — *not* a wrapper around one. It freezes
**who** runs (`agent`, version-pinned), **where** (`environment_id` + `resources` + `vault_ids`),
**what to say** (`initial_events`), and optionally **when** (`schedule`). Each fire (cron or manual)
re-instantiates a *fresh* session and replays `initial_events` verbatim as its kickoff.

A **deployment run** (`drun_`) is the audit row for each trigger *attempt*. It exists even when no
session is created (archived env, rate limit) — decoupling *trigger-success* from *session-success*.

```
deployment (depl_)        ── /run (manual)  ──▶  deployment_run (drun_)  ──▶  session (sesn_)
  agent (pinned vN)       ── cron fire      ──▶    trigger_context            title  = deployment.name
  environment_id          (always 200 +run)        session_id | error        events[0] = initial_events[*]
  initial_events  ───────────────────────────────────────────────────────────────▶ kickoff (verbatim)
  schedule? (nullable)
```

| Resource | Path | ID prefix | Pagination |
|---|---|---|---|
| Deployment | `/v1/deployments` | `depl_` | `PageCursor`: `{data, next_page}`, `?page=<tok>&limit=N` (limit default 20, max 1000) |
| Deployment run | `/v1/deployment_runs` | `drun_` | `PageCursor` (bare `{data}` at small N) |

The `?beta=true` query param the docs sprinkle on every example is **redundant** ✅ — the
`anthropic-beta` header alone gates. (The SDKs send both unconditionally, but don't rely on it.)

---

## 1. Endpoints

| Method | Path | Op | SDK method | Notes |
|---|---|---|---|---|
| `POST` | `/v1/deployments` | Create | `deployments.create` | ✅ requires `name`,`agent`,`environment_id`,`initial_events`; `schedule` **optional** |
| `GET` | `/v1/deployments` | List | `deployments.list` | ✅ cursor; **excludes archived by default** (`include_archived` opt-in) |
| `GET` | `/v1/deployments/{id}` | Get | `deployments.retrieve` | ✅ byte-identical to create response |
| `POST` | `/v1/deployments/{id}` | Update | `deployments.update` | ✅ partial; **no** version lock; strict (unknown field → 400) |
| `POST` | `/v1/deployments/{id}/pause` | Pause | `deployments.pause` | ✅ |
| `POST` | `/v1/deployments/{id}/unpause` | Unpause | `deployments.unpause` | ✅ |
| `POST` | `/v1/deployments/{id}/archive` | Archive | `deployments.archive` | 📄 terminal (observed via cascade, §11) |
| `POST` | `/v1/deployments/{id}/run` | Manual run | `deployments.run` | ✅ always 200 + `deployment_run` |
| `GET` | `/v1/deployment_runs` | List runs | `deployment_runs.list` | ✅ `?deployment_id=&has_error=&trigger_type=&created_at[…]=` |
| `GET` | `/v1/deployment_runs/{id}` | Get run | `deployment_runs.retrieve` | 🧰 single `drun_` retrieve (not wire-hit) |

SDK namespace: `client.beta.deployments` / `client.beta.deployment_runs` (TS: `deploymentRuns`).
`deployment_runs` has only `retrieve`/`list` — runs are produced by `deployments.run` or the scheduler, never created directly.

---

## 2. Create — request & full response ✅

Request (minimum viable + scheduled):

```json
{
  "name": "aios-probe-deploy",
  "agent": "agent_01KC8coCbPzCWy2H2EMqqL3A",
  "environment_id": "env_01A98FSqXLSt4EUMMBA519yk",
  "initial_events": [
    {"type": "user.message", "content": [{"type": "text", "text": "Reply with exactly: ok"}]}
  ],
  "schedule": {"type": "cron", "expression": "0 20 * * 5", "timezone": "America/New_York"}
}
```

**Full captured response** (`GET /v1/deployments/{id}` returns the same shape byte-for-byte):

```json
{
  "type": "deployment",
  "id": "depl_01Mnw655idRwxKWHGWfN9Xmr",
  "name": "aios-probe-deploy",
  "description": null,
  "agent": {"type": "agent", "id": "agent_01KC8coCbPzCWy2H2EMqqL3A", "version": 1},
  "environment_id": "env_01A98FSqXLSt4EUMMBA519yk",
  "initial_events": [
    {"type": "user.message", "content": [{"type": "text", "text": "Reply with exactly: ok"}]}
  ],
  "resources": [],
  "vault_ids": [],
  "metadata": {},
  "schedule": {
    "type": "cron",
    "expression": "0 20 * * 5",
    "timezone": "America/New_York",
    "last_run_at": null,
    "upcoming_runs_at": [
      "2026-06-13T00:00:00Z", "2026-06-20T00:00:00Z", "2026-06-27T00:00:00Z",
      "2026-07-04T00:00:00Z", "2026-07-11T00:00:00Z"
    ]
  },
  "status": "active",
  "paused_reason": null,
  "archived_at": null,
  "created_at": "2026-06-10T17:24:16.380044Z",
  "updated_at": "2026-06-10T17:24:16.380044Z"
}
```

`agent` was sent as a **bare string** and came back **resolved + version-pinned** to
`{type:"agent", id, version:1}` (see §7). `0 20 * * 5` (Fri 20:00 EDT) → `Sat 00:00:00Z` confirms
`upcoming_runs_at` is local-wall-clock computed, returned in **UTC**, **5** entries.

---

## 3. Deployment object — field reference

| Field | Type | Req | Notes |
|---|---|---|---|
| `type` | `"deployment"` | — | constant |
| `id` | `depl_…` | — | |
| `name` | string | ✅ | becomes the spawned **session title** ✅ |
| `description` | string\|null | — | |
| `agent` | bare-id \| `{type:"agent",id,version}` | ✅ | bare string → resolved & **version-pinned** ✅ |
| `environment_id` | `env_…` | ✅ | validated; mutable on update; archived/unknown → 404 ✅ |
| `initial_events` | array | ✅ | ≥1; replayed verbatim as kickoff ✅. **Union of 3 types** 🧰 (below) |
| `resources` | array | — | **4-member union** 🧰 (below); only `memory_store` wire-exercised |
| `vault_ids` | `[vlt_…]` | — | **validated at create** (bogus → 404) ✅ |
| `metadata` | object | — | round-trips ✅ |
| `schedule` | object\|**null** | — | omit → schedule-less, manual-run-only deployment ✅ |
| `status` | `"active"`\|`"paused"` | — | **schedule state only** — never `"archived"` ✅ |
| `paused_reason` | union (below) | — | |
| `archived_at` | ts\|null | — | **the** archival signal (orthogonal to `status`) ✅ |
| `created_at`/`updated_at` | ts | — | |

**`initial_events` union** 🧰 (typed surface — the live beta accepted only `user.message` at probe time ✅; `user.interrupt` → 400):
- `user.message` `{content:[…]}` — ✅ accepted, replayed as `events[0]`
- `system.message` `{content:[…]}` — 🧰 typed, not wire-confirmed
- `user.define_outcome` `{description, rubric:(file|text), max_iterations}` — 🧰 typed, not wire-confirmed (deployment-as-graded-outcome)

**`resources` union** 🧰 (mirrors session resources; only `memory_store` wire-exercised ✅):
- `file` `{file_id, mount_path?}`
- `github_repository` `{url, checkout?:(branch|commit), mount_path?}` — the most common real use (Claude's `/schedule` skill is built on it)
- `memory_store` `{memory_store_id, access?:(read_write|read_only), instructions?}`
- `session` resource config

**`paused_reason` union** 🧰 — **three** variants (capture saw only the first two ✅):
- `null` (active)
- `{type:"manual"}` — operator pause ✅
- `{type:"error", error:{type:…,message:…}}` — **system auto-pause** on a structural resource error (14-member subset of the run-error enum, §10). Distinct from the agent-archive auto-*archive* cascade.

---

## 4. Schedule semantics ✅

### Cron expression
- **Strictly 5-field POSIX**, numeric only: `minute hour day-of-month month day-of-week`. DoW `0–7`, both `0` and `7` = Sunday 🧰.
  - 6-field (seconds) → `400 … expected 5 fields, got 6`.
  - `@weekly`/`@daily` macros → `400 … predefined shortcuts (@daily, @hourly, …) are not supported`.
  - Named day `FRI` → `400 … invalid value "FRI" in field day-of-week` (**numeric only**).
  - Out-of-range (`99 …`) → `400 … field minute value 99 is out of range`.
  - Step syntax (`*/1`) ✅ works. Minute is max granularity.
- `schedule.type` must be `"cron"` — `"interval"` → `400 … "interval" is not a valid value`.

### Timezone
- IANA identifier; bad value (`Mars/Phobos`) → `400 … is not a valid IANA timezone`.
- `upcoming_runs_at` returns **5** future fire times, **UTC**, computed from local wall-clock.

### DST — literal wall-clock, with a create-time guard ⚠️
- **Fall-back (clocks repeat):** `30 1 1 11 * / America/New_York` →
  `["2026-11-01T05:30:00Z","2026-11-01T06:30:00Z",…]` — **fires twice** (01:30 EDT *and* 01:30 EST). ✅
- **Spring-forward (clock gap):** an annual `30 2 14 3 * / America/New_York` whose sole occurrence is
  the nonexistent 02:30 → **rejected at create**: `400 … does not produce any occurrences in the next year`. ✅
  → **Create-time rule:** ≥1 real occurrence within ~1 year; gap instants are excluded.
  A *daily* `30 2 * * *` in the same tz succeeds and silently skips only the gap day.
- 📄 ≤10 s of fire-time jitter; `upcoming_runs_at` timestamps are exact.

### `last_run_at`
- ✅ **Tracks scheduled (cron) fires only — a manual `/run` does NOT update it** (stayed `null` after a
  successful manual run). It's the "last cron tick", not "last execution". ❓ Whether a *failed* scheduled
  fire advances it is unconfirmed (§14).

---

## 5. Validation catalog ✅

All failures are `400 invalid_request_error` (unless noted), with precise messages:

| Probe | Result |
|---|---|
| `expression:"not a cron"` | `… not a valid 5-field cron expression: expected 5 fields, got 3` |
| 6-field cron | `… expected 5 fields, got 6` |
| `@weekly` | `… predefined shortcuts (@daily, @hourly, …) are not supported` |
| `* * * * FRI` | `… invalid value "FRI" in field day-of-week` |
| `99 20 * * 5` | `… field minute value 99 is out of range` |
| `timezone:"Mars/Phobos"` | `… "Mars/Phobos" is not a valid IANA timezone` |
| `schedule.type:"interval"` | `… schedule.type: "interval" is not a valid value` |
| **omit `schedule`** | **`200` — schedule is OPTIONAL** (`schedule:null`, `status:"active"`) |
| omit `initial_events` | `initial_events: Field required` |
| `initial_events:[]` | `initial_events: value must contain at least 1 item` |
| `initial_events:[{type:"user.interrupt"}]` | `… initial_events[0].type: "user.interrupt" is not a valid value` |
| omit `name`/`agent`/`environment_id` | `<field>: Field required` |
| unknown field (update) | `<field>: Extra inputs are not permitted` (strict schema) |
| bogus `environment_id` (create **or** update) | `404 not_found_error` — *Referenced environment not found or not accessible.* |
| bogus `vault_ids` | `404 not_found_error` — *Referenced vault not found or not accessible.* |

---

## 6. Pagination & list filters

List uses an **opaque forward cursor** (`PageCursor`), *not* the `has_more`/`after_id` style of
environments/sessions: `{ "data":[…], "next_page":"page_<tok>"|null }`; consume `?page=<tok>&limit=N`
(limit default 20, max 1000) 🧰. ✅ confirmed `next_page` round-trips and is `null` at the end.

- `deployments.list` filters 🧰: `agent_id`, `status` (`active`|`paused`), `include_archived` (bool;
  **default excludes archived** ✅), `created_at[gte]`, `created_at[lte]`, `page`, `limit`.
- `deployment_runs.list` filters 🧰: `deployment_id` (omit ⇒ all in workspace), `has_error`
  (`true`=non-null error, `false`=non-null session_id) ✅, `trigger_type` (`manual`|`schedule`),
  `created_at[gt|gte|lt|lte]`, `page`, `limit`.

---

## 7. Versioning ✅ (mirrors sessions exactly)

- `agent` bare string → snapshots **latest-at-write-time** into concrete `agent.version` and **freezes** it.
- A pinned deployment **does not auto-follow** later agent updates (v1 deployment stayed v1 after agent→v2);
  a fresh bare-string create pinned **v2**; updating with a bare string **re-pinned to v2**.
- Explicit `{type:"agent", id, version:N}` pins `N`.
- **Agent** update is optimistic-locked (`{version:<current>,…}`, else `version: Field required`).
  **Deployment** update has **no** version lock — plain partial merge.
- Mutable on update 🧰/✅: `name`, `metadata`, `schedule` (recomputes `upcoming_runs_at`),
  `initial_events`, `environment_id` (validated), `agent`, `resources`, `vault_ids`, `description`.

---

## 8. Lifecycle ✅

`pause`/`unpause` return the **full deployment object**. Three lifecycle transitions that must not be conflated:

| Trigger | `archived_at` | `status` | `paused_reason` | run recorded? |
|---|---|---|---|---|
| **manual pause** ✅ | null | `paused` | `{type:"manual"}` | — |
| **manual unpause** ✅ | null | `active` | `null` | — |
| **subagent archived** 📄 | null | `paused` | `{type:"error", error:{type:"agent_archived_error",…}}` | ✅ failed run, then auto-pause (recoverable) |
| **agent archived** ✅ | set (cascade) | stays `active` | unchanged | ✗ no run (auto-*archive*, terminal) |

- Pause: `upcoming_runs_at` **stays populated** (suppression is at trigger time, not a schedule wipe) ✅.
- **Manual `/run` works while paused** ✅. Re-pausing an already-paused deployment is **idempotent** ✅.
- Unpause resumes from the next occurrence; 📄 **missed triggers are not backfilled**.

---

## 9. Runs ✅

### `deployment_run` object

```json
{
  "type": "deployment_run",
  "id": "drun_01Bk5vZY5FApeu756uFoGCYs",
  "deployment_id": "depl_01Mnw655idRwxKWHGWfN9Xmr",
  "trigger_context": {"type": "manual"},
  "session_id": "sesn_01H2yvMQhZ5BUNwGvQC1s9aV",
  "error": null,
  "agent": {"type": "agent", "id": "agent_01KC8coCbPzCWy2H2EMqqL3A", "version": 2},
  "created_at": "2026-06-10T17:31:10.902714Z"
}
```

- **No `status` field** — success ≡ `error == null`.
- `trigger_context`: `{type:"manual"}` for `/run` ✅; `{type:"schedule", scheduled_at:<RFC3339>}` for cron fires 🧰 (SDK-confirmed; `scheduled_at` is the intended fire instant — ❓ pre- vs post-jitter unconfirmed, §14).
- `agent` = the deployment's pinned agent snapshot **at run time**.

### `/run` semantics
- **Always returns HTTP 200 + a `deployment_run`.** Failure is encoded *in* `error`, `session_id:null` ✅
  — `/run` creates a *run record*, which may or may not yield a session.
- On success: deployment **`name` → session `title`** ✅; `initial_events` → session **`events[0]`
  `user.message`** verbatim ✅; `session_id` synchronous; session row created ~0.3 s *before* the run row.
- `?has_error=true` filters to failed runs only (success excluded → `n=0`) ✅.

---

## 10. Error types ⚠️🧰 (discrepancy resolved; full enum recovered from SDKs)

The wire returns the **`_error`-suffixed** form — ``"type":"environment_archived_error"``,
message ``environment `env_…` is archived`` ✅. **The official docs page is right; the
`anthropics/skills` summary that dropped the suffix is WRONG**, and its invented `service_unavailable`
**does not exist** in any SDK. ⚠️

The complete **run-error enum is 16 members** 🧰 (byte-identical across Python/TS/Go/Java/C#, the
authoritative Stainless-generated source — far more than the 3 documented):

| `error.type` | Source | Meaning |
|---|---|---|
| `environment_archived_error` | ✅ wire | referenced env archived |
| `agent_archived_error` | 🧰📄 | (sub)agent archived — drives both auto-archive (top agent) & auto-pause (subagent) |
| `session_rate_limited_error` | 🧰📄 | session-create rate-limited — **run-only**, retries next slot, never pauses |
| `session_creation_rejected_error` | 🧰 | other session-create transient — **run-only** |
| `environment_not_found_error` | 🧰 | env missing/inaccessible (≠ archived) |
| `vault_not_found_error` | 🧰 | (resolves the skills-repo's bare `vault_not_found` — real, suffixed) |
| `vault_archived_error` | 🧰 | |
| `file_not_found_error` | 🧰 | `resources` file member |
| `memory_store_archived_error` | 🧰 | `resources` memory_store member |
| `skill_not_found_error` | 🧰 | |
| `session_resource_not_found_error` | 🧰 | |
| `workspace_archived_error` | 🧰 | |
| `organization_disabled_error` | 🧰 | |
| `self_hosted_resources_unsupported_error` | 🧰 | self-hosted env class |
| `mcp_egress_blocked_error` | 🧰 | |
| `unknown_error` | 🧰 | catch-all terminal bucket |

**`paused_reason` error enum = 14 members** 🧰 — the run set **minus** `session_rate_limited_error`
and `session_creation_rejected_error`. The asymmetry is meaningful: a transient *session-capacity*
failure fails one run and retries; only a *structural resource* failure auto-pauses the deployment.

---

## 11. Terminal & cascade behaviors ✅

- **Archive environment** (terminal) → `env.state:"archived"`. A still-active deployment's `/run` then
  returns **200 + a failed run** (`environment_archived_error`, `session_id:null`). **Creating** against
  an archived env → **404** (archived ≡ "not found/not accessible").
- **Archive agent** → **atomic auto-archive cascade**: each referencing deployment's `archived_at` is set
  to the **exact same timestamp** as the agent archive; **`status` stays `"active"`** (archival lives only
  in `archived_at`).
- **`/run` on an archived deployment** → **400 `"Cannot modify archived deployment"`** — **no run recorded**
  (contrast the env-archived case, where the deployment is still active so a failed run *is* recorded).
- **Delete surface:** environments/memory_stores/sessions have `DELETE` (env →
  `{type:"environment_deleted"}`). Deployments and agents have **archive only**, no delete.

---

## 12. Observed live behavior ✅

The scheduler is **real and prompt**: a `*/1 * * * *` test deployment produced **~1 session/minute**
during the ~6-minute probe (6 scheduled sessions + 2 manual). Independently confirms scheduled triggers
create sessions, minute-granularity firing, and prompt dispatch. *Op note:* an every-minute deployment is
a real session faucet — pause/short-TTL it during testing. (No deployment/run webhooks exist 🧰 — failed
fires are observable only by polling `has_error=true`.)

---

## 13. Implications for the aios `triggers` primitive (`#818` / `#819`)

> Synthesised from the capture + SDK cross-check, **verified against the tree**. Aligned to the unified
> `triggers` primitive (the 2026-06-10 trigger-primitive design that folds the old `#467` cron-umbrella +
> `#636` `scheduled_tasks` into one entity with `source` + `action` unions). The structural argument below
> was posted as the wire-probe comment on `#818`.

### Where the deployment lands in the triggers model

An Anthropic deployment is a trigger with `source=cron` (tz-aware), an action that **spawns a fresh session
per fire** (a `spawn_session` kind **not yet in the slice-1 union**), and an **operator owner** (slice-4
nullable `owner_session_id`). It is *not* slice-1 `wake_session` (which wakes an *existing*, session-owned
session) and *not* `sandbox_command` (bash, no model wake). Three agent-target primitives collapse if you
call `wake_session` "the deployment-compatible shape":

| Action kind | Target | Owner | Anthropic has it? |
|---|---|---|---|
| `sandbox_command` (slice 1) | bash in the owner's sandbox, no model wake | session (`owner_session_id NOT NULL`) | ❌ |
| `wake_session` (slice 1) | **existing** owner session | session | ❌ — no persistent session to re-ping |
| **`spawn_session`** (unmodeled) | **fresh** session from frozen agent-vN + env + `initial_events` | operator (slice 4) | ✅ — *this is the deployment* |
| `workflow` (slice 2 / `#819`) | deterministic run | session/operator | ❌ — their only target is a model session |

Verified: a `*/1 * * * *` deployment produced **6 distinct `sesn_` IDs for 6 fires**. The launch-bound bundle
(agent-version-pinned + `environment_id` + frozen `initial_events`) is the same framing aios uses for vaults /
run-scoped filesystem — but it belongs to the `spawn_session` × operator-owned cell, the *opposite* end of the
slice plan from where "deployment-compatible" currently sits. **Don't label slice-1 `wake_session{session_id}`
the deployment shape**, so slice 4 stays additive.

### The net-new schema: `trigger_runs` (the strongest implication)

`#818` already defines the `triggers` table (the `session_scheduled_tasks` rename + `source`/`action` unions).
What it lacks — and the probe's strongest finding argues for — is a **per-fire audit row**. Anthropic's
`deployment_run` exists *even when no action target spawns* (rate-limit / archived-resource → `session_id`
null + `error` set), decoupling *trigger-fired* from *action-succeeded*. `#818` keeps lossy `last_fire_status`.

```sql
CREATE TABLE trigger_runs (
    id              text PRIMARY KEY,            -- 'trun_'
    trigger_id      text NOT NULL REFERENCES triggers(id) ON DELETE CASCADE,
    account_id      text NOT NULL,
    trigger_context jsonb NOT NULL,              -- echoes `source`: {type:cron, scheduled_at} | {type:manual}
                                                 --                 | {type:run_completion, run_id} | ...
    action          jsonb NOT NULL,              -- snapshot of the action at fire time (kind + frozen target version)
    result_ref      text,                        -- session_id | workflow run_id | NULL on failure-before-spawn
    error           jsonb,                        -- NULL on success; {type,message} otherwise
    created_at      timestamptz NOT NULL DEFAULT now()
    -- NO status column: success == (error IS NULL). Correct-by-construction.
);
```

**Row-per-fire** (not row-per-success) is the right invariant — it generalises aios's
**tool-always-appends-result** to the scheduler tick: the *fire* always appends a run record regardless of the
action's downstream outcome. It is especially load-bearing for reactive `source=run_completion` (`#819`),
which is ~unobservable without it. `trigger_context` is just the `source` discriminator echoed at fire time —
Anthropic's `{type:manual}` / `{type:schedule, scheduled_at}` is the 2-value version of our richer `source` enum.

### Columns the probe sharpens on the existing `triggers` table

- **cron `source_spec` validation** — adopt the create-time **"≥1 real occurrence within ~1 year"** check on
  top of `croniter.is_valid()` grammar (today `scheduled_tasks` does only the grammar check). Strict 5-field
  numeric, no seconds/macros/day-names — matches the wire's rejection messages.
- **action-target version pin (the explicit pin-or-float choice, adopt-(a) in the `#818` datapoint)** —
  Anthropic **pins-and-freezes**: a bare ref snapshots latest-at-write into a frozen integer, **no
  auto-follow**, re-pinned only on an explicit update. Store the action target's version frozen
  (`agent_version` for `spawn_session`/`wake_session`; `workflow_id@vN` for `workflow`); "float" is a
  resolve-at-fire sentinel.
- **lifecycle (later slices)** — archival is an orthogonal `archived_at`, **never** a status value
  (`status` ∈ {active, paused}; agent-archive cascades `archived_at` while status stays `active`);
  `paused_reason` is discriminated `{type:manual}` (operator) vs `{type:error, error:{…}}` (system
  auto-pause) — the model-consciousness line. Matches aios's `sessions.archived_at` convention; model both as
  discriminated jsonb, not booleans.

### Reuse the existing scheduler verbatim

`harness/scheduler.py` already does claim-and-defer for `scheduled_tasks`: a `MIN(next_fire)` sleep loop woken
by NOTIFY on `aios_scheduled_tasks_due` (`listen_for_scheduled_tasks_due`), claim-due-rows-and-advance-
`next_fire` in one transaction, then `defer` a procrastinate job per row. Extend it so **the claim transaction
also inserts the `trigger_runs` row, then defers the action job** — a row exists per tick even if the action
later fails, and NOTIFY-after-commit keeps subscribers from seeing an uncommitted run. Composes onto "the loop
is a `wake_session` job calling `run_session_step` once": a scheduler-tick defers an action job which, for
`spawn_session`, creates a fresh session whose own loop then runs.

### Where aios should DIVERGE from Anthropic
- **Add a failed-fire NOTIFY; don't inherit poll-only observability.** Anthropic has no run webhooks — failed
  fires are visible only by polling `has_error=true`. aios is LISTEN/NOTIFY-native; emit on a failed fire so
  it's push-observable (model-consciousness: the owner of a consciously-scheduled trigger should *see* a
  failed fire, not a silence).
- **Keep `error.type` an open string, not a closed 16-variant enum** — "model sees raw errors" +
  "generalize over enumerate": mirror the `{type,message}` *shape*, let `type` carry the underlying cause;
  don't maintain a curated union in lockstep with Anthropic.
- **Preserve the run-only-vs-auto-pause asymmetry** — transient session-capacity failures fail one run and
  retry; only structural resource failures auto-pause. Same "model's fault vs substrate's" line aios draws.
- **Timezones only at the operator/`spawn_session` tier**, not on slice-1's UTC `sandbox_command`/`wake_session`
  primitives. Where tz lands, preserve literal wall-clock (fall-back "fires twice", not deduped);
  `compute_next_fire` takes a `tz` and stores a UTC `timestamptz`.
- **Overlap policy is a stated decision, not emergent.** Anthropic overlaps freely (each fire = a brand-new
  session). With aios's per-session `lock="{session_id}"`, distinct fresh `spawn_session` fires each get their
  own lock, so "overlap freely" is the cheap default — but cap it against the environment's concurrency so a
  `*/1` trigger with long sessions can't stampede the sandbox pool.

---

## 14. Cross-verification & open follow-ups

**Sources reconciled:** official docs (`scheduled-deployments.md`, `reference.md`) + the 5 Stainless SDKs
(read directly from `raw.githubusercontent.com`, `main`). SDKs agree field-for-field and enum-for-enum, so
they are treated as authoritative where the wire couldn't induce a state.

**Corrections folded above:** 16-member run-error enum (vs 3 documented); `paused_reason` error variant +
14-member subset; `resources` 4-member union; `initial_events` 3-type union; `GET /v1/deployment_runs/{id}`;
full list-filter surface; `trigger_context` scheduled shape; `service_unavailable` disconfirmed;
`vault_not_found_error` (suffixed). The capture's suffix-skepticism, schedule-optionality, no-version-lock,
DST rules, archived-vs-status orthogonality, and `last_run_at`-scheduled-only inference were all confirmed.

**Open (would need a second live probe — key was shredded):**
1. `trigger_context.scheduled_at` — pre-jitter cron slot vs post-jitter actual? (diff a scheduled run's
   `scheduled_at` against the matching `upcoming_runs_at` and `created_at`).
2. Does a *failed* scheduled fire advance `schedule.last_run_at`? (manual confirmed not to.)
3. Multi-message / non-text `initial_events` — N kickoff turns vs concatenation; image/document blocks.
4. Concurrency/overlap when a prior fire's session is still running at the next slot (overlap vs skip vs queue).
5. Subagent-archived auto-pause end-to-end (failed run recorded + `paused_reason:{type:error}`).
6. `file` / `github_repository` resource live-validation envelopes (only `memory_store` exercised).
7. `deployment_runs` list ordering (newest- vs oldest-first), cursor-at-scale, retention window.
8. Can `model`/`allowed_tools`/`mcp` be set at deployment level, or are they agent-frozen? (extends the
   unknown-field 400 probe — would confirm "one agent, N parameterised deployments" can/can't vary the model).
9. Limit matrix beyond 1000/org: max `initial_events` bytes, max `resources`/`vault_ids`, per-workspace vs
   per-org cap, name/metadata caps.

---

## Appendix — probe hygiene

- Every call's `request-id` was captured (response header dumps) for traceability.
- Created: 1 environment, 1 agent (→v2), 1 memory store, ~12 deployments, 8 sessions (2 manual + 6 from the
  `*/1` scheduled faucet).
- **All cleaned up:** sessions deleted, memory store deleted, environment deleted, agent archived, all
  deployments archived (agent-archive cascade). Final state verified: 0 active deployments, 0 sessions in the
  probe env. The API key was shredded from disk post-probe.
