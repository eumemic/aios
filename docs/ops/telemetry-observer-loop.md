# Machine-observer anomaly-scan loop (aios#1326)

The deterministic, substrate-different observer over the host's per-run telemetry journal.
Plane C, roadmap item 8 of the substrate-different-verdict epic (aios#1330).

Code: `src/aios/workflows/telemetry_observer.py`
(`build_observer_script` / `build_observer_workflow_create` for the observer;
`build_sweep_script` / `build_sweep_workflow_create` for the dead-man twin).
Fixture proof: `tests/integration/test_wf_telemetry_observer_fixture.py`.

---

## What it is

A scripted (NOT agent) workflow that, on every terminal run of a watched workflow, re-derives
a waste verdict from the host-emitted `WfRunUsage` numbers (`get_run`'s `usage` object, #1324 /
PR#1334). It is uncorrelated with the LLM nodes by **substrate** (host numbers, not the run's
self-report `output`) and by **stage** (post-terminal, not in-pass). It never calls `agent()`
or `gate()`: a numeric-threshold predicate has a universally-correct policy, so the verdict is
a pure deterministic script — the moment a verdict needs judgment it would have to become an
`agent()`, which would re-correlate it.

### Honest scope (non-goals)

The observer detects ONLY mechanically-legible pathologies. Semantically-wrong-but-cheap-and-
on-budget output (a Ralph-Wiggum loop producing a plausible diff), design-misallocation, and
coverage-selection blind spots trip NO numeric anomaly and stay chairman/seat-found (the
irreducible residue, by the uncorrelated-immune theorem). **v1 ships with EXACTLY ONE live
signal — the LEAD `cost-on-errored` signal.**

---

## Signals and substrate-availability

The read-path `WfRunUsage` surfaces ONLY:
`cost_microusd · input_tokens · output_tokens · cache_read_input_tokens ·
cache_creation_input_tokens · iteration_count (ALWAYS null today) · wall_clock_ms`.

| Signal | Field(s) | Status |
|---|---|---|
| **LEAD cost-on-errored** | `status` + `cost_microusd` | **LIVE** |
| iteration-at-max | `iteration_count` (always null today) | DEFERRED — gated on a real host iteration counter |
| byte-buffer | `input_tokens` + `net_diff_lines` | DEFERRED — gated on a host primitive emitting `net_diff_lines` |
| cost-per-net-accepted-line | `cost_microusd` + `net_diff_lines` | DEFERRED — with byte-buffer |
| cost-on-no-PR | PR-presence (lives only in the run `output`, forbidden) | DEFERRED — gated on a host primitive emitting PR-presence |

The three DEFERRED signals are specified as frozen-band constants and fixture-locked against a
**fabricated** `get_run` return to PIN their predicate contract, but they are **NOT wired into
the live verdict path**. They cannot masquerade as live: the fail-loud rule makes any null/
absent field used by a fired predicate yield `cannot-determine`, never a false `ok`/`anomaly`.

**A NEW host primitive is required to lift byte-buffer / cost-on-no-PR / cost-per-net-accepted-
line out of DEFERRED** (emit `net_diff_lines` and PR-presence onto the run read path, WITHOUT
making the observer read the run's `output`). **A NEW per-run `iteration_count` counter** is
required to lift the iteration signal out of DEFERRED. File both as follow-up dependencies.

### The LEAD verdict (deterministic, vs frozen git-versioned bands)

- `status == 'errored'` ∧ `cost_microusd > COST_WASTE_FLOOR` → `anomaly` (`cost-on-errored`).
- `status == 'errored'` ∧ `cost_microusd` absent/null → `cannot-determine` (fail-loud).
- `status == 'errored'` ∧ `cost_microusd <= COST_WASTE_FLOOR` → `ok` (cheap fast-fail).
- any non-errored terminal status → `ok` (the lead predicate does not fire; no other live
  signal in v1).

The bands are PREPENDED constants in the committed script header (`COST_WASTE_FLOOR`,
`ITER_MAX_BAND`, `BYTE_BUFFER_INPUT_BAND`, `DIFF_FLOOR`, `BANDS_VERSION`). The body imports
neither `datetime` nor `time` and never infers a band at runtime — it gates on a frozen
threshold, never on recency. Bumping any band MUST bump `BANDS_VERSION` (the residue table
#1328 keys stored verdicts on it).

### The verdict record (the observer's ONLY output)

```
{run_id, workflow_id, verdict ∈ {ok, anomaly, cannot-determine},
 signals_fired:[...], telemetry_snapshot:{...}, bands_version}
```

Machine-readable telemetry, not a log scrape. The observer files NO GitHub issue / obliging
action-item — this is enforced **structurally** (no `http_request` in its tool surface), so it
cannot file into the Class-closed:NO graveyard before the fractal-retro closure machinery
(eumemic-company#24) lands. Its output is the `finder=internal-armed-check` axis-1 rows for the
residue table (#1328), and **a passing observer is FORBIDDEN as a safety signal for any
autonomy increment** (the de-Goodharted gauge).

---

## The input-envelope unwrap (TRACED, not assumed)

A `WorkflowAction` fire's run input is ALWAYS `{"trigger": <firing context>, "input":
<template verbatim>}`. For a `run_completion` fire the completing run rides BY VALUE under
`input["trigger"]["run"]` = `{id, workflow_id, status, output, error}`
(traced to `harness/trigger_runner.py`'s `compose_workflow_run_input` + `_run_workflow`).

**There is NO `input["trigger"]["event"]` for run_completion fires** — `event` is passed as
`None` (it is non-None only for `external_event` fires). The earlier draft's citation of
`input["trigger"]["event"]["run_id"]` was FALSE; the correct key path is
`input["trigger"]["run"]["id"]`. The observer's `_run_id` reads that path AND accepts a bare
`{"run_id": ...}` (the fixture / arm-time path). It reads `trigger.run` ONLY for the id — it
NEVER reads `trigger.run.output` (the forbidden self-report substrate); the verdict is computed
from `get_run`'s host numbers.

---

## Wiring blueprint (git-versioned watched-set)

`RunCompletionSource.workflow_id` is a SINGLE id — there is no all-workflows wildcard — so the
loop is **one trigger per watched workflow**, all pointing the same `WorkflowAction` at
`telemetry_observer`. The observer is stateless w.r.t. which workflow fired it (it reads the
run by id). **The watched-set is this git-versioned list; adding a workflow = adding a trigger,
never a code change.**

Watched workflows (v1):

- `dev_pipeline`
- `triage_pipeline`
- (each conveyor/loop workflow as it lands)

For each watched workflow `W`, create (via the `trigger_create` tool):

```
source = RunCompletionSource(workflow_id=<W>, statuses=["completed","errored","cancelled"])
action = WorkflowAction(workflow_id=<telemetry_observer>, input_template=None)
```

`statuses` defaults to all-terminal (the no-silent-omission default). The observer reads the
completing run's id from `input["trigger"]["run"]["id"]`.

### The dead-man twin (`observer_sweep`) — REQUIRED

A `RunCompletionSource` AUTO-DISABLES after `MAX_CONSECUTIVE_FAILURES` and CANNOT fire on its
own non-firing — a dead observer is silent, and silence ≠ health. The cron twin is the dead-man
over the alarm. It is a SEPARATE workflow (its crash mode must not entangle the observer),
fired by:

```
source = CronSource(schedule="*/15 * * * *")   # sized to the watched-workflow completion rate
action = WorkflowAction(workflow_id=<observer_sweep>,
                        input_template={"watched_workflow_id": <W>,
                                        "observer_workflow_id": <telemetry_observer>,
                                        "after": <window cursor>})
```

It reads the terminal-run count (`list_runs` over `completed`/`errored`/`cancelled`) and the
observer's own scan count, and rings (`alarm: True` / `cannot-determine`) when `N_runs >
N_scans`. The `list_runs` read applies the **no-silent-degrade invariant (#1323)**: a FULL page
(returned count == requested limit) means rows may exist unseen → the sweep cannot prove the
count → `cannot-determine`, NEVER a silent under-count read as "all scanned".

---

## Arm-time known-anomalous proof (criterion 14)

> A monitor is not armed until its alarm has rung once.

After deploying `telemetry_observer` AND wiring its triggers, fire a synthetic run carrying
KNOWN-anomalous telemetry through the LIVE observer and assert it returns `verdict:anomaly`:

1. Cause (or replay) a run that **errored** with `cost_microusd` above `COST_WASTE_FLOOR`.
2. Invoke the observer against that run id (bare `{"run_id": <id>}` is accepted).
3. Assert the returned record is `verdict:anomaly`, `signals_fired:["cost-on-errored"]`.

**This requires PR#1334 DEPLOYED on the worker — not merely merged.** A pre-#1334 worker
returns a bare `WfRun` with no `usage` object, so the synthetic run reads `cost_microusd` as
absent → `cannot-determine`, silently failing to actually arm. The CI fixture
(`test_lead_cost_on_errored_above_floor_is_anomaly`) is the offline twin of this proof, and
`test_errored_with_absent_usage_object_is_cannot_determine` demonstrates exactly the pre-#1334
failure mode.

---

## Sequencing gates (criterion 15)

- **Go-live is gated on PR#1334 DEPLOYED** (telemetry on the LIVE read path — "deployed", NOT
  merely "merged" / "#1324 closed") **AND deploy-on-merge #1175** (a verification node the
  worker isn't running is hollow). Both are merged in this repo; go-live waits on the live
  worker re-registration.
- Sequenced AFTER #1324 (telemetry surface, PR#1334) and #1323 (list-read invariant). FEEDS
  item 9 (#1328 residue table).
- **Observer output is axis-1 only and FORBIDDEN as a justification for any autonomy
  increment.**

## Named open risks (carried, out of scope here)

- **Frozen-band drift.** The bands drift by construction under lily-pad cost arbitrage — they
  need their own closure-sentinel re-tune cadence + TTL dead-man (a band nobody re-tunes is a
  dead alarm). MUST be recorded as a known miss class.
- **The DEFERRED-signal host primitives** (`net_diff_lines` / PR-presence on the read path; a
  real per-run `iteration_count` counter) are the dependencies that lift signals 5/6/7 out of
  DEFERRED. Filed as follow-up dependencies; this loop does NOT assume them.
