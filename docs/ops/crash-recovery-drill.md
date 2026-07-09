# Crash-recovery drill (issue #147's manual repro, scripted)

`./scripts/crash-recovery-drill.sh` is the scripted form of issue #147's
manual repro: boot a real `aios worker`, get a file-barrier bash tool call
in flight, `kill -9` the whole process group, boot a second worker, and
assert the session's turn completes.

## Why this exists, and why it isn't in CI

An audit into "should aios run a real `kill -9` end-to-end test in CI"
concluded no — see the "Crash-recovery test architecture" note at the top of
[`tests/e2e/conftest.py`](../../tests/e2e/conftest.py) for the full
reasoning. In short: Postgres commits `append_event` atomically under a
per-session row lock, so a real kill's *only* content a faithful simulator
(`Harness.simulate_sigkill`) can't already reproduce is timing and Postgres
teardown semantics — neither of which is aios code — while a permanent
out-of-process harness (subprocess worker, cross-process fake model server,
dedicated DB) has no CI lane to live in (the `slow` marker's "run nightly" is
aspirational; nothing runs `-m slow`).

Three deterministic tests close the gaps that analysis actually found:

- `tests/integration/test_worker_startup_recovery.py` — the boot-recovery
  **composition** test (seeds the full post-kill residue at once, asserts
  one `run_startup_recovery` pass converges it in the pinned order).
- `tests/e2e/test_sandbox_salvage.py::test_running_corpse_is_salvaged_then_resumed`
  — the **running-corpse** test (the one state only a real SIGKILL produces:
  graceful shutdown STOPs every container; a kill leaves it RUNNING).
- `Harness.simulate_sigkill`'s docstring — the **crash-state contract**
  every crash-recovery test in the suite relies on, plus the checklist that
  defends against simulator drift as new process-local worker state is
  added.

The accepted residual risk: no *automated* anchor exercises a genuinely
killed process end-to-end. This drill is the mitigation — a manual tool, run
by a human (or a future non-gating nightly job, if one is ever built) before
promoting a change to the crash-recovery path, not a CI check.

## When to run it

Before promoting a change that touches:

- `src/aios/harness/worker.py`'s startup-recovery sequence
  (`run_startup_recovery` and its ordering)
- `src/aios/harness/sweep.py`'s ghost repair / stalled-job reaping
- `src/aios/sandbox/registry.py`'s crash-corpse salvage preamble
- anything that adds new **process-local** worker state (an in-memory
  registry like `InflightToolRegistry` or `_INFLIGHT_HARVESTS`) — the drill
  is the last line of defense that a boot-time reset was actually wired in
  and actually fires against a real kill, not just against the simulator.

## Prerequisites

- A dev worktree bootstrapped via `./scripts/dev-bootstrap.sh` (or
  equivalent: `AIOS_DB_URL` / `AIOS_API_KEY` / `AIOS_VAULT_KEY` /
  `AIOS_EGRESS_CA_KEY` set in `.env`, migrations applied, the api reachable
  at `AIOS_URL`).
- `uv` on `PATH`.
- A real model configured for the drill's scratch agent (the script defaults
  to `openrouter/test`; edit the agent payload in the script if your
  worktree's credentials point elsewhere) — this drill exercises a real
  worker doing real inference, unlike the test suite's scripted fake model.
- Run on the HOST, not via `docker compose` — the drill needs direct
  process-group control (`kill -9 -$PID`) that compose's kill semantics
  don't give cleanly.

## Running it

```console
$ ./scripts/crash-recovery-drill.sh
```

The script:

1. Creates a scratch agent/environment/session.
2. Boots worker #1 and sends a message that asks the model to run a
   file-barrier bash command (`while [ ! -f BARRIER.go ]; do sleep 0.2;
   done`) — this blocks the tool task in-container, giving a reliable window
   to kill mid-execution instead of racing a fast command.
3. `kill -9`s worker #1's **entire process group** (`kill -9 -$PID`, not
   just the parent PID — a bare parent kill can leave orphaned
   grandchildren, which would understate what a real crash leaves behind).
4. Boots worker #2 and watches its log for `sweep.reaped_stalled_jobs` (the
   production telemetry signal issue #1757's audit named as the mitigation
   for this residual risk).
5. Polls the session via the API until it reaches `status: idle`, proving
   the turn actually completed post-recovery — not just that the ghost was
   silently discarded.
6. Cleans up (archives the scratch agent/session, removes the barrier file).

Exit code `0` iff the session reaches `idle` within the poll window after
worker #2 boots. On failure, the script points at the two worker log files
it kept in `$TMPDIR` for inspection.

## Scope note

This drill is **not** wired into any CI workflow and must never be added to
the PR-gating docker shard — see the rationale above. If a non-gating
nightly CI lane is ever created, promoting this script there can be
reconsidered at that time.
