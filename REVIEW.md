# Uncorrelated review — gVisor snapshot empty-floor, tip `e28d3c399f0568d43d37b0332b71f2ba985188ed`

- **Round**: `gvisfloor2` (implementer grok-4.6, worktree `aios-gvisfloor2`; reviewer claude-opus-5, worktree `aios-gvisfloor2rev`)
- **Product tip**: `e28d3c39` — `fix(sandbox): skip-empty identity is SizeRw minus create-time baseline`
- **Prior round**: `77e18b27` (FAIL, review `b3cf1342`) — runtime-keyed 64 KiB absolute floor
- **Base**: `cc9a3c0e` — `fix(sandbox): apply runsc egress rules in the target Sentry (#2410)`
- **Target defect**: master RED gVisor Validation, Actions 35146073686 —
  `tests/e2e/test_sandbox_persistence.py::test_zero_write_release_is_skipped_empty`
  expected `skipped_empty`, got `committed` at `empty_floor_bytes=8192`.
- **Verdict**: **PASS** (3 Medium notes, 3 Low notes — none blocking; see Follow-ups)

Both blocking items from the prior FAIL are addressed at the cause, not at the
expectation. The runtime threading makes the gVisor job's containers actually
gVisor containers, and the identity gate stops guessing at an unmeasured
constant: it subtracts a **per-corpse measurement** taken on that same
container at create. That inverts the failure mode of the prior tip — where a
64 KiB absolute floor sat exactly on `_write_substantial`'s 65536-byte blob and
could have *discarded* a real tenant write — into a fail-closed one (unstamped
⇒ commit). mypy, ruff and every focused unit test pass locally.

---

## Verified

**1. e2e specs now thread the runtime (TASK item 1).** `runtime=get_settings().sandbox_runtime`
is set in all five direct-backend spec builders — `test_sandbox_persistence.py:78`,
`test_sandbox_provision_path.py:81`, `test_sandbox_salvage.py:94`,
`test_sandbox_seccomp.py:82`, `test_sandbox_broker_reachability.py:132` —
matching the pre-existing precedent at `test_sandbox_ipv6_lockdown.py:230`.
`get_settings` is already imported in each of the five files (checked; no
NameError at collection). The gVisor leg exports `AIOS_SANDBOX_RUNTIME=runsc`
for the whole e2e step (`.github/workflows/gvisor-validation.yml:177`), and
`DockerBackend.create` emits `--runtime` under `if spec.runtime:`
(`docker.py:471-472`), so these containers now genuinely carry a Sentry. The
prior tip's gate keyed on a condition its own container never met; that is
fixed.

**2. Identity is container-aware (TASK item 2).** `create` stamps SizeRw before
any tenant exec (`docker.py:504-506` → `_stamp_snapshot_baseline`, `docker.py:1558`)
onto the backend's `_snapshot_baselines` and the handle (`base.py:216-219`); the
gate at `docker.py:889-891` compares `writable_layer_delta(size_rw, baseline)`
against the configured floor. Keying checked: `create` stores the full 64-hex id
from `docker run` stdout, and the salvage/GC ref path re-inspects with
`{{.Id}}` after `docker ps --quiet` (`docker.py:637`), so refs carry the full id
and lookups hit — no short-id mismatch. Entries are popped in `destroy`
(`:578`) and `force_remove` (`:677`); the pop precedes removal and follows the
snapshot, so the release ordering invariant is intact.

**3. Fail-closed paths are real, not nominal.** `run_docker_cli` raises
`SandboxBackendError` on launch failure *and* timeout (`_subprocess.py:113-115`),
which `_stamp_snapshot_baseline` catches; an unparseable `.SizeRw` returns
`None` (`docker.py:1556`). Both degrade to baseline 0 ⇒ commit. Importantly,
neither can turn a successful `docker run` into a raised `create()`: the
registry's create-failure handler deliberately does **not** destroy ("there is
no sandbox yet", `registry.py:693-706`), so a raising stamp would have leaked a
live container. It doesn't.

**4. Safety vs a real tenant write (TASK item 3).** The floor is now 8 KiB *of
tenant delta*, against `_write_substantial`'s 65536-byte blob
(`test_sandbox_persistence.py:86`) — an 8× margin, where the prior tip's runsc
floor was 1.0×. The discard window is bounded at one page of tenant writes
regardless of store or runtime, and no longer widens with the copy-up. The
delta framing also *strengthens* the identity claim rather than weakening it:
what it subtracts is precisely the daemon-injected per-life metadata
(`/.dockerenv`, hosts/hostname/resolv.conf + parent dirs), which is
regenerated identically on the next container and is not tenant content.
Unit coverage pins the boundary — `70_000+8192` skips, `70_000+8193` commits,
unstamped `65536` commits (the master RED signature), negative delta clamps
(`tests/unit/sandbox/test_snapshot_verb.py:209-263`), plus the stamp itself and
its unparseable-SizeRw degrade (`:284-321`).

**5. Budget arithmetic left alone.** `projected_unique` / flatten sizing still
use absolute `size_rw` (`docker.py:910-912`, `:930`) — correct: storage is
storage, only *identity* is tenant-relative. No inconsistency introduced.

**6. #2410 intact (TASK item 4).** `git diff cc9a3c0e..HEAD -- src/aios/sandbox/backends/docker.py`
touches no hosts-first / operator-mount / Sentry-exec / proxy-key code; the only
runsc mention in the diff is a comment in the identity block. The helper
`snapshot_empty_floor_bytes` and `RUNSC_SNAPSHOT_EMPTY_FLOOR_BYTES` are deleted
outright (not shimmed), `registry.py:1822` is back to the plain setting, and a
tree-wide grep finds no stale references in src, tests, docs, README or
`.env.example`. Design doc §3 amendment and the §5.2 pseudocode were both
updated to the delta form.

**7. Commit message vs diff (TASK item 6).** Claims check out, with one
overstatement (M1) and one wording inaccuracy (L1) below.

### Commands run (focused; no full suite, no `-n`)

```
uv run pytest tests/unit/sandbox/test_snapshot_verb.py \
  tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/test_config.py -q   → 83 passed
uv run pytest tests/unit/sandbox tests/unit/test_sandbox_init_reaper.py \
  tests/unit/test_sandbox_pull_always.py tests/unit/test_sandbox_resource_caps.py \
  tests/unit/test_networking.py -q                                             → 797 passed
uv run pytest tests/unit/test_gvisor_validation_workflow.py -q                 → 5 passed
uv run mypy src tests            → Success: no issues found in 1101 source files
uv run ruff check src tests      → All checks passed
uv run ruff format --check       → 1101 files already formatted
```

No Docker (and no runsc) in this environment, so the gVisor e2e was not run —
light-review scope per the prompt. CI remains the oracle for M2/M3.

---

## Findings

### M1 (Medium, non-blocking) — the baseline is process-local, so the gate is inert for cross-process corpses

`_snapshot_baselines` is an in-memory dict on the `DockerBackend` instance. The
single snapshot call site `_snapshot_and_record` (`registry.py:1805`) serves
three callers: planned release with an in-process handle (`:1800`), the salvage
preamble (`:2004`) and the GC tick (`:3129`). The latter two pass a corpse id
discovered from `docker ps` labels — and after a worker restart (or a GC tick in
a sibling worker) that container was created by a *different process*, so the
lookup misses and baseline is 0. Under the containerd image store / runsc the
empty layer is well above 8192, so the gate never fires on exactly the crash
path §5.4 exists for: a chat-only session that crossed a process boundary
commits an identity layer.

This is **by design and blessed by TASK item 2** ("unstamped corpses fail
closed"), it is not a regression against master (master compares absolute
SizeRw to the same 8192), the outcome is safe (commit, never discard), and it
matches an established convention in this file — the per-corpse
`disk_limit_bytes` already falls back to the global default for the same reason
(`registry.py:2007`, and the docstring at `:1952` says so). No cheap durable
stamp exists: a Docker label can't be written post-`run`, and the handle is
itself process-local, so durability would mean a DB column keyed by container
id (a migration).

What is worth correcting is the **claim**, not the code: the commit body says
the skip holds "under runc, containerd-snapshotter, and runsc alike" without
the caveat, and `_stamp_snapshot_baseline`'s docstring hedges it only as "in
this process". Recommend a sentence in the design doc §3 amendment naming the
cross-process corpse as the one case that still commits, and an issue for the
durable stamp if the chain growth shows up in prod.

### M2 (Medium, non-blocking) — start-time vs stop-time SizeRw under runsc is still unmeasured

The baseline is measured on a *running* container immediately after `docker run`
returns; the gate compares SizeRw of a *stopped* one (`snapshot` does `stop -t 5`
first, `docker.py:855`). If gVisor's Sentry flushes overlay state to the
writable layer at exit — the general shape of gvisor#10256, cited by the prior
tip — the delta for a no-write container is not 0 and the e2e stays red. The
design no longer *depends* on knowing the empty-layer magnitude (that is the
central improvement over `77e18b27`), but it does still assume the empty layer
is stable across the container's life. Nothing in-tree or in the commit
measures that under runsc, and it cannot be measured here.

Failure mode if the assumption is wrong is benign (commit, not discard) and the
gVisor leg reports it. If that leg comes back red on this test, the cheapest
next step is to widen the e2e assertion message: it currently prints
`baseline` and `out.unique_bytes` (`test_sandbox_persistence.py:185-188`), but
`unique_bytes` on a `committed` outcome is the tag's unique bytes, not the
observed SizeRw — so the message cannot distinguish "baseline stale" from
"tenant actually wrote". Printing the corpse's `SizeRw` would make the next
failure self-diagnosing.

### M3 (Medium, pre-existing, adjacent) — in production the delta window opens *before* provisioning writes

The stamp is taken inside `backend.create`, but the registry then runs
`install_egress_ca` and `install_packages` against the container
(`registry.py:715-717`). `install_egress_ca` writes the PEM into
`/usr/local/share/ca-certificates/` and runs `update-ca-certificates`
(`setup.py:161-169`), which rewrites the aggregate bundle — hundreds of KiB into
the writable layer, attributed to "tenant" by the delta. Unless the
`_prewarmed_setup_satisfied` skip fires (`registry.py:715`, which keys on a
prewarm label a session snapshot tag will not carry), a chat-only session's
release therefore still commits.

This is **not caused by this tip** — the absolute floor had the identical
property — and fixing it is a registry-level change (re-stamp after setup, or
pass `baseline_bytes` into `backend.snapshot`), outside this task's scope. But
it is the reason the commit's "read/chat-only sessions never grow a chain"
(design doc §3) is, in prod, true only on the prewarm path. Worth an issue: a
post-setup stamp would make the gate actually load-bearing in production and
would simultaneously let the backend drop its dict (M1/L1 fall out with it),
since the registry holds the handle at that point.

### L1 (Low) — `SandboxHandle.snapshot_baseline_bytes` is write-only

The field is set at `docker.py:516` and read nowhere in `src/` — the gate uses
the backend dict. Its only consumer is an e2e assertion message
(`test_sandbox_persistence.py:187`). Per CLAUDE.md "compose, don't accrete",
either consume it (registry passes it into `backend.snapshot` — which is also
the shape M3 wants) or drop it and keep the dict as the single source. The
commit body's "stamped … onto the handle and the backend" reads as though both
are load-bearing; only one is.

### L2 (Low) — one extra `docker inspect --size` on every create

`_stamp_snapshot_baseline` runs on *every* `create`, including run and browser
sandboxes that are never snapshotted, on the cold-start path #1348 was
explicitly tuned for. The walk is over an empty layer so it is cheap, and it is
budgeted by `sandbox_inspect_size_timeout_seconds` rather than the blanket CLI
bound — but it is a new unconditional round trip. If cold-start latency
regresses measurably, skipping the stamp when the spec has no
`snapshot_budget_bytes`/durable rootfs would recover it.

### L3 (Low, watch-item, not a defect) — newly-runsc e2e specs may surface known gVisor defects

Threading the runtime is what TASK asked for, and it is correct — but four spec
files that previously ran under runc in the gVisor leg now run under gVisor for
the first time. Two are worth watching on the next run:
`test_sandbox_broker_reachability.py::test_sandbox_resolves_worker_alias_via_docker_dns`
is a `curl` to a *name* (`:137-144`), the same shape as the test already marked
`runsc_dns_unresolved` for aios#2430; and the seccomp e2e now asserts deny-list
behavior against gVisor's own syscall surface. A red in either is a
pre-existing gVisor defect newly exposed, not a regression from this tip —
and per the workflow's own stance (`gvisor-validation.yml:173`) it should be
fixed or honestly marked with its cause, never silently deselected.

---

## Follow-ups (none blocking this tip)

1. Document the cross-process-corpse caveat in design doc §3 / the stamp
   docstring; file the durable-baseline issue (M1).
2. Print the corpse `SizeRw` in the e2e assertion message so a runsc red is
   self-diagnosing (M2).
3. Issue: stamp the baseline after provisioning setup so the gate is
   load-bearing in production; folds in L1 (M3).
