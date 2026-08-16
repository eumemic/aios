# Durable session sandboxes: full-filesystem persistence

**Status**: design recommendation, pre-implementation (2026-06-10)
**Recommendation**: Mechanism B — disposable containers + commit-snapshot — with the
amendments in §5, the production prerequisites and durability/scale decision in §9, and
the sign-off items in §11. Horizontal scale (a second worker, host
replacement/rebalancing) is a stated requirement and is carried by the design from v1.

---

## 1. Summary

Make the whole sandbox root filesystem survive idle reclaim: every *planned* teardown
becomes **stop → `docker commit` to a per-session local image → rm**, and every resume is
the **existing cold-start provision path** running from that image instead of the base.
Containers stay disposable; the filesystem becomes the durable artifact. Drop `--rm`, so
*unplanned* deaths (crash, OOM, daemon restart) leave a stopped corpse that is salvaged
(committed) before the session's next container starts — container death stops losing data
entirely, which is strictly better than today.

Horizontal scale — a second worker, and/or host replacement/rebalancing — is a **stated
requirement**, so durable state splits cleanly: **the DB carries a snapshot pointer
(`snapshot_ref` / `snapshot_host` / `snapshot_bytes` / `snapshot_updated_at` — location
and existence), while the daemon/store remains the source of truth for artifact content
and lineage.** Snapshot transport sits behind a `SnapshotStore` seam whose v1
implementation (`LocalDaemonStore`) is an identity wrapper over the local daemon —
today's behavior exactly — so multi-host is configuration plus a second store
implementation, not redesign (§5.11). The migration drops the dead
`sessions.container_id` and adds the pointer columns. The artifact itself is a
deterministic local tag in v1 (a host-independent store ref under a remote store); GC
metadata rides on image labels; a single retain-rule
reconciler replaces `reap_orphans` and doubles as the convergence authority that corrects
pointer drift.

Two facts discovered on the production host are prerequisites, not details (§9): Server B
runs an **hourly `docker image prune -af --filter until=48h` cron** that would silently
delete every snapshot idle >48h (a lockstep eumemic-ops change must exempt
`aios.managed=true` images *and* introduce a snapshot disk budget, because the exemption
removes the only bound that exists today), and prod runs the **containerd image store**,
not the overlay2 graphdriver every Docker experiment in this design was verified on — the
verification battery must be re-run there before implementation.

## 2. Corrections to the brief (verified against code)

The brief's current-state sketch was mostly right. Four corrections:

1. **The lease columns are gone.** `lease_worker_id` / `lease_expires_at` were dropped in
   migration `0002` (Phase 5 replaced the DB-lease protocol with procrastinate's `lock`).
   Only `sessions.container_id` (migration 0001) survives, and it is dead — zero SQL
   reads/writes anywhere. It should be **dropped**, not repurposed.
2. **`reclaim_session_if_idle` (loop.py:256) is not the sandbox idle hook.** It is
   archive-on-quiescence for `archive_when_idle` workflow sessions (a conditional
   `UPDATE sessions SET archived_at = now()`). The sandbox idle path is the registry's
   own reaper: `_reap_idle_once` → `release()` → `docker rm --force`
   (registry.py:631-657).
3. **There are six teardown/loss sites, not one**: idle reaper; mount-snapshot drift
   (`release_if_mounts_changed`); `spec_version` drift recycle inside `get_or_provision`
   (Postgres trigger from migration 0077 bumps it on memory-store / github-repo
   attach/detach/token-rotate); `evict()` on tool failure; `release_all()` at worker
   shutdown; `reap_orphans()` at worker boot. Persistence must hook all of them.
4. **Host topology in the code today**: the worker is a Postgres advisory-lock-enforced
   **singleton per database** (worker.py:301) — one worker, one Docker daemon (DooD
   socket mount), one host. Horizontal scale (a stated requirement) therefore takes one
   of two shapes (§9.3): **shard-by-DB**, which preserves the singleton invariant per
   shard, or an **elastic worker pool**, which lifts the advisory lock while
   procrastinate's per-session `lock` — a DB-level lock — continues to serialize steps
   across workers. Either way, a resuming worker must be able to *find* a snapshot that
   another daemon produced, which is what forces the DB snapshot pointer (§5.1).

Also verified and load-bearing: prod has **no working `--storage-opt` quota** (the flag
requires overlay2-on-xfs+pquota; `AIOS_SANDBOX_DISK_BYTES` defaults to None and would be
rejected at `docker run` on this daemon), so any quota story must work without it.

## 3. The persistence contract (validated, two amendments)

**Persists** (via the snapshot image): the entire writable root FS — `/root`, `/etc`,
`/usr`, `/var`, `/opt`, `/home`, **`/tmp`** (a normal overlay dir, not tmpfs), global
apt/pip/npm installs, and deletions of base-image files (`docker commit` captures
overlayfs whiteouts — verified).

**Persists** (via host bind mounts, exactly as today, *not* via the snapshot):
`/workspace`, `/mnt/attachments`, `/mnt/uploads`, `/mnt/memory/*`, github working trees.
`docker commit` excludes bind-mount contents (verified), so there is no duplication and no
interaction between snapshot GC and workspace data.

**Does not persist**: processes, fds, locks, sockets, netns/iptables state, `/dev/shm`
(tmpfs), the container IP. Resume = fresh process tree on the persisted disk.

This contract is the natural semantics of disk persistence — a fresh boot on a persisted
disk — and the strongest version of it aios can honestly promise. **Process persistence
is rejected decisively**: Docker checkpoint/CRIU is experimental, bridge-network restore
is structurally broken since Docker 28 (moby#50750, open), and the `criu` package is
absent from Ubuntu 24.04 LTS entirely. Even a hypothetical working process snapshot
cannot preserve live TCP connections or other kernel-held resources, so the marginal
value over disk-only does not justify the machinery.

Two deliberate amendments to the proposed contract, both strengths:

- **Environment variables do not persist.** Resume re-derives env from the *current*
  spec. The committed image config is actively scrubbed (§5.2) so a secret removed from
  `env_config.env` cannot resurrect inside a resumed container.
- **Security posture does not persist.** seccomp profile, no-new-privileges, resource
  caps, and the network lockdown are re-applied from *current* settings at every resume.
  A session parked across a security-hardening deploy resumes with the new flags. (This
  is the structural advantage over mechanism A's frozen stopped containers.)

**Two containerd-image-store amendments (verified on the production store, which OVERRIDE
the overlay2-verified §5.2/§5.6 pseudocode and are what the implementation ships):**

- **Skip-empty threshold is a byte floor, not `== 0`.** A no-write container reports
  `SizeRw == 4096`, not 0, on the containerd image store. The identity short-circuit fires
  on `SizeRw <= sandbox_snapshot_empty_floor_bytes` (default 8 KiB), so read/chat-only
  sessions never grow a chain (§5.7's premise). An `== 0` test would never fire in prod.
- **Flatten is budget-driven; the layer wall does not exist.** The overlay2 ~125-layer
  commit wall is absent on the containerd store (a chain ran cleanly through 250 layers).
  Flatten is therefore driven by the per-session unique-bytes budget (storage), with layer
  depth as a *soft* performance guard at a generous ceiling (`_FLATTEN_DEPTH_CEILING = 200`,
  below the kernel overlayfs lower-layer max), not a hard-wall dodge. The rest of the
  containerd verification battery (label inheritance through commit, `rmi` parent-chain
  cascade, `images -a` residue visibility, `export|import` flatten + config strip, the
  `prune --filter 'label!='` exemption) confirmed the design unchanged.

## 4. Mechanism decision

**Preserve the disposable-instance model; persist the filesystem as an image.** The whole
codebase is built around containers being recreatable at any moment (six teardown sites,
spec rebuilt fresh each provision, secrets re-minted, lockdown re-applied). Mechanism B
is the only option where resume *is* the existing provision path — every drift problem
(env, mounts, secrets, seccomp, broker URL, iptables) dissolves structurally because
resume is a cold start that happens to run from a persisted rootfs.

| | **B: disposable + commit (chosen)** | A: durable stopped container | C: overlay-upper / podman `--rootfs` |
|---|---|---|---|
| Resume path | existing provision, unchanged shape | new `docker start` path + adoption logic | existing shape, new runtime |
| Config/mount/secret drift while idle | dissolves (fresh `docker run` from snapshot) | **fatal flaw**: env/mounts/seccomp frozen at create; broker secret 401s; needs commit machinery anyway as drift escape hatch → two resume paths | dissolves (artifact carries zero config) |
| `spec_version` recycle (a today-feature) | FS preserved | **loses entire FS** (pure A) or imports B's machinery (hybrid) | FS preserved |
| Snapshot cost per idle cycle | 0.3–15 s commit, delta-sized (fresh container ⇒ empty layer ⇒ per-cycle delta) | ~0 (stop) | ~0 (structural) |
| Failure surface | lineage gate + salvage + flatten + image GC + pointer/store reconciliation | smallest | smallest (no snapshot step exists) |
| Layer/chain management | flatten on the unique-bytes budget (depth ≥ 200 soft guard — §3); flattened image duplicates base (~466 MB) | none | none |
| Quota on prod ext4 | commit-time accounting (`SizeRw`, image sizes) | `SizeRw` scan | **best**: per-session loopback ext4 = kernel ENOSPC at write time |
| Deployment change | none beyond the §9 prune-cron/budget lockstep (shard shape); elastic pool adds a registry/object store (§9.3) | same as B | privileged worker + podman/netavark/crun in the worker image; macOS dev can't run it (two backends, flagship behavior untestable locally) |
| Security posture refresh at resume | **yes** (current flags) | no (frozen) | yes |
| Future fork/replay of sandboxes | natural (immutable image artifacts) | requires committing anyway | natural (CoW dirs) |
| Verified mechanics | live-verified (on overlay2 — §9 caveat) | live-verified | documented, unverified spikes needed |

A's steelman conceded that the hybrid needed to match B's preservation guarantees is
*more* total mechanism than B alone. C's steelman conceded that at aios's per-host
scale (one daemon per host/shard, tens-to-hundreds of sessions each) the failure surface
C deletes is mostly surface aios won't hit, while its deployment cost is concrete and
immediate; C's case only wins if write-time disk quotas get promoted to a hard
requirement. No researched platform ships commit-based per-idle persistence at scale —
but the published cautionary tale against whole-filesystem serialization (O(size) per
*change*, at hyperscale) doesn't transfer to per-*idle*, delta-sized commits at this
deployment's scale, and unlike every alternative, B needs no new container backend and
no privileged deployment change — its ops footprint is the §9 prune-cron/budget
lockstep.

Horizontal scale is a stated requirement and is carried *within* B: snapshot transport
sits behind a `SnapshotStore` seam (§5.2) — local-daemon in v1, registry/object-store as
a drop-in — and snapshot location lives in the DB (§5.1), so a second worker or a host
replacement is configuration plus a store implementation, not redesign (§5.11). If aios
later needs write-time disk quotas or sandbox forking as a product feature, the
C/microVM per-session-rootfs-artifact remains the successor design for the *backend*;
the store seam and the DB pointer carry over unchanged.

## 5. The design

### 5.1 Identity and state: deterministic tag, image labels, DB snapshot pointer

**Tag**: `aios-sbx-{instance_id}-{session_id.lower()}:latest`. ULID lowercasing is
bijective; the single-path-component form means `_is_registry_image()` returns False, so
the existing `--pull always` logic never touches the registry for snapshots (verified
against docker.py:352-377). `instance_id` isolates concurrent dev worktrees on one daemon.

**State splits: DB = location + existence; store/daemon = content + lineage.**
Single-host, the daemon alone was a defensible source of truth — a DB column would have
been a cache with no consumer able to trust it over the local probe (the original draft's
position). With horizontal scale a requirement, the column gains the consumer no daemon
probe can replace: *where does this session's snapshot live?* A worker resuming a session
whose snapshot another daemon produced is blind without it. Sessions carry a pointer:

```
snapshot_ref         text         -- artifact identity: a deterministic, host-independent name
                                  -- (the local tag string in v1; each store maps it into its
                                  -- own namespace, §5.2); NULL = none
snapshot_host        text         -- owning worker/daemon/host id; NULL = none
snapshot_bytes       bigint       -- last-known unique size; reporting only, never an
                                  -- enforcement input (§5.7)
snapshot_updated_at  timestamptz
```

**Drift discipline** (the answer to "a cache that lies in every failure mode"): the
pointer is written **inside the snapshot critical section — after commit success, before
`rm` — under the per-session lock**, so it never claims a snapshot that wasn't committed.
The GC reconciler (§5.5) is the convergence authority: each tick re-derives truth from
its own store and corrects the pointer (commit-succeeded/DB-write-failed crash residue,
operator `rmi`, manual mess). The pointer is a reconciled routing hint, never blindly
trusted — but unlike the rejected single-host cache, removing it would leave multi-host
resume with no way to function at all. Content truth — the lineage gate, labels, sizes —
stays local to the owning daemon, unchanged. The worker-side daemon-enumerating sweep is
still structurally required (session delete runs in the API process, which has no Docker
socket); the pointer doesn't replace it, it routes around its locality.

`snapshot_host` is `settings.instance_id` in v1 (one worker, so trivially unique).
Multi-worker shapes need a per-worker host-id setting **distinct from `instance_id`**:
the tag/ref namespace must stay deployment-stable while host identity varies — if each
worker minted refs from its own id, every cross-host handoff would change the session's
ref and reopen the first-commit crash window (§5.3). Invariant (§5.11): `snapshot_ref`
is a pure function of (deployment, session), never of which worker committed. The four
columns are internal — not exposed on the session wire shape, the same stance the dead
`container_id` had.

**Labels** (all stamped at `docker run`, inherited into committed images automatically —
verified):

- `aios.managed=true`, `aios.instance_id`, `aios.session_id` — exist today.
- `aios.env_keys=<comma-separated names>` — **new**: the names (never values) of every
  run-injected env var. Makes any corpse self-describing for the commit-time ENV scrub
  (§5.2) without any DB lookup, including corpses salvaged after a crash when the
  original spec is gone.
- `aios.base_image=<resolved spec.image ref>` — **new**: which base this session's chain
  is rooted on. Drives both base-drift detection (§5.3) and accurate accounting (§5.7).
- `aios.flattened=true` — **new**, applied via `--change` at import only: marks images
  that no longer share layers with the base, so accounting charges them full size (§5.7).

No `aios.account_id` label: it would let any daemon-local reader enumerate the
tenant graph. The GC's existing batch DB query maps session→account where needed.

### 5.2 The snapshot operation

Drop `--rm` from `DockerBackend.create`. One new backend verb performs:

```
docker stop -t 5 <id>                          # idempotent on stopped corpses
docker inspect <id>          → .Image (parent), .Config.Env, labels
docker image inspect <tag>   → .Id, layer depth, .Size      # absent → first snapshot
LINEAGE GATE: proceed iff tag absent OR tag.Id == corpse.Image   # else skipped_stale
docker inspect --size <id>   → SizeRw
SizeRw <= empty_floor → skipped_empty          # identity short-circuit (containerd no-write == 4096; §3)
flatten? (unique-bytes over per-session budget [primary], or depth+1 ≥ 200 [soft guard] — §3)
  COMMIT:  docker commit --change 'ENV K='  (per key in aios.env_keys ONLY)  <id> <tag>
  FLATTEN: docker export <id> | docker import
             --change 'CMD ["tail","-f","/dev/null"]'
             --change 'WORKDIR /workspace'
             --change 'ENV HOME=/home/aios'
             --change 'LABEL …'  (managed/instance/session/env_keys/base_image/flattened)
             - <tag>
```

The caller then runs the existing `force_remove`. **Ordering invariant: the container is
removed only after the snapshot verb succeeds.** On failure the stopped corpse is
retained and the error goes to the ops log; convergence is the salvage rule (§5.4).

Adversarial-review corrections folded in (three of these were verified failures in the
original spec):

- **ENV scrub scope** (was a verified container-bricker): scrub *exactly* the
  `aios.env_keys` set. Scrubbing all of `.Config.Env` empties `PATH`/`HOME` inherited
  from the base image config, and Docker does not re-inject a default `PATH` when the
  key is present-but-empty — the snapshot's CMD then fails `exec` lookup and **every
  resumed container fails to start** (reproduced live). Key-set diffing against the
  parent image is also wrong: at generation ≥2 the parent already carries scrubbed keys
  as `K=`, so a diff would skip re-scrubbing and leak fresh secret values. The label is
  the only sound source. Known wart: `ENV K=` empties rather than unsets; removed vars
  read as empty strings until the next flatten strips config entirely.
- **No `rmi` inside the flatten path** (was a deterministic failure): the corpse being
  flattened was created *from* the old chain's image, and a container — even stopped —
  blocks `rmi`, so an in-verb cleanup is refused 100% of the time and, under the
  corpse-retained-on-failure rule, would wedge the session in a permanent release/salvage
  loop. Snapshot success = new tag exists, nothing else; the orphaned old chain is
  collected by the GC retain rule on the next tick, after the corpse is gone. One
  reconciler, not two.
- **Size-derived timeout** (was a permanent-brick path): a fixed 300 s commit timeout
  turns a ~110 GB writable layer into an infinite retry loop (commit times out → corpse
  retained → salvage times out → provision fails → forever). Timeout = constant floor +
  per-byte budget at ~10× measured throughput, derived from the `SizeRw` already read in
  the sequence — fires only on a genuinely hung daemon, never on size.
- **Restore `WORKDIR` and `HOME` at flatten** (was a silent behavior shift): import
  strips all config; without these, post-flatten sessions exec with cwd=`/` and
  `HOME=/root`. `PATH` is deliberately *not* restored so Docker injects its default.
- **`skipped_stale` is for content-equal crash residue only**: under the salvage-
  before-provision invariant, the only reachable corpse-with-newer-tag states are
  crash-between-commit-and-rm and crash-between-import-and-cleanup, both content-equal to
  the tag. The gate never discards live data; assert that as a unit-tested invariant.
- **Skip-commit-if-`SizeRw==0`** is included as an identity short-circuit, not
  belt-and-suspenders: a zero-byte layer produces content identical to the existing tag.
  It is what keeps chat-only and read-only sessions from ever growing a chain, and what
  makes deploy-time salvage commits free for unchanged sessions.

Protocol additions (`backends/base.py`): `snapshot(sandbox_id, tag, *,
flatten_if_unique_bytes_over) -> SnapshotOutcome` (the depth wall is backend-internal),
`stop(sandbox_id)`, `list_managed_images(instance_id)`, `remove_image(ref) -> bool`
(rmi-no-force; False when refused), `image_size(image)`. `SandboxSpec` and
`SandboxHandle` gain `snapshot_image: str | None`; the handle gains
`disk_limit_bytes: int | None`. `ManagedSandboxRef` gains `running: bool`
(`list_managed` switches to `docker ps --all`).

**`SnapshotStore` seam** (`sandbox/snapshot_store.py`) — transport is pluggable; the
lifecycle/salvage/lineage/flatten/GC-classify logic above never sees it:

```python
class SnapshotStore(Protocol):
    async def put(self, local_image_tag: str, ref: str) -> str: ...  # persist a just-committed local image; returns stored ref
    async def get(self, ref: str) -> str: ...                        # make ref locally runnable (pull/load if remote); returns local tag
    async def exists(self, ref: str) -> bool: ...                    # verified-negative semantics; indeterminate raises
    async def remove(self, ref: str) -> bool: ...
    async def size(self, ref: str) -> int: ...
```

v1 **`LocalDaemonStore`**: `put`/`get` are identity (the image is already local),
`exists` = `docker image inspect` under the §5.3 verified-negative rule, `remove` =
`rmi`, `size` = image size — exactly today's behavior, now behind the seam. A future
`RegistryStore`/`ObjectStore` (`put` = push or `save | upload`, `get` = pull or
`download | load`) drops in with no lifecycle changes.

**Refs are host-independent names**, a pure function of (deployment, session); each
store maps the name into its own namespace deterministically (`LocalDaemonStore`: the
name *is* the local tag; a `RegistryStore` prefixes its configured registry path). The
pointer therefore never needs rewriting after a push, and the shard+async-push
configuration is a *hybrid* store: `get` prefers local, falls back to the remote
namespace. Digest-pinned refs are possible only under the elastic pointer-after-`put`
ordering (the digest doesn't exist when a pointer-before-push write happens).

Two boundaries attach to the snapshot verb without changing its internals: the **DB
pointer write** (`snapshot_ref/host/bytes/updated_at`) happens after the verb succeeds
and before `force_remove`, in the same critical section — and a **failed pointer write
is treated identically to a failed snapshot verb**: corpse retained, no `rm`, ops-log
error, convergence via salvage. When durable transport is enabled (§9.3), an **async
`store.put` is enqueued after the pointer write** — never on the release critical path.
In the elastic-pool shape the pointer is instead written only after `put` returns, so it
never routes a peer to an artifact it cannot fetch; `put` then sits inside the critical
section and obeys the same size-derived timeout rule as commit/flatten.

### 5.3 Resume

Resolution runs through the store: `sessions.snapshot_ref` set → `store.get(ref)` → run
from the returned local tag; else the base path (env override → `settings.docker_image`)
with today's pull logic. With `LocalDaemonStore`, `get` is an inspect of the
deterministic local tag — today's behavior exactly. Rules:

- **Verified-negative only, extended to the store** (was a silent double-data-loss path):
  only a *verified* not-found from `get`/`exists` selects the base path. Any
  indeterminate result (daemon or registry hiccup, timeout) raises and fails the
  provision — treating an indeterminate probe as absence would silently cold-start the
  session and then, at the next idle, the lineage gate would discard all post-hiccup
  work as `skipped_stale`.
- **Snapshot-missing is detected, not silent**: pointer set + store verified-not-found
  can only mean external mutation (operator `rmi`, image-store loss, host replacement
  without transport). The provision clears the pointer, appends a model-visible
  `sandbox_fs_reset {reason: "snapshot_missing"}`, and cold-starts from base. (This was
  the original draft's one accepted silent case; the pointer makes detection free, so
  silence is no longer defensible.)
- **Base-image drift is detected, never silent** (was a textbook silent fallback): if the
  snapshot resolves but its `aios.base_image` label ≠ the currently resolved
  `spec.image` *ref*, the operator deliberately redefined the environment's image. The
  snapshot is discarded — **`store.remove(ref)` + pointer cleared, in the same provision
  step** (the artifact must actually be gone: a surviving tag would be re-pointered by
  GC pass 4, and the next idle's lineage gate would see a corpse rooted on the new base
  against the old tag and discard live post-drift work as `skipped_stale`) — a
  model-visible `sandbox_fs_reset {reason: "environment_image_changed"}` lifecycle event
  is appended, and the session cold-starts on the new image. (Content changes under the *same* ref — base rebuilds, `:stable`
  promotes — do not trigger this; parked chains keep their pinned base until the session is
  reset or archived.) Keep-the-FS-instead is a defensible alternative semantic; the event is the
  non-negotiable part — sign-off item §11.
- **First-commit crash heal**: the salvage preamble (§5.4), already under the per-session
  lock, also reconciles the pointer against local truth — a crash after the first-ever
  commit but before the pointer write leaves `snapshot_ref` NULL while the local
  canonical tag exists; the preamble sets the pointer before resolution runs, closing
  the only window where resolution could miss a local snapshot. (Later commits rewrite
  the same ref value in v1, so the window is first-commit-only; the GC tick covers the
  same case out-of-band.)

After resolution, the provision path runs **unchanged**: fresh env from current config,
fresh broker secret, fresh netns, lockdown re-applied (§5.8), `install_packages`
reconciling current package config idempotently onto the persisted FS.

### 5.4 Lifecycle: all six sites

Two registry helpers under the existing per-session `asyncio.Lock`:
`_snapshot_and_remove(...)` (§5.2 sequence) and `_salvage_session_corpses(session_id)`
(list this session's containers → stop if running-and-uncached → lineage-gated snapshot →
remove).

| Site | Today | New behavior |
|---|---|---|
| Idle reap | `docker rm --force` | snapshot → rm. The main snapshot point. |
| Mount-drift release | destroy | snapshot → rm; resume rebuilds current mounts on the persisted FS. |
| `spec_version` recycle | destroy mid-step | snapshot → rm; the immediately following provision resolves the just-written tag — an FS-preserving reboot. |
| `evict()` (tool failure) | drop cache (`--rm` self-removes) | **code unchanged**; the corpse now persists and is salvaged by the session's next provision or the GC tick, whichever first. |
| **Provision preamble (new)** | — | `_provision` starts with `_salvage_session_corpses` under the lock, then reconciles the snapshot pointer against local truth (§5.3). **Container death no longer loses data.** Salvage failure fails the provision — raw error into the tool result, model-actionable; never a silent resume from the stale tag. |
| Worker shutdown | destroy all | `stop_all()`: parallel `docker stop -t 5` under one overall `wait_for` (~8 s) so a hung daemon can't eat the SIGTERM grace; no commits (unbounded). Per-container failures are ops-log noise — the boot tick converges regardless. |
| Worker boot | `reap_orphans` removes all | **deleted**, replaced by the GC sweep's immediate first tick as a background task. Boot is not blocked; a session waking mid-reconcile salvages its own corpse inline under its own lock. |

**Worker crash**: containers keep running; the advisory lock frees when the dead worker's
PG connection drops; the returning worker's first GC tick stops + salvages.
(Singleton/shard shapes. In an elastic pool the crashed worker's host retains its corpses
until *its own* worker returns — peers cannot reach that daemon, which is the §9.3
async-push motivation.) **Daemon restart/host reboot**: running → stopped corpses
(restart policy stays `no`) → salvaged. The worker is a container on the same daemon, so
"daemon restarted but worker didn't" is structurally impossible (live-restore=false,
verified).

**Concurrency** (one container ever runs per session): three locks at three scopes — the
DB advisory lock (one worker per database; singleton/shard shapes — lifted in the
elastic pool, where procrastinate's DB-level per-session lock carries cross-worker *step*
exclusion but not host co-location, see §8/§9.3), the procrastinate per-session lock
(one step), and the registry per-session asyncio lock (provision / release / salvage /
GC-per-corpse mutually exclusive within the worker). The lineage gate makes the residual
crash windows content-equal no-ops.

### 5.5 GC: one retain-rule reconciler

`start_gc(pool, ...)` — background loop, hourly tick, immediate first tick at boot:

1. **Corpse pass** — per corpse, under `_lock_for(session_id)`: re-read the
   session lifecycle. Deleted and archived sessions are bare-destroyed immediately;
   non-archived session corpses are salvaged before removal. Ephemeral workflow-run
   corpses are
   always bare-destroyed because they have no durable rootfs.
2. **Image pass** — enumerate all managed images, including untagged residue.
   Canonical images for existing non-archived sessions are protected. Archived
   canonical images, deleted-session images, and non-canonical leaf residue are
   collectible immediately.
   Live-chain interiors are skipped structurally by parent relationship.
3. **Pressure passes** — host-pool and per-account snapshot limits are
   observational: they never delete lifecycle-protected filesystems. Pressure
   is fed back to cold-provision admission. Host pressure is global; account
   pressure rejects only durable session provisions owned by that account, and
   never ephemeral run sandboxes. A later clear report restores admission.
4. **Pointer reconciliation** — local store truth heals missing/stale pointers.
   Destructive archive cleanup requires a fresh, exact archive timestamp,
   matching pointer and positive ownership
   (`snapshot_host == instance_id`) under the per-session lock; an unarchive,
   rearchive, pointer move, or ambiguous host fails closed.

**Every per-session image removal takes `_lock_for(session_id)` and re-checks
lifecycle and ownership under the lock.** This closes the scan-to-delete race
without using activity age as a deletion condition.

**Ownership across hosts**: each host GCs only its own store. For a session archived
past grace whose `snapshot_host` is another host, the non-owning tick **skips** — the DB
pointer tells it not to reach across; the owning host's tick removes. On multi-host
shapes the retain rule gains an ownership clause: a local image whose session's
`snapshot_host` is elsewhere is a transport cache, reclaimable once unreferenced, never
the canonical copy. With a `RegistryStore`, registry-artifact GC is keyed on the DB
pointers (the registry has no session knowledge) — run by a **single designated
ticker** (config), with an age grace period exceeding the put→pointer window so a
just-pushed artifact is never collected before its pointer lands. A **permanently
decommissioned host** is an explicit operator action: clear `sessions.snapshot_*` for
that `snapshot_host`, which converts future resumes into detected resets and fixes
global accounting — without it, never-waking sessions' pointers (and their
`snapshot_bytes`) leak indefinitely, since non-owning ticks skip by design. All of this
is vacuous single-host.

**Every per-session image removal takes `_lock_for(session_id)` and re-checks
under the lock** (cached handle present → skip; lifecycle and ownership re-verified).
Without this, archive cleanup can race an unarchive or rearchive between the scan and
`rmi`, deleting the canonical snapshot for a newly protected session.

`docker rmi` of a tag cascade-deletes the entire untagged parent chain down to the first
still-referenced image (verified on overlay2; re-verify on containerd, §9) — GC of an
arbitrarily long session history is one command. Each tick logs a one-line summary
(images retained/removed, bytes, top sessions by usage) so "why is the disk full" is
answerable from logs.

### 5.6 Flatten policy

Each nonzero idle cycle adds one layer. On the overlay2 graphdriver commits hard-fail at
a ~125-layer wall; the production **containerd image store has no such wall** (a chain ran
cleanly through 250 layers — §3), so flatten is driven primarily by the per-session
**unique-bytes budget**, with layer depth as a *soft* performance guard. Flatten
(`export | import`) when unique bytes exceed the per-session budget, or when `depth+1 ≥
200` (a generous backstop below the kernel overlayfs lower-layer max) — flatten applies
whiteouts, so it is also what makes "delete files to
shrink" actually work, and it strips baked config entirely (the definitive secret-residue
scrub). Cost honesty: a flattened image is a standalone rootfs that stops sharing the
466 MB base — acceptable at this scale, and rare once the idle-TTL change (§5.10) cuts
cycle counts by an order of magnitude.

### 5.7 Quotas and accounting (must work on ext4)

- **Delete the `--storage-opt` emission entirely** and **delete `sandbox_disk_bytes`**,
  introducing `sandbox_snapshot_budget_bytes` (per-session, finite default ~4 GB; env
  override keeps working through `EnvironmentConfig`). Silently repurposing a deployed
  env var's semantics is the config-surface version of the shim "don't deprecate,
  delete" forbids.
- **Metric**: unique bytes = `tag.Size − base.Size` where base is the image named by the
  chain's own `aios.base_image` label (not `settings.docker_image` — per-env overrides
  would otherwise be billed ~1.5 GB they never wrote, triggering wrongful pressure reports and admission blocks);
  `aios.flattened=true` images charge full `.Size` (they share nothing — subtracting the
  base would hide ~466 MB per flattened session from the very accounting that must see
  the host filling). Computed at read time from the owning daemon, which stays
  authoritative for enforcement; the figure is also written to `sessions.snapshot_bytes`
  at each commit so cross-host reporting needs no daemon round-trips.
  **`snapshot_bytes` is reporting-only — never an enforcement input**: enforcement
  always derives from the owning store's live enumeration (the column can be one
  generation stale in a crash window; a future bytes-based destructive cleanup reading it would
  silently break this).
- **Per-session enforcement**: commit-and-flag, never refuse (refusal destroys the
  agent's work as punishment for a state it wasn't awake to prevent). Over budget →
  flatten instead of commit; if still over, append `sandbox_fs_over_limit` —
  **edge-triggered** on the crossing (previous size ≤ limit < new size), not re-fired
  every cycle.
- **Pool budgets** (`sandbox_snapshot_pool_bytes`, operator-set): the load-bearing
  bound. Required in v1 because the prune-cron exemption (§9) removes the only existing
  disk control; on a 75 GB host at 76% with a documented full-disk → Postgres-PANIC →
  11-app-cascade incident, shipping unbounded retention is not an option. Enforced
  **per-host** by GC pass 3 from local enumeration; the **global** cross-host total is
  computed by summing `sessions.snapshot_bytes` pointers. v1 enforces per-host and
  reports global.
- **Per-account caps** (`accounts.config.sandbox_snapshot_bytes`): **deferred to a
  follow-up.** A destructive quota engine is the most intricate code in the design and would
  ship dead (no account will have a cap set); v1 ships per-account *usage* as a read-time
  ops metric only. (Ethos review; consistent with the no-belt-and-suspenders rule.)
- **Stated residual**: nothing bounds the *live* writable layer between commits on ext4 —
  prod's status quo today. Host disk monitoring (plus the eumemic-ops audit extension,
  §9) is the mitigation; a write-time quota requires xfs+pquota or mechanism C.

### 5.8 Security

**Network lockdown moves off the tenant-writable filesystem (blocker, and a pre-existing
hole).** Today `apply_network_lockdown` runs `bash`/`iptables`/`getent` *inside* the
sandbox via `docker exec` — i.e., from the container's own root FS — and grants the
sandbox `--cap-add NET_ADMIN`. Under persistence this is a working bypass: a tenant
replaces `/usr/sbin/iptables` with `exit 0` in an Unrestricted session, the snapshot
persists it, and when the environment later flips to Limited the fail-closed gate
trusts the poisoned binary's exit 0 — open egress, no audit artifact. (It is *already* a
hole today in milder form: a root agent holding NET_ADMIN can simply `iptables -F` its
own lockdown.) Fix, using only the docker CLI: apply the lockdown from an **ephemeral
sidecar** — `docker run --rm --cap-add NET_ADMIN --network container:<sandbox>
<settings.docker_image> bash -c <script>` — which joins the sandbox's netns but executes
the *operator-trusted* image's binaries (never `env_config.image`, which is
tenant-controlled), then verifies `-P OUTPUT DROP` through the same sidecar. The sandbox
itself **loses `--cap-add NET_ADMIN` entirely**: root-in-sandbox can no longer modify
netfilter at all. This closes both the persisted-FS bypass and the pre-existing
flush-your-own-lockdown hole, and hardens the Limited profile beyond today's.

**Cross-tenant snapshot isolation (blocker).** `spec.snapshot_image` is derived only from
the session row the worker is stepping. But `env_config.image` is a free-form
tenant-writable string, snapshot tags are single-component (no pull → resolved locally),
and `instance_id` is `default` in prod — so `image: aios-sbx-default-<victim-ulid>:latest`
would mount another session's entire root FS and its baked secrets. ULIDs are not
secrets (they appear in URLs, logs, events). **Gate**: reject any `env_config.image`
matching the reserved `aios-sbx-` prefix (case-insensitive) in `build_spec_from_session`
— the choke point all writers traverse, including direct SQL — **plus** a
`field_validator` on `EnvironmentConfig.image` returning a 422 at the API boundary. The
two layers cover different writer sets (deliberate, flagged: the validator is UX +
window-shrinking; the spec gate is the boundary).

**Secrets.** Run-injected env (which can include operator API keys in `env_config.env`)
is scrubbed from image config via the `aios.env_keys` label (§5.2); flatten strips config
entirely. What remains: secrets the agent itself wrote to disk (`~/.netrc`,
`~/.aws/credentials`) are in the snapshot *filesystem* — the same exposure class as
`/workspace` on host disk today, but now also readable by anything with daemon access
(`docker save`, image inspect of FS layers). **Accepted threat-model boundary requiring
sign-off**: daemon access ≈ host root, already a near-total compromise; the marginal
regression is retention duration, bounded by explicit archive lifecycle and grace.

**Deferred hardening, dispositioned rather than re-deferred**: `--read-only` is
**permanently incompatible** with this contract (the contract *is* a writable persistent
root) — close it as superseded, don't re-defer it. `--cap-drop=ALL` stays deferred
(apt/dpkg interaction unchanged by this design), but this design already removes
NET_ADMIN, the largest grant. The resume-fresh-flags property means any future hardening
retrofits every parked session automatically.

**Supply-chain accumulation**: agent-installed software re-executes at every resume —
inherent to persistent disk. Bounds: archive lifecycle/grace, pressure alarms,
fresh security flags each resume, and the sidecar fix removing the worst consequence
(lockdown subversion). `install_packages` still runs tenant-FS package managers at
resume; with the lockdown gate moved off the sandbox FS, a poisoned pip/apt harms only
the tenant's own sandbox — unchanged from the agent running them directly.

### 5.9 Agent observability

`build_messages` skips all non-`message` events today, so loss events need a render path:
a minimal allowlist in context.py (`sandbox_fs_expired` for legacy records, `sandbox_fs_over_limit`,
`sandbox_fs_reset`) rendered as bracketed user-role notices at their seq position —
append-only in, append-only out (monotonicity holds), **not** stimulus-bearing
(`find_sessions_needing_inference` ignores them, so a GC append never wakes a session or
costs a model call; the notice is read at the next genuine wake). Event text uses runtime
vocabulary only, e.g.: *"The persisted sandbox filesystem for this session was discarded
(retention limit). The next command runs on a fresh base filesystem; /workspace and
mounted directories are unaffected."*

Saves are silent in the session log (model-consciousness: loss is actionable, routine
saves are noise). The existing `sandbox_provision_*` span gains the resolved image ref +
chain depth + size, so "why did this session cold-start" is answerable after the fact
(consecutive spans flipping snapshot→base expose even operator-caused wipes).

The original draft accepted one silent case — operator `rmi`, host migration — because
zero DB state made "never snapshotted" and "externally removed" indistinguishable. The
snapshot pointer (§5.1) reverses that: external loss is now detected at resume (pointer
set, store verified-not-found) and surfaced as `sandbox_fs_reset {reason:
"snapshot_missing"}` (§5.3). Every FS-loss path the model can encounter is now evented.

### 5.10 Idle-TTL economics (the constant this design inverts)

`container_idle_timeout_seconds=300` was tuned when teardown was a free `docker rm`.
Under B, teardown costs a commit, a layer, and eventual flatten — keeping an idle
container alive is now the *cheap* option (a parked `tail -f` container is ~1 MB RSS).
At 300 s, a daily-driver session commits 20–80×/day and a 15-minute cron-trigger session
~96×/day, hitting the flatten wall in days and turning the design's cost model into
fiction. **Raise the default to 1800 s (30 min) in the same change**, with the rationale
in the field description. A 15-min cron session then never idles out at all (zero
commits until it stops); a conversational session commits a handful of times a day.
Re-run the §5.6/§5.7 arithmetic at both values in the PR description.

### 5.11 Cross-host invariants (hold even when deployed single-host)

1. A session's snapshot is addressable by `(snapshot_ref, snapshot_host)` from the DB
   alone — no daemon enumeration required to *find* it.
2. Resume on a non-owning host goes through `store.get(ref)` or fails loud — never a
   silent cold start on an indeterminate probe.
3. GC removes only artifacts its host owns; local images of sessions owned elsewhere are
   transport caches, and a non-owning tick skips rather than reaches across.
4. The DB pointer is written after commit success and before `rm`, under the per-session
   lock, and is reconciled by GC against store truth every tick.
5. Lineage/content truth (the lineage gate, labels, sizes) stays local to the owning
   daemon; the DB holds location and last-known size, never content claims —
   `snapshot_bytes` is reporting-only, never an enforcement input.
6. Pointer writes and clears are ownership-gated (only the `snapshot_host` owner — or
   the NULL-heal — touches `sessions.snapshot_*`), and multi-host pointer advancement is
   compare-and-swap, so a moved-on session is never clobbered by a stale host.
7. `snapshot_ref` is a pure function of (deployment, session_id) — never of which worker
   performed the commit. Host identity lives only in `snapshot_host`.

A pointer whose owning host no longer exists converges only via operator decommission or
session wake — a stated leak, not a discovered one (§5.5).

A single-host v1 with `LocalDaemonStore` satisfies all seven trivially — which is the
point: nothing in the lifecycle assumes one host. The **shard-by-DB** rollout is
configuration plus a store implementation; the **elastic pool** additionally requires
the §9.3 prerequisites (wake→host affinity while a session has a live container or
uncommitted state) — stated there as a named follow-on, not hand-waved here.

## 6. Schema and settings

**Migration 0080** (raw `op.execute`, repo style; the entire migration):

```python
def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE sessions
            ADD COLUMN snapshot_ref        text,
            ADD COLUMN snapshot_host       text,
            ADD COLUMN snapshot_bytes      bigint,
            ADD COLUMN snapshot_updated_at timestamptz;
        """
    )
    op.execute("ALTER TABLE sessions DROP COLUMN container_id;")


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE sessions
            DROP COLUMN snapshot_ref,
            DROP COLUMN snapshot_host,
            DROP COLUMN snapshot_bytes,
            DROP COLUMN snapshot_updated_at;
        """
    )
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS container_id text;")
```

All four new columns are nullable with NULL = no snapshot — zero backfill, metadata-only
DDL. They are internal (excluded from the session wire shape), so no openapi/SDK churn
from this migration itself.

**Settings** (config.py): `sandbox_snapshot_budget_bytes: int = 4 GiB` (per-session; per-env override via a **new**
`EnvironmentConfig` field replacing `disk_bytes` — never a silent semantic rename of the
existing field, per §5.7's own no-repurposing rule); `sandbox_snapshot_pool_bytes: int | None = None` with the eumemic-ops
deploy setting it (§9); **delete** `sandbox_disk_bytes`; raise
`container_idle_timeout_seconds` default to 1800. The snapshot store implementation is
selected in code (`LocalDaemonStore` in v1 — a settings knob arrives with the second
store implementation, not before). `EnvironmentConfig` model changes flow
through FastAPI introspection → run `./scripts/regen-openapi.sh && ./scripts/regen-client.sh`.

## 7. Code touchpoints

- `sandbox/backends/base.py` — `SnapshotOutcome`, new label-key constants, Protocol +=
  `snapshot/stop/list_managed_images/remove_image/image_size`; `SandboxSpec/Handle +=
  snapshot_image`, handle `disk_limit_bytes`; `ManagedSandboxRef += running`.
- `sandbox/backends/docker.py` — drop `--rm`; delete `--storage-opt` block; remove
  `--cap-add NET_ADMIN`; `create()` runs from the locally-resolved tag (resolution
  itself lives in the provision path, below); the five new verbs (lineage gate, env-keys
  scrub, skip-empty, flatten with full config restore, size-derived timeouts) plus a
  label-inspect primitive for the drift check; `list_managed --all`.
- `sandbox/_subprocess.py` — `run_docker_pipeline` (export|import over an `os.pipe()` fd
  pair; no host shell).
- `sandbox/snapshot_store.py` — **new**: `SnapshotStore` protocol + `LocalDaemonStore`
  (§5.2); the only module a future `RegistryStore` touches.
- `sandbox/setup.py` — `apply_network_lockdown` rewritten to the sidecar invocation
  (new backend verb or a dedicated `run_in_netns_sidecar` helper); same fail-closed
  semantics plus the `-P OUTPUT DROP` verification readback.
- `sandbox/spec.py` — pure `snapshot_tag()` (the ref mint); reserved-prefix gate at
  image resolution; `aios.env_keys`/`aios.base_image` labels in `_assemble_plan`;
  populate `spec.snapshot_image` from the session row's pointer (NULL → unset).
- `sandbox/registry.py` — `release()` → snapshot-then-remove **+ DB pointer write in the
  same critical section** (host-side cleanup sequencing unchanged); drift-recycle path
  uses the same helper; `_provision` salvage preamble **+ pointer reconcile + snapshot
  resolution via `store.get`** (verified-negative rule, base-label drift check, pointer
  clears and `sandbox_fs_reset` events — store consumption and event appends live at
  this layer, not in the Docker backend);
  `release_all` → `stop_all` (bounded); `reap_orphans` deleted; `start_gc` / `_gc_once`
  with a pure `_classify_images()` for table-driven unit tests, pass-4 pointer
  reconciliation, and the ownership clause; per-session budget policy reads
  `handle.disk_limit_bytes`; registry holds the pool (events and pointer writes from the
  reaper path need it).
- `harness/worker.py` — boot: `start_gc(...)` replaces awaited `reap_orphans`; shutdown:
  `stop_all()`.
- `harness/context.py` — `_MODEL_VISIBLE_LIFECYCLE` allowlist + render branch.
- `models/environments.py` — `image` field validator (reserved prefix).
- `db/queries/__init__.py` — GC freshness query keyed on the `(session_id,
  last_event_seq)` event row; `unscoped_set_session_snapshot` /
  `unscoped_clear_session_snapshot` (worker-side, per the unscoped naming convention);
  the GC batch query gains the pointer fields; `get_session_provisioning` returns
  `snapshot_ref/host` so `build_spec_from_session` can populate the spec.
- `config.py`, migration 0080, openapi/SDK regen as §6.
- `harness/tool_dispatch.py` — **no changes** (evict semantics preserved).

## 8. Failure-mode analysis (consolidated)

| Failure | Outcome |
|---|---|
| Worker crash mid-commit | Tag still at S_n; corpse parent == S_n → lineage passes → recommitted at salvage. Untagged partial image = GC residue, swept. |
| Crash between commit and rm | Tag at S_n+1 (child of corpse's image) → `skipped_stale` → corpse removed; content already captured. |
| Crash (or DB failure) between commit and pointer write | Treated like a failed snapshot verb: corpse retained (no rm), and the preamble/GC pointer-reconcile sets the ref from local truth. First-commit-only window for *ref resolution* in v1 (later commits rewrite the same ref value; a gen≥2 crash leaves only `snapshot_bytes/updated_at` stale ≤1 tick — harmless, the column is reporting-only). |
| Crash between flatten-import and corpse rm | New tag exists; old chain orphaned-untagged → swept by retain rule next tick (visible only under `images -a`, which the sweep uses). |
| Daemon restart / host reboot | Running containers → stopped corpses (`--restart no`); worker dies with daemon (same host), boot tick salvages. |
| Double resume / concurrent workers | Single-host/shard: impossible (advisory singleton + procrastinate per-session lock + registry asyncio lock). Elastic pool: procrastinate's DB-level per-session `lock` serializes *steps* across workers but does not co-locate them — live containers, fire-and-forget tool tasks, and the in-process reaper outlive a step on the host that ran it, so the elastic shape additionally requires wake→host affinity while a session has a live container or uncommitted state (§9.3); without it, consecutive wakes hopping hosts produce silent stale resumes and two live containers per session. |
| Disk full during commit | Snapshot verb fails → corpse retained, ops-log error, pointer untouched; session's next provision retries salvage and fails loud (model-actionable raw error). Operator remediation: GC budgets + audit canary (§9). |
| Snapshot missing at resume (operator rmi / image-store loss) | **Detected**: pointer set + store verified-not-found → pointer cleared + model-visible `sandbox_fs_reset {reason: "snapshot_missing"}` + cold start (§5.3); provision span records the resolution. |
| Host loss | Detected at resume via the pointer (reset event), but the data is unrecoverable until a host-independent store exists — the §9.3 argument for the async `put`. With async push enabled: the replacement host's `store.get` falls back to the **last successful push** (a crash-dropped push is re-enqueued by the GC's push reconciliation, §5.5, so the lag is bounded by one tick in steady state). Elastic shape: the store is authoritative; loss bounded to unpushed deltas. |
| Env image ref changed while parked | Detected via `aios.base_image` label → snapshot discarded (`store.remove` + pointer cleared) + `sandbox_fs_reset` event + cold start (§5.3). |
| Mid-commit wake | Waking step blocks on the per-session lock ≤ commit duration (+ `put` duration in the elastic shape — both size-bounded), then provisions from the fresh tag. |
| Giant layer (100 GB) | Size-derived timeout admits it; budget enforcement flattens and pressure blocks new durable provisions; never an infinite retry loop. |
| Session deleted while corpse/image exist | API can't touch Docker; GC removes both within ≤1 h (corpse pass checks retain rule before salvaging — no wasted commit). |

## 9. Deployment prerequisites (eumemic-ops, lockstep)

1. **Prune-cron exemption + budget interlock (blocker, verified live).** Server B's
   hourly `docker-retention-prune` runs `docker image prune -af --filter "until=48h"` —
   it would silently delete every snapshot parked >48 h (and skip-empty pins `.Created`,
   so even daily-active read-only sessions get wiped). Lockstep change: add
   `--filter "label!=aios.managed=true"` to the cron and to
   `disk-fill-recovery.md` step 2; never enable Coolify's `force_docker_cleanup`. The
   exemption removes the only existing disk bound → it must land together with
   `sandbox_snapshot_pool_bytes` set to real host headroom (~15 GB on Server B), and the
   eumemic-ops `disk-usage` audit gains a label-filtered snapshot total so the canary
   fires before Postgres PANICs (the 2026-05-14 incident shape).
2. **Containerd-store verification battery (blocker for implementation, not direction).**
   Prod runs the containerd image store (`io.containerd.snapshotter.v1`, verified via
   `docker info`), while every Docker experiment behind this design ran on
   overlay2/graphdriver. Re-verify on a containerd-store daemon (half a day): label
   inheritance through commit, retag-in-place residue and its `images -a` visibility,
   `rmi` cascade semantics, prune `label!=` filtering, `SizeRw` probe cost at high file
   counts, the layer-depth wall, commit/flatten latency. Divergences are design input.
3. **Durability timing × scale shape (decision).** Until a host-independent store
   exists, **host loss = loss of that host's snapshots** — the DB pointer makes it
   *detectable* (reset events at resume, §5.3), not recoverable. The transport timing
   follows the scale shape (sign-off, §11):
   - **Elastic worker pool** (any worker resumes any session): requires **both** a
     host-independent `RegistryStore`/`ObjectStore` (a peer cannot find a daemon-local
     image; the pointer is written only after `put` returns; `snapshot_host` becomes
     advisory) **and wake→host affinity while a session has a live container or
     uncommitted local state** — procrastinate's per-session lock serializes steps but
     does not co-locate them, and tool tasks plus the idle reaper hold the container
     *between* steps; without affinity, consecutive wakes hopping hosts silently resume
     from the last committed generation while the previous host's container still holds
     newer state. The affinity mechanism (live-handle owner gating on the wake claim, or
     commit-per-step semantics) is a **named follow-on design**, not specified here:
     choosing this shape means designing it first.
   - **Shard-by-DB** (per-DB singleton worker, sessions pinned to their shard): affinity
     makes resume host-stable, so the registry store is **deferrable** — v1 runs
     `LocalDaemonStore`; `snapshot_host` records the home host for GC and future
     migration/rebalancing.

   **Recommended regardless of shape**: an async, off-critical-path `store.put` after
   each successful commit (commit local → pointer write → rm → enqueue background push;
   resume prefers local, falls back to `get` — the hybrid store of §5.2). Commit and
   resume latency stay local while a dying host stops erasing a user's invested
   environment — staleness bounded by the **last successful push**, with the GC tick
   re-enqueueing dropped pushes (§5.5) so the bound is one tick in steady state. The
   explicit sign-off: **async push in v1** (recommended) vs **seam-ready, defer the
   second store**.

   Backup reality: local snapshots live in `/var/lib/docker` — covered by whole-disk
   backups; a DB-only restore now yields *detected* resets rather than silent cold
   starts. Add the row to the eumemic-ops topology backup table either way.

## 10. Verification plan

E2E (`docker_harness`, gated by the existing `needs_docker` marker; force idle via
`registry._reap_idle_once(0.0)` or direct `release()`):

- **Contract positive/negative** (`test_sandbox_persistence.py`): write `/root/marker`,
  `/etc/marker`, `/tmp/marker`, `/dev/shm/marker`; start `sleep 1000 &`; release; assert
  tag exists; resume → first three markers present, `/dev/shm` marker **absent**,
  `pgrep sleep` empty. `apt-get install -y figlet` variant → `dpkg -l figlet` survives a
  second cycle. Zero-write release → image id unchanged (`skipped_empty`). Run the tag
  directly → `/workspace` empty (bind mounts excluded from snapshot).
- **Drift dissolution** (the headline claim): write `/root/marker` → attach a memory
  store mid-session (spec_version bump → recycle) → next tool call → marker present AND
  new mount visible.
- **Resume integrity**: resumed container has non-empty `$PATH`, `pwd`==/workspace,
  `HOME=/home/aios` — both after a plain commit and after a flatten round-trip with the
  depth threshold monkeypatched to 1 (this test catches both historical blockers).
- **Secrets**: `docker image inspect .Config.Env` on the committed tag → the
  `env_config.env` secret *value* absent; env-removal variant → removed key reads empty,
  not the old value, in the resumed container.
- **Salvage** (`test_sandbox_salvage.py`): out-of-band `docker stop` (simulated crash) →
  `evict()` → next tool call sees the pre-crash marker; stale-corpse variant never
  regresses tag content.
- **GC** (`test_sandbox_snapshot_gc.py`): session delete → `_gc_once` → image gone;
  archive boundary → image gone; unarchive/rearchive races
  fail closed; orphaned-chain residue is collected; pressure is signalled without
  deleting lifecycle-protected snapshots.
- **Lockdown sidecar** (`test_networking.py` extension): Limited env on a *resumed*
  snapshot whose `/usr/sbin/iptables` was replaced by `exit 0` pre-idle → lockdown still
  enforced (curl to unlisted host fails); sandbox cannot `iptables -F` (no NET_ADMIN).
- **Pointer discipline** (unit, FakeBackend + fake store): pointer written after
  snapshot success and before `force_remove`; never written on snapshot failure;
  salvage-preamble reconcile heals a NULL pointer when the local canonical tag exists;
  GC pass 4 clears the pointer on every removal.
- **Missing-snapshot detection** (e2e): commit → out-of-band `docker rmi` → resume →
  cold start + `sandbox_fs_reset {reason: "snapshot_missing"}` in the event log +
  pointer cleared.
- **Store seam** (unit): `LocalDaemonStore` verified-negative semantics (not-found vs
  indeterminate raises); snapshot transport/lookup (`put`/`get`/`exists`/`remove`/`size`
  of session artifacts) goes only through `SnapshotStore` — daemon *enumeration* stays
  on backend verbs; registry code never shells to docker directly.
- **Base-image drift** (e2e): commit → change the environment's image ref → resume →
  cold start on the new base + `sandbox_fs_reset {reason: "environment_image_changed"}`
  in the log + pointer cleared + old tag removed (the next idle commits fresh — no
  `skipped_stale`).
- Unit: `FakeBackend` grows the five verbs; snapshot-before-remove ordering;
  snapshot-failure ⇒ no `force_remove` but host-side cleanup still runs; lineage-gate
  truth table; `_classify_images` table-driven; tag derivation + `_is_registry_image`
  false; reserved-prefix gate (spec + pydantic); context allowlist rendering without
  watermark movement.

Docker availability: e2e requires real Docker (`DOCKER_HOST` per conftest); CI as today.

## 11. Decisions requiring sign-off

1. **DB snapshot pointer ships** (reverses the original draft's zero-DB-state stance,
   forced by the horizontal-scale requirement): the DB is source of truth for snapshot
   *location/existence* (`snapshot_ref/host/bytes/updated_at`), the store/daemon for
   *content/lineage*, with the GC reconciler as convergence authority. The
   previously-accepted silent operator-caused loss is now detected and evented (§5.3).
2. **Scale shape — elastic worker pool vs shard-by-DB**: decides whether the
   host-independent store is mandatory in v1 or deferrable, and whether `snapshot_host`
   is advisory or load-bearing (§9.3). Note the elastic shape carries a second
   prerequisite beyond the store: wake→host affinity for live containers (§9.3) — a
   named follow-on design. Shard-by-DB is the shape this document fully specifies.
   Needed to finalize §5.5/§9.3.
3. **Durability timing — async `store.put` in v1 (recommended) vs seam-ready-defer**:
   whether each commit is pushed to a host-independent store off the critical path, so
   host loss stops being data loss (§9.3).
4. **Base-image-ref drift semantics**: discard-snapshot + event (operator intent wins) vs
   keep-snapshot + event. Recommended: discard. The event is non-negotiable either way.
5. **Daemon-access threat boundary**: snapshot filesystems (and any agent-written
   secrets in them) are readable by anything with Docker-socket/host-root access for the
   retention duration — same trust domain as workspace dirs, longer retention. Accept +
   document. (An async-push store extends the same boundary to the registry/bucket —
   scope its ACLs with the sign-off.)
6. **Two-layer image-prefix gate**: spec-build gate (the boundary) + pydantic 422 (UX).
   Deliberate near-redundancy across different writer sets.
7. **Idle-TTL default 300 s → 1800 s**: operator-visible behavior change shipped inside
   this feature because the feature inverts the constant's economics.
8. **Per-account caps deferred** (usage metric ships, destructive quota enforcement doesn't);
   **`--read-only` closed as incompatible** rather than re-deferred; **`--cap-drop=ALL`
   stays deferred** (NET_ADMIN removal ships now via the lockdown sidecar).
9. **Rollback posture**: disable new snapshot creation while retaining lifecycle-
   protected artifacts; destructive cleanup remains tied to archive.

---

*Provenance: produced via two orchestrated workflows — (1) 9 agents verifying the brief
against the codebase and live Docker/podman/prior-art research (incl. live experiments:
commit whiteouts/latency/layer-wall, stop/start netns semantics, GC cascade); (2) a
design panel (full mechanism-B design + genuine A/C steelmen) followed by four
adversarial reviewers (security, crash/concurrency, ops, simplicity) — the ops reviewer
probed the production host directly (prune cron, disk headroom, containerd store). All
blockers and majors are folded into §5/§9; nothing in this document is unreviewed
first-draft material.*

*Revised 2026-06-10: horizontal scale promoted from "later" to stated requirement —
reverses the zero-DB-state decision (snapshot pointer columns, §5.1/§6), introduces the
`SnapshotStore` transport seam (§5.2), store-routed resume with detected snapshot loss
(§5.3), cross-host GC ownership (§5.5), the durability-timing × scale-shape decision
(§9.3), and the cross-host invariants (§5.11). Mechanism B, the snapshot sequence
internals, salvage, lineage gate, flatten, lockdown sidecar, prefix gate, idle-TTL
raise, and the containerd verification battery are unchanged. The revision was itself
adversarially checked (a residue sweep + a cross-host correctness attack) and those
findings — including the drift-discard artifact-removal fix, ownership-gated/CAS pointer
writes, the elastic-shape affinity prerequisite, host-independent ref naming, push
reconciliation, and the decommission procedure — are folded in above.*
