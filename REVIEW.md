# Uncorrelated review — gVisor snapshot empty-floor, tip `77e18b2745408725a8748e8ac322bf38f8668133`

- **Round**: `gvisfloor` (implementer grok-4.6, worktree `aios-gvisfloorrev`, reviewer claude-opus-5)
- **Base**: `cc9a3c0e` — `fix(sandbox): apply runsc egress rules in the target Sentry (#2410)`
- **Target defect**: master RED gVisor Validation, Actions 35146073686 —
  `tests/e2e/test_sandbox_persistence.py::test_zero_write_release_is_skipped_empty`
  expected `skipped_empty`, got `committed` at `empty_floor_bytes=8192`.
- **Verdict**: **FAIL**

The change is small, well-documented and mechanically sound in isolation
(single production call site, `max(configured, …)` so an operator override
still wins, runc untouched, #2410 operator-mount / Sentry-exec / proxy-key
paths untouched, two new unit tests — both pass locally). It fails review on
its **causal claim**: the container that produced the red is not a runsc
container, so the condition the commit gates on is not the condition that
caused the failure. The e2e goes green by moving the expectation, not by
matching the cause, and the real trigger stays ungated in production.

---

## F1 (blocking) — the failing container does not run under runsc; the gate is keyed on the wrong thing

`test_zero_write_release_is_skipped_empty` builds its spec with the module-local
`_spec()` helper (`tests/e2e/test_sandbox_persistence.py:53`), which constructs
`SandboxSpec(...)` with explicit keyword arguments and **never sets `runtime=`**.
`SandboxSpec` is a pure dataclass with no `__post_init__` and
`runtime: str | None = None` (`src/aios/sandbox/backends/base.py:132`), and
`DockerBackend.create` emits the flag only under `if spec.runtime:`
(`src/aios/sandbox/backends/docker.py:460-461`). The `daemon` fixture
(`tests/e2e/conftest.py:130`) hands back a bare `DockerBackend()` and does not
thread settings in either.

So in Actions 35146073686 this container ran under **Docker's default runtime
(runc)**, not gVisor — even though the step exports `AIOS_SANDBOX_RUNTIME=runsc`.
Sibling e2e files that genuinely need gVisor thread it explicitly
(`tests/e2e/test_sandbox_ipv6_lockdown.py:230` → `runtime=settings.sandbox_runtime`);
this one does not.

Consequences:

1. **The stated mechanism cannot be the mechanism.** "runsc's overlay adds more
   (gvisor#10256)" is inapplicable to a container with no Sentry. Likewise
   `--mount type=image` is appended only under `if spec.runtime == "runsc"`
   (`docker.py:356`), so the operator-root copy-up named in the commit message
   is not present here either. The only environmental delta in that job that
   actually reaches this container is daemon-wide:
   `features.containerd-snapshotter: true` in `/etc/docker/daemon.json`
   (`.github/workflows/gvisor-validation.yml`, "Register runsc Docker runtime"),
   plus the runner's Engine version — both of which apply identically to runc.

2. **The gate does not cover the failing configuration.** `snapshot_empty_floor_bytes`
   keys on `settings.sandbox_runtime == "runsc"` (`src/aios/config.py:1392`). A
   deployment on the containerd image store running the default runtime — which
   is exactly the shape CI just demonstrated produces `SizeRw > 8192` — keeps the
   8 KiB floor and keeps growing a snapshot chain on every idle for chat-only /
   read-only sessions. That is the #923 condition the floor exists to prevent,
   and after this change no test can detect it.

3. **The e2e's green is coincidental.** The test now derives its own expectation
   from the same helper production uses (`test_sandbox_persistence.py:176-178`),
   reading `AIOS_SANDBOX_RUNTIME` — an env var that in this test affects only the
   assertion, never the container under test. The assertion and the code now move
   together; the test no longer pins the floor against observed reality.

## F2 (blocking) — 64 KiB is unmeasured; neither bound it claims is verified

The commit picks "16 inodes" but never records the **actual** no-write `SizeRw`
from the failing run — the one number that would justify a value. The CI failure
message at the time printed only `got committed`, and the run link is cited
without the observed figure. Both bounds asserted in the message are therefore
unverified:

- lower bound ("above the empty baseline"): unknown baseline — if it is, say,
  70 KiB, the job is still red and the next patch is another guess;
- upper bound ("still sits below a 64 KiB tenant write plus directory inodes
  (~72 KiB)"): the helper writes **exactly** 65536 bytes
  (`_write_substantial`, `test_sandbox_persistence.py:80-84`;
  `test_sandbox_provision_path.py:126`), i.e. exactly the new floor. `st_blocks*512`
  for that file is 65536, so it clears `size_rw <= empty_floor_bytes`
  (`docker.py:872`) **only** by the baseline overhead whose size is the very
  unknown that prompted the change. That is circular.

`tests/e2e/test_sandbox_salvage.py` makes this concrete: it drives the real
registry (`SandboxRegistry(backend=backend)` → `_salvage_session_corpses`,
lines 103/118, 161/193) and therefore now runs against the **production**
65536 floor under the gVisor job env, with a corpse that wrote
`echo pre-crash > /root/survivor` + a 65536-byte blob (line 110/166) —
clearing the floor by ~4 KiB plus that same unmeasured baseline.

Also stale: `_write_substantial`'s docstring still reads "well over the
empty-floor (8 KiB)", which is now false for the runsc path (it is *at* the floor).

## F3 (major) — the discard window widens 16× and is not bounded or mentioned

`skipped_empty` is not a no-op: `_snapshot_uncounted` returns before commit
(`docker.py:872-882`) and `_snapshot_and_remove` then `destroy()`s the corpse
(`registry.py:1800-1802`), so the writable layer is **discarded**. Raising the
floor to 64 KiB means any runsc session whose rootfs delta lands under
~(64 KiB − baseline) silently loses its writes at release.

`/workspace` is a bind mount and is excluded from the snapshot
(`test_sandbox_persistence.py:130-131`), so tenant workspace files are safe.
Not safe: `/root` dotfiles, `~/.ssh`, `~/.config`, `git config`, small
`pip install --user` results, `/etc` edits — the `/root`, `/etc`,
`/usr/local` persistence the suite's first test exists to guarantee. The commit
neither quantifies this window nor mentions the trade-off. An eightfold-plus
increase in silently-dropped writes is a durability change that needs to be
stated, and ideally bounded by measurement rather than by a round number.

## F4 (minor) — floor keyed on the current global setting, not the corpse

`registry.py:1822-1824` passes `settings.sandbox_runtime`, i.e. the runtime
configured *now*, to judge a corpse that may have been created under a different
one (salvage of a pre-flip corpse, `_snapshot_and_record` called with a bare
`sandbox_id`). The backend already inspects the container on this path
(`_inspect_container_for_snapshot`), so the corpse's own `.HostConfig.Runtime`
is available and would be exact.

---

## On the review question: floor bump vs. runtime-aware `SizeRw`

Runtime-aware — and more precisely **container-aware** — is the right shape, and
F1 is why: the quantity that varies is the writable layer's *empty baseline*,
which is a property of the image store and the container, not of a runtime
string in settings. The robust form is to measure it rather than name it:

- stamp the container's `SizeRw` at create, before any tenant exec, onto the
  handle (or a label, so salvage of a corpse can read it back), and
- make the identity short-circuit `size_rw - baseline <= configured_floor`.

That keeps the discard window at one page regardless of store or runtime, needs
no magic constant, removes the special case instead of adding one, and is
correct-by-construction in the CLAUDE.md sense ("unify toward minimal
primitives… encode variation as a *kind*, never a flag"). It also fixes the runc
+ containerd-snapshotter case F1 leaves open.

If that is judged too large for a CI-red unblock, the minimum acceptable
smaller fix is: (a) capture the real no-write `SizeRw` under the failing
configuration and cite it, (b) key on the image store (or make it a plain
setting the gVisor job sets) rather than on `sandbox_runtime`, and (c) state the
resulting discard window explicitly in the config docstring.

## Regressions / #2410 integrity

- No #2410 surface touched: `_runsc_operator_image`, `--mount type=image`,
  the Sentry-exec chroot path and proxy-key isolation are unchanged. Confirmed
  by diff inspection of `docker.py` (comment-only change at the identity
  short-circuit) and `registry.py` (one call-site change).
- Sole production consumer of the new helper is `registry.py:1822`. All other
  `empty_floor_bytes=` sites are tests passing a literal `8192`, unaffected.
- `runtime == "runsc"` matches the established idiom (`docker.py:325,356,1383`).
- New unit tests pass: `uv run pytest tests/unit/test_config.py -k snapshot_empty_floor` → 2 passed.
- Not run in this round (OOM budget): mypy, full ruff, gVisor e2e.

## Leftovers applied

None. F1 requires re-keying the gate (and, for a real fix, measuring the
baseline) — not a leftover that can be applied cleanly on top of this tip.
Product code left untouched.
