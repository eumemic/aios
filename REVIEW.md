# Uncorrelated review — `botpost2410g` tip `0b0effc9` (aios#2410)

**Verdict: changes requested — 1 High, 1 High/Medium, 2 Low. The High and the
High/Medium are fixed on this review branch (`botpost2410grev`); the Lows are
recorded below, one fixed in comment form, one left as noted.**

The bake path the implementer chose is the right *shape*. Every alternative
route dies on the same fact: glibc reads `_PATH_RESCONF` = `/etc/resolv.conf`
and nothing else, there is no env override, and the runsc operator root is
mounted READ-ONLY so `setup._RESOLV_PREAMBLE`'s `printf … || true` is a
guaranteed no-op there. TASK.md's advisory direction — bake to a non-special
path and reach it via symlink/bind/operator read path — cannot deliver the
required property from the Dockerfile alone. A same-path bake is the only
Dockerfile-level shape that can work, so `COPY --link` stays.

What the branch got wrong is everything around it: it ships red, and it asserts
as established fact a mechanism nobody has verified.

---

## F1 (High) — the branch is red on the unit shard

`tests/unit/test_detect_filter_sync.py::test_build_sandbox_triggers_on_every_copied_file[docker/sandbox-resolv.conf]`
fails on `0b0effc9`:

```
assert None
 +  where None = re.search('(?m)^COPY docker/sandbox\-resolv\.conf\s', '# aios sandbox base image...')
AssertionError: 'docker/sandbox-resolv.conf' is no longer COPYed by docker/Dockerfile.sandbox — drop it
from this parametrization (and from the build trigger) rather than pinning a path the image does not consume
```

The pattern anchors `^COPY <path>` with no room for flags, so `COPY --link …`
reads to it as a removal. This is a plain CI-red regression: the unit shard
fails long before the e2e that the change exists to turn green. It was missed
because DONE.md's verification is `pytest -q tests/unit/sandbox/test_sandbox_resolv_conf.py`
— a single file, which cannot see a drift check living two directories away.
CLAUDE.md asks for `uv run pytest tests/unit -q` before a commit.

**Fixed** — widened to `^COPY (?:--\S+ )*<path>\s`, with a comment saying the
pin is on the *source path*, not on the flags.

## F2 (High/Medium) — a hypothesis asserted as a root cause

DONE.md and the Dockerfile comment both state flatly that "BuildKit treats
`/etc/resolv.conf` as a daemon-managed build mount, so a plain `COPY` can be
absent from the committed image layer", and that `--link` "forces the resolver
bytes into a linked image layer".

The first half is a real phenomenon (moby/buildkit#1267, still open) but the
documented mechanism is a *RUN-time* bind mount of `/etc/resolv.conf` and
`/etc/hosts` into build containers — and this COPY already sits after every
`RUN`, which is precisely why the plain COPY was expected to work. The second
half has no support at all: no upstream source lists `--link` as a remedy for
this path. The external workarounds that are documented are "write it at
runtime" or "use a non-special path" — neither available here (see the preamble
above).

`--link` may well work; the mergeop shape is a plausible way around a
placeholder in the parent snapshot. But it is a **bet**, and nothing on this
branch can settle it: the unit pins read the Dockerfile text, and no source-level
test can tell a surviving layer from a stripped one. The only oracle is
`tests/e2e/test_sandbox_image_contract.py::test_image_layer_carries_the_embedded_dns_resolver`,
which needs a Docker daemon — unavailable in this checkout and unrun by the
implementer (that caveat in DONE.md is truthful).

Shipping an unverified bet is acceptable here; shipping it labelled as a
diagnosed root cause is not, because the next person to read that comment will
not know there is anything left to check.

**Fixed** — the Dockerfile comment now separates OBSERVED (a freshly built
image read back through `docker cp` had a 0-byte `/etc/resolv.conf`) from
HYPOTHESIS (the `--link` mergeop mechanism), names the e2e as the sole oracle,
and says what to conclude if it stays red: the same-path bake is dead and the
resolver has to reach the chrooted operator on the read path instead — a
non-special path on its own does *not* do it. DONE.md is rewritten to the same
standard. The resolver unit module gained a SCOPE paragraph stating it cannot
see the built image and that a red e2e must never be "fixed" by relaxing
anything there, and the e2e assertion now fails with a sentence instead of
`[] == ['127.0.0.11']`.

## F3 (Low) — `--link` raises the minimum builder, silently

`COPY --link` needs a Dockerfile frontend with mergeop support: BuildKit >= 0.10,
i.e. Docker >= 23. The file carries no `# syntax=` pin. That turns out to be the
*right* configuration — without a pin, an older daemon rejects the unknown flag
and fails the build loudly, rather than parsing it away and shipping an empty
resolver. Adding a `# syntax=docker/dockerfile:1.x` line would make the flag
work on older daemons and is tempting; it would also remove the loud failure.

**Fixed in comment form** — the requirement and the deliberate absence of the
pin are now written down in the Dockerfile, so the next person doesn't "helpfully"
add one.

## F4 (Low, pre-existing) — `Dockerfile.sandbox` advertises a command that does not exist

Line 22 offers `uv run python -m aios build-image` as the local-dev shortcut.
There is no `build-image` command anywhere in `src/`. Out of the blast radius of
this fix and left alone; noted so it can be deleted (per "don't deprecate,
delete") in a pass that owns that file's header.

---

## Items that verify clean

- **Isolation** — untouched. No change to `_RUNSC_OPERATOR_ROOT`, the chroot
  chain, the read-only operator mount, or the Limited-networking lockdown. The
  fix is one Dockerfile flag plus comments and tests. This is the smallest
  honest fix available given F2's constraint.
- **Test integrity (review item 3)** — no test was weakened. The resolver pin
  was *strengthened*: it now matches on the destination path, so a plain
  `COPY … /etc/resolv.conf` is found and rejected by name rather than silently
  missed. The e2e contract is unchanged in what it asserts.
- **Ancestry and retention (review item 4)** — `origin/master` (`abe20173`) is
  an ancestor of HEAD, 28 commits ahead. `17ef3de9` (chroot before the
  privileged runsc loader), `538ab985` (the original resolver bake) and the full
  review-harness series are all present.
- **DONE.md's Docker caveat (review item 5)** — truthful; `docker` is genuinely
  absent here. Its "2 passed" was also literally true, which is exactly why it
  was misleading: it is the report of a command narrow enough to miss F1.

## Verification run here

```
uv run pytest -q tests/unit/sandbox/test_sandbox_resolv_conf.py \
               tests/unit/test_detect_filter_sync.py \
               tests/unit/sandbox/test_docker_runtime_argv.py \
               tests/unit/test_gvisor_validation_workflow.py    # 24 passed
uv run ruff check / ruff format --check / uv run mypy  (touched files)  # clean
```

The e2e image contract was **not** run — no Docker daemon here. Whether this
branch actually fixes aios#2410 is still open, and only that test can close it.
