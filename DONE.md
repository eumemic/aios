# Done — aios#2410 fixround

Branch `botpost2410i`, on top of `e10f4e07` (`origin/master` is an ancestor;
31 commits ahead, no rebase needed). All prior #2410 security/harness work is
kept — the chroot-before-loader operator chain, the runsc machine allow-list,
the operator shadow map and its test. Not pushed, no PR.

Two items were asked for. Both are done.

---

## 1. The empty `/etc/resolv.conf` layer

**Verdict: baking `/etc/resolv.conf` is impossible under BuildKit.** The
sanctioned second branch was taken — the operator/chroot read path is fixed by
removing its dependence on that file entirely.

### Evidence

No Docker daemon is available in this environment, so the evidence is registry
blobs, the CI job log, and moby/BuildKit source — not another local build.

* **E1.** `docker/sandbox-resolv.conf` was 838 bytes containing exactly one
  `nameserver 127.0.0.11`, and is not `.dockerignore`d. The input was correct.
* **E2.** The failing lane genuinely rebuilt. Job log for run
  `34807834078` (`gh api .../jobs/103863091603/logs`) shows
  `#11 [6/7] COPY --link docker/sandbox-resolv.conf /etc/resolv.conf`,
  `#11 DONE 0.0s`, `writing image sha256:d98fe7ad…`,
  `naming to docker.io/library/aios-sandbox:ci` — with
  `AIOS_DOCKER_IMAGE=aios-sandbox:ci`. Not a stale `:smoked` pull.
* **E3.** `python:3.13-slim-bookworm` (amd64) ships `etc/resolv.conf` in layer 0
  as a **regular 104-byte file** (`# https://1.1.1.1 …`, `nameserver 1.1.1.1`,
  `nameserver 1.0.0.1`) — read out of the layer blob over the registry v2 API.
* **E4.** `ghcr.io/eumemic/aios-sandbox:smoked` (9 layers, built before the COPY
  existed) carries `etc/resolv.conf` **only** in that base layer, same 104
  bytes, no whiteout and no aios layer rewriting it.
* **E5.** The CI oracle read `''` — neither 838 bytes nor 104.
* **E6.** The probe is faithful. In moby v28, `docker cp` on a **never-started**
  container goes `containerArchivePath` → `openContainerFS` → `daemon.Mount` +
  `setupMounts`, which appends `Container.NetworkMounts()`; that method emits
  `/etc/resolv.conf` **only when `ResolvConfPath != ""`**, and that field is set
  only at container **start** (`buildSandboxOptions` /
  `initializeNetworkingPaths`) or by an explicit volume (`TrySetNetworkMount`).
  `daemon/create.go` never sets it. So nothing is mounted over the path and
  `docker cp` reads the layer's own content.
* **E7.** The oracle was introduced in `bce1dfc2` alongside the bake; it has
  never been green.

**Conclusion.** E3+E4 are decisive: if the `COPY` were merely *ignored*, the
read would return the base image's 104 bytes. It returned `''`. So the `COPY`
does not land the bytes — it *replaces* the inherited file with an **empty
entry**. Observed with plain `COPY` (which is what motivated `ce9619b7`) and
again with `COPY --link` (run `34807834078`). This matches
moby/buildkit#1267, where tonistiigi states these files "are configured by the
container runtime… BuildKit will not allow writes instead of silently ignoring
them" and "BuildKit currently mounts these read-only". `--link` changes nothing
because the path, not the write mechanism, is what is special-cased.

### The fix

Stop reading a resolver config file. glibc (`getent`) can only read
`/etc/resolv.conf` — there is no env override for nameservers — so `getent` had
to go. Resolution now **names the server as an argument**:

```sh
resolve_ipv4() { busybox nslookup "$1" 127.0.0.11 2>/dev/null \
  | awk '/^Name:/ { answer = 1 }
         /^Address:/ && answer && $2 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { print $2 }' \
  | sort -u; }
```

* `busybox` is already in the image and already the runsc chroot's entry
  binary. Being **static** it carries no `PT_INTERP`, so it is bound in a new
  `_RUNSC_OPERATOR_STATIC_COMMANDS` family invoked directly rather than through
  the operator image's `ld.so` (running a static binary through the loader would
  fail). It is shadowed to the operator root exactly like every other command,
  so the tenant cannot substitute it.
* Debian bookworm's `busybox-static` is built with `CONFIG_NSLOOKUP=y`
  (verified in `debian/config/pkg/static` for `1:1.35.0-4+deb12u1`). The e2e
  contract pins the applet's presence anyway.
* The parse was checked against busybox 1.35 `networking/nslookup.c`: answers
  print `Name:\t<name>` then `Address: <ip>`, the server block prints
  `Server:`/`Address:\t<ip>:<port>` first. Taking answers only **after** a
  `Name:` line means the resolver's own address can never be mistaken for an
  answer (which would ACCEPT 127.0.0.11 or DNAT every credential host at the
  resolver); the dotted-quad test drops AAAA, preserving the IPv4-only
  fail-closed semantics of #978. A miss prints nothing → no rule → fail-closed.

Deleted as a consequence: `setup._RESOLV_PREAMBLE` (the `printf … >
/etc/resolv.conf || true` that was a silent no-op under runsc anyway),
`docker/sandbox-resolv.conf`, the Dockerfile's `COPY --link` and its
hypothesis comment block, and the file's entries in both workflow triggers.

### Rejected alternatives

* **Symlink `/etc/resolv.conf` → a baked non-special path.** Another unverified
  BuildKit bet, untestable here, and it keeps glibc's single-file dependency.
* **Bind-mount a writable resolv.conf into the chroot.** Reintroduces exactly
  the tenant-poisonable resolver the read-only operator root exists to prevent.
* **Resolve worker-side and pass literals in.** Loses in-netns resolution
  through Docker's embedded DNS (container aliases, the proxy alias) and is a
  much larger architectural change.

### Oracles

`test_image_layer_carries_the_embedded_dns_resolver` is gone — it asserted a
property that cannot hold. Replacing it:

* `tests/e2e/test_sandbox_image_contract.py::test_busybox_ships_the_nslookup_applet`
  — the applet is configurable at busybox build time; without it every
  allow-list empties silently.
* `…::test_busybox_nslookup_answers_from_the_embedded_dns` — the live oracle.
  On a throwaway user-defined network (where Docker serves embedded DNS), the
  image resolves its own alias against `127.0.0.11` and the **output shape** is
  asserted: the server block precedes the `Name:` line, and an IPv4 answer
  follows it. Hermetic — no upstream DNS, no internet.
* `tests/unit/sandbox/test_sandbox_dns_resolution.py` — executes the real
  emitted `resolve_ipv4` against a stub busybox printing that exact shape:
  A records only (AAAA and the server's own address excluded), and a miss
  yields nothing. Also pins that no generated script mentions `/etc/resolv.conf`
  or `getent`, and that the Dockerfile bakes no resolver (the regression guard
  for "just COPY one in").

---

## 2. The runsc operator image mount

`DockerBackend.create()` mounted `get_settings().docker_image` at
`_RUNSC_OPERATOR_ROOT` while the tenant container ran
`spec.snapshot_image or spec.image` — two independent sources for one
relationship.

New `_runsc_operator_image(spec)` derives the mount from **`spec.image`** and
**rejects a mismatch** with `settings.docker_image`, raising
`SandboxBackendError` before the daemon is touched (same shape as
`_require_runsc_supported_machine`). Deriving alone would be a privilege
escalation: `EnvironmentConfig.image` is a free-form tenant field (#724), and
the operator root is the first thing a `--privileged` exec enters — a
tenant-chosen `/usr/bin/busybox` would run there holding `NET_ADMIN` the
sandbox itself was denied. `spec.snapshot_image` is deliberately never
considered: it is the tenant's own mutated rootfs, the exact filesystem the
chroot exists to escape. The gate is runsc-only; runc keeps applying its
lockdown from a separate operator-image sidecar, so a custom sandbox image
still works there. `run_netns_sidecar`'s docstring was corrected to match.

---

## Checks

`uv run ruff check src tests`, `uv run ruff format --check src tests` and
`uv run mypy src tests` (1098 files) are clean. `uv run pytest tests/unit` is
green: 6202 tests, all passing. Under `-n 2` on this memory-tight host a
handful of unrelated tests (image downscaling, an MCP pool retry, a browser
quota) flake differently on each run — two runs produced disjoint failure sets,
and every one of them passes serially. Nothing in `tests/unit/sandbox` or
`tests/unit/test_networking.py` flaked.

The e2e docker shard cannot run here (no daemon); the two replacement
image-contract tests above are what CI must prove.
