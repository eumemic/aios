# Uncorrelated review — aios#2410 rebase round `botpost2410n`

**Verdict: PASS.**

| | |
|---|---|
| Reviewed tip | `563da60d1665cac65ed7b0369a6cc7e6dfab2041` |
| Branch | `gvisorgrn` (this checkout `botpost2410nrev` @ same SHA; not pushed) |
| Implementer | gpt-5.6-sol on `botpost2410n` |
| Checker | claude-opus-5 on `botpost2410nrev` |
| `origin/gvisorgrn` | `563da60d` — matches HEAD (already pushed by implementer) |
| `origin/master` | `e4679395d5eb315f63dc4737bd9bc5b3aca4197b` (`#2429`) — ancestor of HEAD; master has not moved |

No merge commits in `e4679395..HEAD`. First unique commit (`fdfee26c`) has parent `e4679395`. Rebase, not a merge. No push, no PR, no gVisor Validation dispatch from this review.

---

## Scope vs TASK

1. **Rebased onto `origin/master` @ `e4679395`.** Holds.
2. **Hosts-first `resolve_ipv4`.** Holds: awk `/etc/hosts` first, then `busybox nslookup "$1" 127.0.0.11`. `getent` is comment-only (banned).
3. **Conflicts.** Only `.github/workflows/gvisor-validation.yml` and `tests/unit/test_gvisor_validation_workflow.py`. Resolution prefers `#2429` runsc-capability scoping and keeps `#2410` bits that still apply. `persist-credentials: false` is not owned by the gVisor workflow; it remains on `actions/checkout` in `.github/workflows/eumemic-bot-review.yml`.
4. **Focused tests.** `uv run pytest tests/unit/test_gvisor_validation_workflow.py tests/unit/sandbox/test_sandbox_dns_resolution.py -q` → **21 passed**. Full unit suite / xdist not run (ops constraint).
5. **DONE.md.** Records rebase onto `e4679395`, master's scoping, hosts-first / no-getent, and passing focused tests. Matches this tip.

---

## Findings

No blocking findings.

### Rebase onto `#2429`

`git merge-base --is-ancestor e4679395 HEAD` is true. `origin/master` is still `e4679395`. Unique history is a linear replay of `#2410` on top of that tip.

### Workflow conflict

Diff vs `origin/master` for `gvisor-validation.yml` is **additive only**:

- Master's selector is intact:
  `-m 'docker and not netns_sidecar_egress and not runsc_dns_unresolved and not perf'`
- `#2410` still present: `containerd-snapshotter`, `--mount type=image` probe after daemon restart.

No conflict markers. Checkout in this file is still `actions/checkout@v4` without `persist-credentials` — correct; that flag lives on the review workflow.

The unit drift test matches the resolved YAML: master's semantic selector pin, plus `#2410`'s `continue-on-error` gate and `test_gvisor_workflow_proves_the_daemon_supports_the_operator_image_mount`.

Non-blocking: a leftover comment still describes ``-m 'docker and not perf'`` immediately above the env block. The **command** (and the test) use the full `#2429` marker. Comment drift only.

### Hosts-first resolver

Emitted helper:

```sh
resolve_ipv4() {
  _hosts_ips=$(awk -v name="$1" '…' /etc/hosts 2>/dev/null | sort -u)
  if [ -n "$_hosts_ips" ]; then printf '%s\n' "$_hosts_ips"; return 0; fi
  busybox nslookup "$1" 127.0.0.11 2>/dev/null | awk '…' | sort -u
}
```

Product commit on this rewrite is `47b85325` (`fix(sandbox): resolve host names from /etc/hosts before DNS`). Pre-rebase SHAs `88c0d3b2` / `ab25c3f6` are not ancestors (expected after rebase). No `getent` in generated scripts.

### DONE.md vs reality

SHA `e4679395`, conflict choice (master scoping + keep `#2410` CI), hosts-first, no `getent` — all true. Brief, as asked.
