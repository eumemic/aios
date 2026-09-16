High: the review coding agent no longer shares a privilege domain with the
process that holds the reusable proxy key. `prctl(PR_SET_DUMPABLE)` does
**not** seal GitHub-hosted runners — passwordless sudo on ubuntu-latest can
still read the launcher's memory — so broker+seal is not the boundary. The
launcher execs the harness as a dedicated unprivileged user (`eumemic-review`)
via `setpriv --no-new-privs`; that user cannot sudo, ptrace, or read the
key-holder. The agent still only sees a loopback broker token. The reusable
key is never in the harness environment.

Medium: writing the `### Code review` artifact is gated on a verifiable
diff-inspection digest (`<!-- inspected: lines=… sha256=… -->` matching
`git diff base...head`). A zero-exit harness that emits only the heading is
refused (`NO EVIDENCE OF INSPECTION`); the digest is the only accepting
channel.

The dropped uid is given `GIT_CONFIG_GLOBAL` with `safe.directory` for the
runner-owned checkout, and the launcher checks it can reproduce the digest
before the harness starts — otherwise every review hashes empty and is
refused as NO EVIDENCE. Only an `agent/` subdirectory is chowned; the
launcher `TemporaryDirectory` stays ours and is torn down with `sudo rm`
plus `ignore_cleanup_errors`, so cleanup cannot swallow the NO_EVIDENCE
exit or prevent writing the artifact.

Focused unit tests: `tests/unit/test_eumemic_bot_review.py`.

Hosts-first resolve, operator-controlled refresh hosts, runsc gateway bake,
and proxy-key-out-of-harness-env are unchanged.
