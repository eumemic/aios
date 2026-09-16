# Uncorrelated review — aios#2432 fixround, round `gvisfloor3`

**Tip reviewed:** `5683c1b861e1b8d28ad71fffc2535225529e4565`
("fix(sandbox): retry credential DNS bind on dual-protocol EADDRINUSE")
**Branch:** `gvisfloor3rev` (forked from implement tip on `gvisfloor3`)
**Implementer:** grok-4.6 · **Checker:** claude-opus-5 (maker ≠ checker)
**PR:** https://github.com/eumemic/aios/pull/2432 (branch `gvisfloor`)
**Scope:** light review — focused unit tests only, no docker e2e, no `-n`.

## Verdict: **FAIL** — one Medium defect (latent fail-open), **fixed in this worktree**

The diagnosis and the shape of the fix are right: TCP-first ephemeral bind +
UDP attach on the same port + bounded `EADDRINUSE` retry is the correct answer
to the CI red, and it does not weaken the one-resolver-per-session property.
The defect is in how `start()` decides it succeeded: it replaced a
structurally-enforced fail-closed postcondition with a check on mutable
instance state that `stop()` never clears, so a resolver that bound once can
return from a *failing* `start()` reporting a stale port with nothing
listening on it. In the module whose entire job is failing closed, that is not
a shape to ship. Reproduced, fixed, and covered by a regression test below.

---

## Findings

### M1 (Medium, FIXED) — `start()` could return "started" while unbound

`src/aios/sandbox/credential_dns.py` — as committed:

```python
for attempt in range(_BIND_ATTEMPTS):
    ...
    except OSError as exc:
        last_exc = exc
        await self.stop()
        if exc.errno != errno.EADDRINUSE or attempt + 1 == _BIND_ATTEMPTS:
            break
if self._port is None:                      # <-- success predicate
    raise CredentialDnsError(...) from last_exc
```

`stop()` does not clear `self._port` (it still reads it for the
`credential_dns.stopped` log line). So `self._port is None` only answers "did
we bind?" for an instance that has never bound. For an instance that bound
once and was stopped, every bind attempt can fail and `start()` still falls
through to `log.info("credential_dns.started", port=<previous run's port>)`
and returns — the caller then DNATs the sandbox's `udp/53` + `tcp/53` at a
port this process no longer owns, with no interception behind it. That is a
fail-open in the path whose own docstring says a failed bind "turns into a
failed provision, because a sandbox whose credential names cannot be pinned
must not be handed a credential", and it contradicts CLAUDE.md's *fail hard,
no fallbacks* / *correct-by-construction*.

Reproduced against the committed tip (`_BIND_ATTEMPTS=2`, all binds raising
`EADDRINUSE`, after one successful start/stop):

```
credential_dns.started  port=40707      <- first, real start
credential_dns.stopped  port=40707
credential_dns.stopped  port=40707      <- retry 1 failed
credential_dns.stopped  port=40707      <- retry 2 failed
credential_dns.started  port=40707      <- BUG: returns OK, nothing bound
```

Not reachable in production **today** — `SecretEgressProxy.__init__` builds a
fresh `CredentialDnsResolver` and `SecretEgressProxy.start()` is called once
per proxy (`spec.py:803`, `spec.py:947`) — which is why this is Medium and not
High. It is nonetheless a defect introduced by this commit: the pre-tip code
raised from inside the `except`, so the postcondition held for any call
sequence.

**Fix applied here:** raise from inside the loop; the loop now either binds or
raises, with no post-loop state check and no `last_exc` bookkeeping (net
simpler than the committed form).

```python
for attempt in range(_BIND_ATTEMPTS):
    try:
        await self._bind()
        break
    except OSError as exc:
        await self.stop()
        if exc.errno != errno.EADDRINUSE or attempt + 1 == _BIND_ATTEMPTS:
            raise CredentialDnsError("credential DNS resolver failed to bind") from exc
    except BaseException as exc:
        await self.stop()
        raise CredentialDnsError("credential DNS resolver failed to bind") from exc
```

Regression test added: `test_failed_restart_does_not_report_a_stale_port`
(verified RED on the committed tip, GREEN after the fix).

### M2 (Medium, FIXED) — the arm the change *creates* had no unit coverage

TCP-first removes the observed race (UDP picks a port, TCP `start_server`
dies on it — exactly the `('0.0.0.0', 56370)` shape in the CI log, since
asyncio's `create_server` re-raises the bind `OSError` with the resolved
address in the message and the errno preserved). It moves the race to the
other side: TCP wins a port whose UDP half is already held, so the **UDP
attach** is the new `EADDRINUSE` site and its retry must also unwind the TCP
server it already bound. The committed
`test_eaddrinuse_retries_on_a_new_ephemeral_port` forces the *TCP* bind onto a
live listener, so it never exercises that arm.

Verified manually that the arm is correct (forcing the first `SOCK_DGRAM`
bind to raise `EADDRINUSE`): the resolver retries, answers the sentinel over
UDP, accepts TCP **on the same port**, and `/proc/self/fd` is back to its
pre-start count after `stop()` (7 → 7, no leaked TCP listener). Added
`test_eaddrinuse_on_the_udp_attach_retries` so a future edit cannot break it
silently. (Passes on the committed tip too — it is coverage, not a bug fix.)

### L1 (Low, note) — retry logs `credential_dns.stopped port=None`

Every failed attempt calls `stop()`, which emits a `credential_dns.stopped`
line with `port=None` (and, after M1's stale-`_port` case, the *previous*
port). Harmless log noise; left alone rather than widening the diff.

### L2 (Low, note, pre-existing) — `CancelledError` becomes `CredentialDnsError`

`except BaseException` in `start()` converts a cancellation into a domain
error. Unchanged by this tip (the pre-tip code did the same) and it still
fails closed, so it is out of scope here.

### L3 (Low, note) — `assert sockets` in `_bind()` is stripped under `-O`

`assert sockets, "asyncio.start_server returned no sockets"` is the repo's
existing idiom (cf. the `port` property), so consistent; noting only that it
is not a runtime guarantee.

---

## Requirement-by-requirement

| # | Requirement | Result |
|---|---|---|
| 1 | Dual-protocol bind no longer fails closed on parallel-e2e `EADDRINUSE` | **PASS** — TCP `start_server(0)` first, UDP attached to the returned port, bounded 16-attempt retry on a fresh pair for either side. Both directions verified (committed test forces the TCP side; added test forces the UDP attach). |
| 2 | Two sessions must not share a resolver; one `dns_port` for udp/53 + tcp/53 | **PASS** — neither socket sets `SO_REUSEADDR`/`SO_REUSEPORT` explicitly; asyncio's `create_server` sets `SO_REUSEADDR` implicitly on POSIX (pre-existing, unchanged) and Linux does **not** let that bind over a socket in `LISTEN`. Empirically: 6 concurrent resolvers get 6 distinct ports, and a third-party `SO_REUSEADDR` bind onto a live resolver's port is refused with errno 98 on **both** TCP and UDP. DNAT still uses a single `dns_port` for `udp/53` + `tcp/53` (`setup.py:704-705`, `718-720`), so the shared-port constraint is real and is preserved. |
| 3 | Non-`EADDRINUSE` bind errors still fail closed | **PASS** — errno-gated: anything else raises `CredentialDnsError` on the first attempt (no 16× burn). Covered by the pre-existing `test_bind_failure_raises_credential_dns_error` (`OSError("no sockets today")`, errno `None`). Exhausted retries also fail closed (`test_persistent_eaddrinuse_still_fails_closed`). |
| 4 | Create-time `SizeRw` baseline + e2e runtime threading from `e28d3c39` intact | **PASS** — `git diff e28d3c39..HEAD` touches only `credential_dns.py`, its test, and `REVIEW.md`; no snapshot/backend file changed. `tests/unit/sandbox/test_snapshot_verb.py` green in the run below. |
| 5 | Focused unit coverage for the bind/retry path | **PASS after M2** — `tests/unit/sandbox/test_credential_dns.py`: retry-on-busy-TCP-port, persistent-`EADDRINUSE` fail-closed (committed) + UDP-attach retry and stale-port regression (added). |
| 6 | Commit message / diff match the claimed rationale | **PASS** — TCP-first, shared port because of the single `dns_port` DNAT, no `SO_REUSE*`, bounded retry, other errnos fail closed: every claim is in the diff, and the cited CI signature matches the failure asyncio actually produces for a UDP-first bind. |

## Checks run (focused; no full suite, no `-n`)

```
uv run pytest tests/unit/sandbox tests/unit/test_networking.py -q -p no:randomly
  -> 780 passed
uv run mypy src/aios/sandbox/credential_dns.py tests/unit/sandbox/test_credential_dns.py
  -> Success: no issues found in 2 source files
uv run ruff check / ruff format --check (both files)   -> clean
```

Docker e2e not run here (light review). The remaining e2e risk is the one this
change cannot remove: `_BIND_ATTEMPTS = 16` fresh ephemeral pairs is a bound,
not a guarantee, under genuine port-space exhaustion — correct behaviour is
still a failed provision, which is what the e2e would report.

## Leftovers for the Shepherd

- M1 + M2 are **fixed in this worktree** on `gvisfloor3rev` (product +
  tests); not pushed, not merged, no PR opened.
- L1-L3 are notes only; no action taken.
- `TASK.md` in the working tree is this round's brief (written by the
  Shepherd after the tip was committed) and is left unstaged.
