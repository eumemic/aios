# Uncorrelated review — implementer tip `7f5666f3` (eumemic/aios#2422)

Maker ≠ checker. Implementer: claude-opus-5 on `trigswap6b` (pushed to
`origin/trigswap2`). Reviewer: grok-4.6 on `trigswap6brev`. Scope: **this tip
only** (`7f5666f3` — *fix(sandbox): force direct-IP credential traffic through
the egress proxy*), against TASK.md / the High (direct-IP exfil of
`CREDENTIAL_SENTINEL_IP` DNAT in Unrestricted credentialed sandboxes).

Do not push. Do not merge. Do not open a PR.

## Verdict

**PASS.**

The High is closed at the ruleset, not by hoping clients ask DNS. Unrestricted
credentialed sandboxes now DNAT every outbound `tcp:443` except loopback to the
secret-egress proxy; the proxy already keys on ClientHello SNI (swap credential
hosts, SSRF-checked blind-relay otherwise). Direct-IP, cached, and never-resolved
addresses stop deciding whether the swap fires. Limited closes the mirror image
by withholding the credential host's own per-address filter `ACCEPT`, with the
shared-address case named rather than sold as closed. Name-based sentinel /
PKTINFO / hosts-first resolve are untouched.

`DONE.md` still narrates earlier rounds (PKTINFO / MASQUERADE / `route_localnet`)
and does **not** describe this tip. Product diff is `setup.py` +
`tests/unit/test_networking.py` only; the commit message is the real write-up.
Not a product miss — Shepherd can restamp DONE if wanted.

Focused tests: `uv run pytest tests/unit/test_networking.py tests/unit/sandbox/test_credential_dns.py -q` → **178 passed**. Docker absent here; CI remains the e2e oracle.

## Finding 1 — Unrestricted catch-all is the right primitive (High closed)

`build_secret_egress_dnat_script` now emits, after the shared `_nat_dnat_lines`
chokepoint:

```
-t nat -A OUTPUT ! -d 127.0.0.0/8 -p tcp --dport 443 \
    -j DNAT --to-destination "$PROXY_IP:<proxy_port>"
```

That is consistent with existing name-based interception: still one proxy, still
`:443` TCP, still `$PROXY_IP` from the same fail-closed alias lookup, still no
sampled `-d`. Nat OUTPUT is flushed on re-apply so the catch-all cannot stack.
Loopback is excluded (`route_localnet` is on). The proxy path this lands on is
already load-bearing (`SecretEgressProxy._dispatch`, `_RELAY_PERMITTED_MODES`):
credential SNI terminates and swaps; unrecognized SNI under Unrestricted is
pinned-resolved and spliced; no SNI is refused. Destination IP no longer
selects a bypass.

`apply_secret_egress_dnat` threads `assert_https_catch_all=True` as a **separate**
flag from `assert_drop`, so Limited cannot acquire a grep for a rule it does not
emit. The v6 companion (`tcp:443 DROP`, guarded like `#1207`) closes the same
bypass one stack down without a blanket Unrestricted v6 policy DROP.

`TestCredentialHostEgressVerdict` flips the High: sampled / unsampled /
`203.0.113.7` are `proxied` under Unrestricted; loopback stays `direct`;
non-443 stays `direct`. The verdict model learned `! -d` and CIDR matching so
it cannot invert the catch-all into “loopback-only.”

## Finding 2 — Limited withhold + honest residual

Limited cannot copy the catch-all: the proxy is STRICT there, so DNATing every
`:443` would black-hole ordinary allowed hosts. Subtracting `dnat_hosts` from
the allowed-host `ACCEPT` loop is the dual: no legitimate in-netns client
reaches those real addresses (the name is the sentinel), so the ACCEPT was
pure direct-IP surface; `-P OUTPUT DROP` refuses it.

Named residual is accurate: an address **shared with a different allowed host**
still carries that host’s ACCEPT. Closing it needs the proxy to carry the
Limited allow-set — a proxy change, not a ruleset one. The pinning test asserts
the mechanism (no credential-host ACCEPT of our own) and does not pretend the
shared-IP case is covered.

Refresh does **not** silently put those ACCEPTs back. Stamp/refresh resolve
inside the netns; a credential name answers only the sentinel, so
`new_limited_ips` never learns a real GitHub address to `-A`. (The refresh
docstring still talks about dual-host ACCEPTs being refreshed — stale comment,
not a hole.)

## Finding 3 — what this tip does not claim (residuals, not silent holes)

These match “smallest correct fix consistent with existing name-based
interception” (`_nat_dnat_lines` is itself `:443` TCP / IPv4):

- **UDP 443 / QUIC / HTTP3** and **TLS on non-443** still leave Unrestricted
  directly. Name-based sentinel traffic that is not `:443` is already REJECT.
- **IPv6 `:443` DROP does not exclude `lo`**, unlike the v4 catch-all. Inert
  today (`aios-sandbox` has no `--ipv6`); would drop in-netns `https://[::1]`.
- **`_run_verify` was not extended** to emit a realistic `iptables -S` catch-all
  and omit it fail-closed (the exact gap that burned earlier `:53` greps). The
  new grep does tolerate `( -m tcp)?`. Script-level + apply-threading tests
  exist; packet-fate tests do not depend on `-S`.

SNI-less `:443` (IP-literal HTTPS) is narrowed **on purpose** and documented:
the proxy cannot establish intent without a name.

## Non-findings (checked, not wrong)

- Hosts-first resolve, PKTINFO reply sourcing, sentinel DNAT, DNS `-I` +
  MASQUERADE, `route_localnet` INPUT guard: **not in this diff**.
- Limited still installs the byte-identical `_nat_dnat_lines` chokepoint;
  name-based credential HTTPS is still `proxied`, not merely dropped.
- `_EGRESS_RULE_RE` still requires a leading `-d <ip>`; the negated catch-all
  does not match, so refresh will not try to age it out.
- Catch-all is unconditional (no resolution loop); coverage is not a DNS sample.

No product change made on this review branch.
