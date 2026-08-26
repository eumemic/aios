# Changelog

## Unreleased

- Distinguish a dead OAuth refresh token (RFC 6749 `invalid_grant` — revoked,
  expired, or otherwise unrecoverable) from a generic/transient
  `OAuthRefreshError` via a new `OAuthReauthRequiredError` subclass
  (`error_type: "oauth_reauth_required"`), so a caller can branch on "this
  connection needs to be reconnected" instead of retrying. Severity/eviction
  wiring (status_code 502, non-evicting on the MCP dispatch path, evicting via
  the `http_request` built-in's oauth2_refresh path) is unchanged (#192).
- Extend the #1975 diagnostic harness with timeout-scoped slow HTTP and
  streaming call graphs inside open transactions, a pre-saturated 16-slot pool,
  13 queued waiters, and per-storm client/server recovery checks.
- Rebuild the asyncpg cancellation investigation harness with synchronized
  acquire, post-query/pre-delivery, and release phase instrumentation, an
  asserted phase census, incident-shaped concurrency, and pool/server leak
  checks after every randomized storm (#1975).
- Close PR #1979 adversarial-review findings: strict called-object pooled-connection linting and verified issue pragmas, cross-process OAuth refresh arbitration/race adoption with bounded local locks, and advisory-lock-safe threaded workspace deletion without holding a pooled connection.
- Make production worker watchdog telemetry fail-open with reconnect/backoff and bounded query, stack, log, and forensic-file capture; correct workflow journal payloads and proxy task attribution; align dead-man session/workflow counters.
- Make durable sandbox tarballs canonical across Docker cache pruning, add CAS publication, separate filesystem GC, and enforce persistent-store disk preflight and capacity pressure.
