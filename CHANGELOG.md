# Changelog

## Unreleased

- Rebuild the asyncpg cancellation investigation harness with synchronized
  acquire, post-query/pre-delivery, and release phase instrumentation, an
  asserted phase census, incident-shaped concurrency, and pool/server leak
  checks after every randomized storm (#1975).
- Close PR #1979 adversarial-review findings: strict called-object pooled-connection linting and verified issue pragmas, cross-process OAuth refresh arbitration/race adoption with bounded local locks, and advisory-lock-safe threaded workspace deletion without holding a pooled connection.
- Make production worker watchdog telemetry fail-open with reconnect/backoff and bounded query, stack, log, and forensic-file capture; correct workflow journal payloads and proxy task attribution; align dead-man session/workflow counters.
