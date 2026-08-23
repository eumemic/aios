# Inference accounting

aios attributes inference through an immutable creation tree containing both
sessions and workflow runs. The rule is simple: the node that creates a child
owns that child's spend. Invoking an existing session does not change ownership.

## Resource usage

`GET /v1/sessions/{id}` and `GET /v1/runs/{id}` expose:

- `usage.own`: inference performed at this node;
- `usage.subtree`: `own` plus every creation descendant, transitively across
  session and workflow-run boundaries;
- `usage.own_rate` and `usage.subtree_rate`: rolling-window counters normalized
  per hour.

The flat fields retained beside these values are compatibility fields. Session
flat counters remain self-only. Workflow-run flat counters retain the older
direct-child budget view. New accounting consumers should use `own` and
`subtree`.

Subtree usage is live. It includes active, archived, errored, cancelled, and
terminated descendants. A child's ownership does not expire when its creator
finishes waiting, so a finished node's subtree can continue growing while a
descendant runs. This is expected.

## Rates and coverage

Rates come from an append-only delta ledger written atomically with each
cumulative meter. `window_seconds` is the requested rolling window;
`observed_seconds` is the portion covered by the ledger. `complete=false` means
the account is newer than the requested window. The values remain normalized to
one hour, using the explicit observed denominator.

## Ranked consumers

`GET /v1/usage/consumers` ranks account root nodes by live subtree rate in one
request. Query parameters:

- `window_seconds` (default `86400`, 60 seconds through 30 days);
- `metric=cost_microusd|total_tokens`;
- `limit` (default `20`, maximum `100`).

Only roots appear in this view. Creation gives each node exactly one root, so
the rows are additive and `share` values do not double-count a descendant under
both parent and child. Point resource reads still expose every intermediate
node's own and subtree values.

Traversal uses a visited-set-equivalent recursive `UNION`. A malformed legacy
cycle terminates and each `(kind, id)` contributes at most once per root.
