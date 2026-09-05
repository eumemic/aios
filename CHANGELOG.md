# Changelog

## Unreleased

- Snapshot flatten trigger and the snapshot pool budget now measure the on-disk
  CHAIN cost (Σ layer bytes, via `docker history`, cached per content-addressed
  image id) instead of `docker image inspect .Size` — the current filesystem
  *view*, which charges a superseded byte once however many copies overlay
  still holds in the interior layers. On server-b three live chains held ~65 GB
  for ~23 GB of content while reporting 6.6/9.0/7.6 GB, so neither the 12 GiB
  per-session budget nor the 200-layer depth ceiling could ever fire (zero
  `flattened` events in 24 h), and the pool reclaimer logged
  `reclaimed_bytes: 0` every tick against a budget it read as 28.6/60 GB used.
  A flatten now also fires on `chain > 2 × view` — more than half the chain is
  dead history — which is the one budget-less trigger short of the depth
  ceiling. The flatten headroom gate still sizes on the view, since the export
  writes content, not history.

- **Operator triggers show up in `list_account_triggers`; a non-finite
  `budget_usd` is a 422 (#2525).** The account-wide trigger read INNER JOINed
  `sessions`, so operator triggers (#2473, no owning session) never appeared
  and every liveness monitor built on it was blind to them. It now LEFT JOINs
  with the scheduler's own liveness predicate: session triggers on archived
  sessions are still excluded, and operator triggers are listed. Each row now
  carries `owner_kind` (`session` | `operator`); `owner_session_id` is `null`
  for operator rows. Separately, `budget_usd` on trigger workflow actions
  (session and operator, create and replace) rejects `inf`/`nan` (`1e400`
  parses to `inf`), and the 422 handler no longer crashes rendering a
  non-finite `input`, which had turned that 422 into a 500.
- **Tool results carry a received time (#2282).** `build_messages` now renders
  every tool result with the same `[received=…]` envelope a user message gets,
  from the result event's own `created_at` (answer time, in the account's
  timezone). Before, a result sat right after its tool call with no time signal,
  so a call that stayed open for hours (an `ask_user` card, an approval) read as
  answered at ask time. String content gets a leading line. List content gets
  it on the first text part, or as a new leading text part. Blind-spot result
  injections get a `[received=…]` line under their `[Tool result: …]` header.
  Append-time token pricing now counts the envelope.
- **`tool` CLI always writes parseable JSON to stdout on transport failure
  (#2227).** A broker request that timed out used to print a Python traceback
  and nothing on stdout, so callers doing `json.loads` saw `Expecting value`,
  and a lenient caller read the timeout as an empty result. Timeouts, an
  unreachable broker, a missing socket, non-JSON broker responses and non-2xx
  broker replies now write one JSON line to stdout, e.g.
  `{"error": {"kind": "timeout", "timeout_s": 30.0, "path": "/builtins/http_request"}}`,
  and exit non-zero (2 for transport failures, 1 for broker HTTP errors). The
  human-readable message goes to stderr. Successful calls are unchanged, and the
  30 s timeout is unchanged too.
- **Oversized request inputs are refused at the caller, never truncated
  (#2122).** `agent()` in a workflow now has a documented input limit:
  1,000,000 characters (Unicode code points, the same `MAX_USER_MESSAGE_CHARS`
  the message endpoint enforces), measured on the serialized input. A string
  counts as-is. Any other value is measured after `json.dumps`, so JSON syntax
  and escaping count too. An input over the limit is refused before the child
  session exists. The `await` raises a catchable `AgentError` with
  `kind="input_too_large"`, and its message gives the actual size and the
  limit. The other request writers check the same bound and return 413
  `payload_too_large` when it is exceeded: the `invoke` API (checked before the
  servicer session is created) and a request to an existing session. The ~8k
  boundary in the issue was never a delivery cap. It is the obligations
  reminder's 8,192-character render budget. Since #2258, a reminder over that
  budget is marked `[TASK ABRIDGED IN THIS REMINDER …]` and points back to the
  full original request instead of telling the agent to refuse.

- **Agents can address a role instead of a cached agent id (#1940).** New
  `resolve_role(role)` model tool (grantable as `{"type": "resolve_role"}`)
  returns the live agent id holding a role in the caller's account. An agent
  holds a role when its `name` or its `metadata.role` equals it. Archived agents
  never match, so after a re-spawn (archive + re-create) the role resolves to the
  new id. Resolution fails loudly: `404 no live binding for role 'X'` when
  nothing holds it, and `409` listing the candidate ids when more than one live
  agent does. It never returns an empty result. This is the v0 cut from the
  addressing design, built on the existing agents table. The durable
  `role_bindings` table, `list_roles` and `wake_role` are still to come.

- **Trigger writes warn when a monitor can disable itself (#2402).** A standing
  (`cron` / `run_completion` / `external_event`) `sandbox_command` whose command
  contains an explicit `exit N` whose bash status (N mod 256; signed or quoted literals included) is non-zero now gets a warning in the
  create/update response's `warnings`. Each non-zero exit is a failed fire, and
  5 in a row auto-disable the trigger. That is how all three
  `company-alive-heartbeat*` emitters went dark: each ran `exit 1` on the
  ping-failed path. The warning asks for the finding to be reported another way
  and an `exit 0` on purpose. It only warns and never rejects the write. It is
  a text check, so it cannot see a command whose last statement simply fails.
  The `sandbox_command` tool schema description now states the same rule.

- **lane_activate sends a complete trigger update again, and lanes can be armed
  capped (#2463).** The trigger PUT omitted `version`, `max_outstanding_runs` and
  `budget_usd`, which `WorkflowActionReplace` now requires, so every trigger
  update 422'd. The script now emits every key, with explicit `null` for any the
  lock does not declare (same behaviour as before). A lock's
  `cron_trigger.action` may now declare `max_outstanding_runs` and `budget_usd`;
  a change to either counts as trigger drift. The lanes test fake now validates
  trigger bodies against the real `TriggerCreate`/`TriggerUpdate` models.
  An invalid cap (`max_outstanding_runs` not null / int >= 1, `budget_usd` not
  null / finite > 0) is refused by `LaneLock.from_dict` (`ValueError`, using
  `WorkflowAction`'s own bounds) and by `lane_activate` at read-lock, before
  any object is created or updated.

- **Run budgets now count the whole subtree, bind a parked run, and can be set
  per trigger (#2446, parts a-c).**
  (a) New `WorkflowAction.budget_usd` (`> 0`, default `null` = no budget, as
  before), passed to each launched run as its `budget_usd`. (b) `budget_usd`
  used to count only the run's direct child sessions plus its own `call_llm`
  meter, so a grandchild (a child's own `call_agent`) or a sub-run spent
  outside it. The gate, the `budget()` builtin and the refusal message now all
  read the creation-subtree cost from the shared accounting rollup. (c) The gate
  ran only when a new `agent()`/`call_llm` opened, so a run parked behind a
  burning child was never stopped. The needs-step sweep now also wakes a
  suspended, budgeted run with an open `agent()` call once its subtree spend
  reaches the budget. The step force-resolves each open call through the #2440
  exit path: `AgentError(kind="timeout", bound="budget")`, the child gets a
  cancel marker, and the script can catch it. Work already journaled is kept.
  **BREAKING for update callers:** `WorkflowActionReplace` now REQUIRES
  `budget_usd` (send explicit `null`), the same as `max_outstanding_runs`.

- **Workflow-action triggers can cap their own outstanding runs (#2446, part d).**
  New opt-in `WorkflowAction.max_outstanding_runs` (`int >= 1`, default `null` =
  uncapped, same as before). A trigger's `running_since` lease clears when
  `create_run` returns, not when the launched run finishes. So a slow or
  suspended run never stopped the next fire from stacking a second run on top of
  it. With the cap set, a fire where this trigger already has that many
  `pending`/`running`/`suspended` runs launches nothing and records `skipped`
  (reason `outstanding_runs_cap: ...`). That is back-pressure, not an error:
  `consecutive_failures` does not move, so a healthy capped lane is never
  auto-disabled. Runs now record their launching trigger (`wf_runs.trigger_id`,
  migration 0182, with a partial active-runs index). The count happens inside
  `create_run` under the same per-account advisory lock as the launcher and
  account fan-out caps, so concurrent fires of one trigger cannot both pass.
  The field only restricts: it never lifts those caps. It is available on the
  HTTP API and on the `trigger_create`/`trigger_update` tools.
  **BREAKING for update callers:** `WorkflowActionReplace` now REQUIRES
  `max_outstanding_runs` (Replace semantics; send explicit `null` for
  uncapped). A PUT of a workflow action without the key now fails with 422.
  Affected callers: the CLI `sessions triggers update`, `trigger_update` tool
  calls, and SDK builds generated before this change. Add
  `"max_outstanding_runs": null` to keep the old uncapped behavior.

- **Model calls now reserve the model's own output ceiling instead of inheriting
  a provider default that silently truncates replies (#2451).** aios never set
  `max_tokens`, and omitting it does NOT mean "unlimited" — on Anthropic-shaped
  routes it means 4096. Extended-thinking tokens are drawn from that same
  budget, so a hard turn could spend all 4096 reasoning and return EMPTY
  assistant content with `finish_reason: "length"`: full cost billed, recorded
  as a clean turn, no error raised — and the failure got *more* likely the
  harder the task. `harness/completion.py` now defaults `max_tokens` from the
  model's `max_output_tokens`, scoped to Anthropic-shaped routes (OpenAI's
  no-`max_tokens` behaviour is already "as much as fits", and reserving there
  regresses into OpenAI context-window 400s and OpenRouter credit-affordance
  402s). An agent-supplied `max_tokens`/`max_completion_tokens` still wins
  verbatim, and an unknown ceiling omits the key rather than sending
  `max_tokens: None`. One resolver (`completion.resolve_output_cap`) decides
  THE cap for every consumer: the first positive-int value in
  `max_output_tokens > max_tokens > max_completion_tokens` order, else the
  model-ceiling default. The request then carries exactly that one cap key —
  `max_tokens` on Anthropic-shaped routes (LiteLLM passes `max_output_tokens`
  through unrecognised and fills its own `max_tokens`, so a second spelling is a
  second, competing cap), the winning spelling verbatim elsewhere (notably
  `openai/responses/*`). Competing or invalid spellings are dropped and logged
  (`explicit_output_cap_discarded`), and context windowing reserves the same
  value the wire carries.

- Vision capability now treats a missing LiteLLM catalog entry or absent
  `supports_vision` field as unknown and lets image consumers attempt safe
  inline delivery by default. Explicit overrides and catalog booleans remain
  authoritative; image size, decoded-format, and resize limits are unchanged.

- Fix the #2294 schema-diet production incident: the dieted opaque arrays
  rendered as bare `{"type": "array"}` with no `items`, and litellm's
  `token_counter` → `_format_type` dereferences `props['items']`
  unconditionally, so `prelude_overhead_local` raised `KeyError: 'items'` on
  every step of every workflow-capable agent (the fleet was rolled back).
  Opaque arrays now render as `{"type": "array", "items": {}}` — also required
  for OpenAI provider validity. `sanitize_mcp_schema` closes the same defect
  class for untrusted third-party MCP schemas (missing/tuple-form/boolean
  `items`, non-dict property values), and a registry-wide regression fence runs
  the real `token_counter` over every registered tool's rendered schema.

- `get_workflow_script_contract` now returns the authoring contract as a plain-text `ToolResult` instead of a `{"contract": …}` dict, so the ~4KB multi-line manual reaches the model as real prose rather than a JSON-escaped single line (#2294, per the #2291 convention).
- Restore the context window's history floor: `window_min` again bounds
  RETAINED HISTORY only, so the per-request prelude (system prompt + tool
  schemas + reserves) is subtracted from `window_max` alone. Subtracting it
  from both bounds let a fat tool prelude satisfy the floor by itself, driving
  the effective floor to 0 so every snap emptied the window down to a single
  event — the agent then saw only the harness's own "history has scrolled out
  of view, search first" notice and looped on `search_events`. A floor the
  band cannot afford is now clamped to 75% of the events budget (keeping the
  snap chunk usable) and reported on the `read_window_end` span plus a
  `window.floor_clamped` warning, rather than silently zeroed (#2289).

- `search_events` and `memory_search` now return their formatted table as plain multi-line text (`ToolResult`) instead of a `{"result": …}` JSON envelope, so an inline or spilled result stays line-oriented for `grep`/`sed`/`wc -l`/`read` (#2291).
- Cut the model-facing `create_workflow`/`update_workflow`/`call_workflow` tool schemas from ~69KB to ~5.8KB combined by moving the script-authoring contract behind a new on-demand `get_workflow_script_contract` builtin and rendering the declared tool/MCP/HTTP surface as opaque arrays; pydantic validation and the HTTP/SDK schemas are unchanged (#2294).
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
