# Changelog

## Unreleased

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
