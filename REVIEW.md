# Uncorrelated review — aios#2453 (OpenRouter `max_tokens` exclusion)

- **Tip reviewed:** `9697f1a9` (`fix/2451-default-max-tokens`), diffed against `b55ff619`
- **Reviewer:** independent checker (Claude Opus 5). Implementer was gpt-5.6-sol.
- **Fix commit landed in this worktree (not pushed):** `56661aeb`
- **Verdict:** **FAIL** at `9697f1a9` — two Medium defects. Both fixed in `56661aeb`; the tree at `56661aeb` passes.

## Scope check — the brief's four asks

| Ask | At `9697f1a9` |
|---|---|
| 1. OpenRouter must not get the Anthropic ceiling; early-return leaving `max_tokens` unset | Met for `openrouter/`-prefixed models; **incomplete** — see F1 |
| 2. Direct `anthropic/*` still gets the ceiling | **Met.** Not weakened |
| 3. Unit tests assert both sides | Both sides asserted, but **two pre-existing tests went vacuous** — see F2 |
| 4. Smallest correct change; ruff/typing clean | Change is minimal and clean; docstring left contradicting the code — see F3 |

## Findings

### F1 — Medium. Exclusion keyed on the model prefix only; `custom_llm_provider` bypasses it

`_apply_default_max_tokens` tested `model.startswith("openrouter/")`. But
`custom_llm_provider` outranks the model string in LiteLLM's dispatch, and this
codebase *already knows that* — `services/model_providers.py:_derive_provider`
threads it into `get_llm_provider` precisely so "the row this function looks up
must match what LiteLLM will actually call, not just the bare model string." It
is also an allow-listed control param (`services/litellm_params.py:14`).

Verified against the pinned litellm:

```
litellm.get_llm_provider("anthropic/claude-opus-4-1")                                -> "anthropic"
litellm.get_llm_provider("anthropic/claude-opus-4-1", custom_llm_provider="openrouter") -> "openrouter"
model_descriptor("anthropic/claude-opus-4-1")  -> cache_channel=ANTHROPIC
```

So an agent with `litellm_extra={"custom_llm_provider": "openrouter"}` reaches
OpenRouter, the prefix test does not fire, `model_descriptor` reads ANTHROPIC,
the full 32K/64K ceiling is injected — and the call 402s. That is the exact P1
failure mode this fixround exists to close, surviving on a routing path the
codebase explicitly supports.

**Fixed:** the guard now also tests `kwargs.get("custom_llm_provider")`. Free —
the caller's `litellm_extra` is merged into `kwargs` before this runs (call site
`completion.py:803`), so no signature change. New test pins it.

### F2 — Medium. Two precedence tests became vacuous; the "caller value wins" rule went untested

`test_explicit_caller_max_tokens_wins_verbatim` ("REQUIRED TEST 2") and
`test_explicit_max_completion_tokens_also_suppresses_the_default` still ran on
`_PROXY_CLAUDE_MODEL` (OpenRouter). The new early return satisfies both
assertions on its own, so neither test can observe the guard it names.

Confirmed by mutation at `9697f1a9` — deleting the caller-precedence guard
**outright**:

```
    if "max_tokens" in kwargs or "max_completion_tokens" in kwargs:
        return          # <- both lines removed
=> 12 passed
```

A guard whose removal is invisible to the suite is not covered. This matters
concretely: the brief's own framing is that ~52 agents carry hand-applied
per-model `max_tokens` values that the central default must never overwrite.

**Fixed:** both moved to `_DIRECT_CLAUDE_MODEL` — the only route where the
default would otherwise fire. The same mutation now fails 2 tests. A module
docstring note records *why* route choice is load-bearing here, so the tests do
not drift back.

### F3 — Medium (docs, repo convention). Docstring contradicted the code

`_apply_default_max_tokens` still asserted it was "**Scoped to Anthropic-shaped
routes** (the same gate `model_descriptor` uses for cache markers)" — no longer
true — and named the OpenRouter 402 only as a reason to skip *OpenAI-shaped*
routes, which reads as an argument *against* the change actually shipped. The
sole rationale for the new early return lived in a test module docstring, and
the return itself had no comment (the branch directly below it has three lines).
In a file where docstrings are the design record, that is a broken window.

**Fixed:** dedicated OpenRouter paragraph stating the exclusion, why (prices
against the reservation, not usage), why it is a no-op rather than a regression
(the route already sent no `max_tokens`), and the residual trade it accepts
(silent truncation on OpenRouter stays, preferred over 402-ing every call).

### Observations — no action

- **Ordering.** The OpenRouter return sits above the `CacheChannel` check. No
  behavioral difference (`openrouter/openai/*` is `OPENAI`, already excluded);
  reads fine as "this provider is special regardless of channel."
- **`bedrock/anthropic.*`** newly receives the ceiling (from `b55ff619`, not
  this diff). Bedrock has no credit-affordance 402, so not a defect here.
- **Coverage gap, minor.** No test for OpenRouter *without* thinking. The code
  has no thinking branch, so this cannot hide a defect. Left alone.
- **Lost coverage, benign.** `test_default_applies_without_thinking_too` was
  removed; its no-thinking arm is preserved by the new direct-route test
  (`params=None`).

## Commands run

All from the worktree root.

| Command | Result |
|---|---|
| `uv run pytest tests/unit/test_completion_max_tokens.py -q` (at `9697f1a9`) | 12 passed |
| Mutation A — caller-precedence guard deleted (at `9697f1a9`) | **12 passed — defect F2** |
| `uv run pytest tests/unit/test_completion_max_tokens.py -q` (at `56661aeb`) | 13 passed |
| Mutation A — caller-precedence guard deleted (at `56661aeb`) | 2 failed, 11 passed |
| Mutation B — whole OpenRouter exclusion deleted (at `56661aeb`) | 2 failed, 11 passed |
| Mutation C — `custom_llm_provider` arm dropped, i.e. the `9697f1a9` shape | 1 failed, 12 passed |
| `uv run pytest tests/unit/test_completion_*.py tests/unit/test_model_binding.py tests/unit/test_run_llm.py tests/unit/harness -q` | 151 passed |
| `uv run pytest tests/unit/test_context_admission.py tests/unit/test_context_budget.py tests/unit/test_litellm_param_validation.py tests/unit/test_model_providers_service.py -q` (+ max_tokens) | 74 passed |
| `uv run ruff check` / `ruff format --check` on both touched files | clean / already formatted |
| `uv run mypy src/aios/harness/completion.py tests/unit/test_completion_max_tokens.py` | Success, no issues |

Mutants were applied to a scratch copy and reverted; `diff` against the restored
file confirmed clean before committing.

## Not done, per brief

No push, no PR open/update, no merge. `56661aeb` sits local in this worktree.
