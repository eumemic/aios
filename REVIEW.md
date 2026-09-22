# Uncorrelated review — aios#2453 / `ormax2453c`

- **Tip reviewed:** `739d42b3` ("fix(harness): reserve output against the provider ceiling, not window_max")
- **Diff range:** `a1412967..739d42b3`
- **Branch in this worktree:** `ormax2453crev` (review fix commit `c358e6fe` on top; **not pushed**)
- **Verdict: PASS** (with one Medium risk documented below that is upstream-data, not logic)

---

## 1. The blocking P1 is genuinely fixed

`effective_window_max` no longer treats `window_max` as a total context budget.
The removed line was:

```python
input_cap = window_max if output_reserve is None else max(1, window_max - reservation)
```

and the unknown-ceiling branch now returns `max(1, int(window_max * shrink_factor))`.
The reservation is subtracted only from a ceiling, via `min(window_max, max(1, ceiling - reservation))`,
where `ceiling = served_ceiling(model) or context_limit`.

Verified against the live LiteLLM catalog with `window_max=150_000` (the model default,
`src/aios/models/agents.py:1039`):

| model | reserve | limit | budget @739d42b3 | budget @a1412967 |
|---|---|---|---|---|
| `anthropic/claude-opus-4-5` | 64000 | 200000 | **136000** | 86000 |
| `anthropic/claude-opus-4-6` | 128000 | 1000000 | **150000** | 22000 |
| `anthropic/claude-opus-4-1` | 32000 | 200000 | **150000** | 118000 |
| `openai/gpt-4.1` | None | None | 150000 | 150000 |
| `openrouter/anthropic/claude-opus-4-1` | 0 | None | 200000 | 200000 |

`claude-opus-4-5` lands on exactly the `min(150000, 200000-64000) = 136000` the brief specifies.
`claude-opus-4-6` — the repo's own canonical model (`src/aios/cli/commands/init.py:38`,
`src/aios/models/agents.py:1006`) — was the worst case of the P1: a **22k** usable input
window. It is now 150000.

## 2. Checklist walkthrough

1. **No `window_max - reserve`.** Confirmed, line removed; the only subtraction is `ceiling - reservation`. ✔
2. **Known ceiling ⇒ `min(window_max, ceiling - reserve)`.** `context_budget.py:81`. ✔
3. **Unknown ceiling ⇒ `window_max` preserved.** `context_budget.py:80`. The ceiling comes from
   existing catalog helpers (`litellm.get_model_info` → `max_input_tokens`), mirroring
   `default_max_output_tokens` exactly — same source, same `@cache`, same `None`-is-a-real-answer
   stance. No invented map. ✔
4. **OpenRouter unchanged.** `resolved_context_limit` is gated on `_uses_anthropic_max_tokens_default`,
   which already excludes both the `openrouter/` prefix and `custom_llm_provider="openrouter"`,
   so OpenRouter resolves `limit=None` and keeps its `window_max`-only budget. Reservation
   behavior untouched. ✔
5. **Wire injection / caller precedence intact.** `_apply_default_max_tokens`,
   `_normalize_explicit_output_cap` and `resolved_output_reservation` are unmodified in this
   diff; `tests/unit/test_completion_max_tokens.py` (incl. the real-LiteLLM wire-body test)
   passes untouched. Explicit `max_tokens` / `max_output_tokens` still win
   (`limit - 1234` asserted for both spellings). ✔
6. **Test coverage.** Good, and notably the tests now assert *relative* to the resolvers rather
   than pinning catalog literals — correct, since unit tests run egress-blocked against the
   bundled backup map while production fetches the live map, and the two disagree. One real
   gap found and closed (§3). ✔
7. **Ruff / mypy clean.** ✔

**Gate agreement (checked, since a drift here is silent):** `resolved_output_reservation` and
`resolved_context_limit` return non-`None` on exactly the same routes, because both funnel
through `_uses_anthropic_max_tokens_default` / the `CacheChannel.ANTHROPIC` gate and both read
the *same* `get_model_info` entry. A model LiteLLM cannot resolve yields `None` from both, so
there is no "reserve against a limit that doesn't exist" state.

## 3. Finding — test gap on the exact changed branch (Medium) — **FIXED** in `c358e6fe`

739d42b3 changed precisely one branch: positive `output_reserve` + unknown ceiling. **Nothing
in the suite covered that combination.** Every Anthropic test resolves a ceiling, and both
OpenRouter tests resolve a *zero* reservation — and `window_max - 0 == window_max`, so
reinstating `max(1, window_max - reservation)` would have left the entire file green. Given
this formula has now been wrong once already (13e8d003), that is worth a pin.

The live instance is OpenRouter with a caller cap:

```
openrouter/anthropic/claude-opus-4-1  {"max_tokens": 8000}
  -> reserve=8000  limit=None  served=None  budget=150000   (was 142000 @a1412967)
```

Added `test_nonzero_reservation_without_a_ceiling_preserves_window_max`, asserting the budget
passes through verbatim and that the overflow shrink ladder still tightens it.

## 4. Finding — stale catalog ceiling on `claude-sonnet-4-5` (Medium, upstream data) — **reported, not patched**

Not a logic defect, but it should be on the record because it is the failure mode this PR's
reservation exists to prevent.

LiteLLM's **live** map reports `claude-sonnet-4-5: max_input_tokens=1000000` (the retired 1M
beta), while Anthropic retired `context-1m-2025-08-07` for Sonnet 4/4.5 on 2026-04-30 — Sonnet
4.5's real window is 200k, and 1M went GA on the `-4-6` models instead. So:

```
anthropic/claude-sonnet-4-5  ->  reserve=64000  limit=1000000  budget=min(150000, 936000)=150000
wire: input ≲150000 + max_tokens 64000 = ~214000  >  the real 200000  ->  provider 400
```

Under a1412967's (wrong-but-conservative) formula this happened to be safe at 86000. The
breakage starts once history exceeds ~136k tokens.

**Why this is not a blocker and not patched here:**
- The data is wrong upstream, not the derivation. The bundled backup map has the correct
  `200000`; only the fetched live map is stale, and it self-corrects when LiteLLM updates.
- `_SERVED_CEILINGS` already exists as the sanctioned override for exactly this
  ("Served ceilings can differ materially from public model-card context windows"), and
  `served_ceiling` correctly outranks `context_limit`. The escape hatch is in place.
- Exposure in this repo is an eval baseline (`evals/wam_fusion/run_eval.py:68`) plus two unit
  fixtures. Production agents use `claude-opus-4-6`, whose 1M figure is correct.
- Patching it would mean hardcoding model strings into `_SERVED_CEILINGS`, which is exact-string
  keyed and so cannot cover `claude-sonnet-4-5` / `anthropic/...` / `...-20250929` /
  `us.anthropic....` / `vertex_ai/...` without whack-a-mole — against CLAUDE.md's
  "extreme simplicity, no defensive guards" and out of this fixround's scope.

**Recommended follow-up (separate issue):** add `_SERVED_CEILINGS` entries (or a normalizing
lookup) if Sonnet 4.5 is ever promoted to a production route before LiteLLM corrects the entry.

## 5. Minor notes (no action)

- `min(window_max, max(1, ceiling - reservation))` floors at 1 when a caller cap exceeds the
  model ceiling (e.g. `max_tokens=500000` on a 200k model). Budget 1 ⇒ `read_windowed_events`
  raises "no budget remains for events". That request would be rejected by the provider anyway,
  and fail-hard is the house stance; pre-existing on the `served_ceiling` path.
- `model_context_limit`'s docstring cross-references `_apply_default_max_tokens` — verified
  present at `completion.py:566`.
- Only one production call site of `effective_window_max` (`loop.py:1042`); it passes both
  resolvers. Grepped the full tree for the new symbols — no stragglers.

## 6. Commands run

| command | result |
|---|---|
| `uv run pytest tests/unit/test_context_budget.py tests/unit/test_completion_max_tokens.py -q` | **31 passed** (30 before the added test) |
| `uv run pytest tests/unit -q -n 4 -x -k "context or window or completion or budget or token"` | **871 passed** |
| `uv run ruff check <4 touched files>` | All checks passed |
| `uv run ruff format --check <4 touched files>` | already formatted |
| `uv run mypy <4 touched files>` | Success: no issues found |

Live-catalog probes were run through `resolved_output_reservation` / `resolved_context_limit` /
`effective_window_max` directly to produce the table in §1.

## 7. Fix commits in this worktree (not pushed)

- `c358e6fe` — `test(harness): pin window_max preservation when a reservation has no ceiling`

---

VERDICT: PASS
