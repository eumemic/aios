# LiteLLM reasoning controls — cross-provider `reasoning_effort` semantics (source-audited)

> **Purpose.** How thinking/effort knobs flow from an agent's `litellm_extra` through LiteLLM to
> each provider, and the per-provider quirks that aren't visible from any one provider's docs.
> Long-form companion to the `AgentCreate.litellm_extra` field description (which carries only the
> short idiom). Upstream-reference material, **not** aios behaviour — aios sets no reasoning
> defaults and passes `litellm_extra` verbatim (`harness/completion.py::_build_litellm_kwargs`
> spreads it after the harness's timeout defaults, so agent values win).
>
> **Method.** Source audit of the installed LiteLLM — **verified against the aios-pinned
> `litellm==1.91.1` on 2026-07-15**. Symbol names below are stable across minor versions; treat any
> quoted behaviour as
> unverified after a bump until §7's checklist is re-run.

---

## 1. Mental model — two kwarg classes, explicit beats derived

`get_optional_params` (litellm `utils.py`) splits every kwarg into two classes with different
pipelines:

| Class | Membership | Pipeline |
|---|---|---|
| **Standard params** | keys of `DEFAULT_CHAT_COMPLETION_PARAM_VALUES` — includes `reasoning_effort` and `thinking` | translated per-provider inside `map_openai_params` (a `for param in non_default_params` loop) |
| **Provider-specific params** | everything else — includes Anthropic `output_config` | copied **verbatim** into `optional_params` by `add_provider_specific_params_to_optional_params`, which runs **after** the mapping loop (for OpenAI-compatible providers they are nested under `extra_body` instead) |

Because the verbatim copy runs last, **an explicitly-passed provider-native param always overwrites
whatever the mapping loop derived from `reasoning_effort`**. Within the mapping loop itself,
iteration order is `get_optional_params`'s signature order: `reasoning_effort` precedes `thinking`,
so an explicit `thinking` also beats the derived one.

Net precedence (Anthropic path): `output_config` (explicit) > `thinking` (explicit) >
`reasoning_effort` (derived). Everything explicit beats everything derived — the "portable base +
native override" idiom in the `litellm_extra` field description relies on exactly this ordering.

⚠️ This precedence is **Anthropic-path behaviour, not a LiteLLM-wide rule** — the Gemini adapter
turns the same combination into a hard error instead (§5).

## 2. The portable knob and its value-set fidelity

`reasoning_effort` (OpenAI's spelling, adopted by LiteLLM as the cross-provider abstraction) is the
one key that works on all four provider families. Value fidelity varies:

| Value | Anthropic (adaptive models) | OpenAI | xAI (grok-4.3+) | Gemini 3.x |
|---|---|---|---|---|
| `low` / `high` | ✅ faithful | ✅ native | ✅ native | ✅ faithful |
| `medium` | ✅ faithful | ✅ native | ✅ native | ⚠️ silently upgraded to `high` except on `gemini-3.1-pro` / 3.x-flash SKUs |
| `minimal` | mapped to `low` | model-dependent | ❌ | `minimal` on 3.x-flash only, else `low` |
| `xhigh` | ✅ (validated per model-map flags) | ✅ GPT-5.2+ when model metadata flags support (including GPT-5.6); older families reject it | ❌ provider 400 | ❌ client-side `ValueError` |
| `max` | ✅ (validated per model-map flags) | ✅ GPT-5.6 only; older families reject it | ❌ provider 400 | ❌ client-side `ValueError` |
| `none` | strips `thinking` + `output_config` | native on `supports_none` models (GPT-5.1+; model-map gated) | reportedly 4.3+ | lowest level + `includeThoughts: false` — thinking cannot be fully disabled |

The **broad cross-provider subset is `low` / `medium` / `high`**, with the Gemini-3-Pro
`medium`→`high` caveat. For the narrower, currently useful Claude-5 ↔ GPT-5.6 intersection,
`xhigh` and `max` are also portable, subject to the exact Claude and OpenAI model-map entries in
use. They are not safe when repointing to older OpenAI families, xAI, or Gemini.

## 3. Per-provider translation

### Anthropic (`llms/anthropic/chat/transformation.py`)

Branch on `AnthropicConfig._is_adaptive_thinking_model` — sourced **solely** from the bundled model
cost map's `supports_adaptive_thinking` flag:

- **Adaptive models** (all current Claude): `reasoning_effort` → `thinking: {type: "adaptive"}` +
  `output_config: {effort: <REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT[value]>}` (identity table,
  plus `minimal`→`low`). `xhigh`/`max` validated against `supports_{level}_reasoning_effort` map
  flags.
- **Legacy models**: `thinking: {type: "enabled", budget_tokens: N}` with the fixed ladder
  `minimal`/`low`=1024, `medium`=2048, `high`=4096, `xhigh`=8192, `max`=16384 (each overridable via
  `DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET` env vars).

**The map-staleness trap:** a Claude model the installed LiteLLM's map doesn't flag as adaptive
falls into the `budget_tokens` arm, which the live API rejects (400) on Opus 4.7+/Sonnet 5/Fable 5.
A brand-new Claude release outrunning the pinned LiteLLM reproduces this; the fix is a lock bump,
not an aios change.

### OpenAI (`llms/openai/`)

Pure passthrough — `reasoning_effort` is OpenAI's native param. The Responses-API path
(`openai/responses/<model>`, the shape used via oai-proxy) bridges the spelling to
`reasoning: {effort: ...}` with "no mapping applied". GPT-5.6 accepts `none`, `low`, `medium`,
`high`, `xhigh`, and `max`; `max` is new to that family. LiteLLM 1.91.1's bundled map marks
`supports_xhigh_reasoning_effort` for `gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, and
`gpt-5.6-luna`. It has no separate `supports_max_reasoning_effort` metadata field, but `max`
remains provider-native because the Responses adapter passes the value through unchanged.

Older families support subsets: GPT-5.1 and earlier do not support `xhigh`; GPT-5.2 and GPT-5.4
are map-flagged for `xhigh`, but `max` remains GPT-5.6-specific. One interlock: on GPT-5-family
models, `temperature ≠ 1` is only accepted when effort is `none`/omitted on models flagged
`supports_none_reasoning_effort` — LiteLLM enforces this client-side.

### xAI (`llms/xai/chat/transformation.py`)

OpenAI-compatible passthrough, but `reasoning_effort` is only added to the supported-params list
when `litellm.supports_reasoning(model)` says so. The gate exists because the parameter's history
zigzagged: grok-3-mini had it, grok-4 **errored** on it (always-reasoning, no knob), grok-4.1
moved the choice into model identity (`-reasoning` / `-non-reasoning` SKUs), and grok-4.3+
re-adopted the OpenAI-shaped `low`/`medium`/`high` (default `high`).

### Gemini (`llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`)

Branch on `_is_gemini_3_or_newer` — a **string match** (`"gemini-3" in model`), not a map flag:

- **Gemini 3.x**: → `thinkingConfig: {thinkingLevel: ...}` via
  `_map_reasoning_effort_to_thinking_level`. Google's level vocabulary is narrower than the effort
  vocabulary and varies per SKU, hence the `medium`→`high` coercion in §2.
- **Gemini 2.5 and older**: → `thinkingConfig: {thinkingBudget: N}` with its own constant ladder.

## 4. Defaults when the param is omitted (provider-side; aios sends nothing)

| Model family | Thinking | Effort |
|---|---|---|
| Fable 5 / Mythos 5 | always on (adaptive; cannot disable) | `high` |
| Sonnet 5 | adaptive **on** | `high` |
| Opus 4.7 / 4.8 | **off** — must opt in with `thinking: {type: "adaptive"}` | `high` (only shapes thinking if it's on) |
| OpenAI gpt-5 / o-series | reasoning on | `medium` |
| OpenAI `supports_none` models (gpt-5.1-style) | effectively `none` | — |
| grok-4.3+ | reasoning always on | `high` |
| Gemini 3.x | thinking always on (lowest level is the floor) | model default |

Same empty `litellm_extra`, materially different posture per mind — most notably Opus 4.7/4.8
running thinking-less while Sonnet 5/Fable 5 think adaptively.

## 5. Conflict semantics are per-adapter, not global

| Combination | Anthropic | Gemini |
|---|---|---|
| `reasoning_effort` + native depth param (`output_config` / `thinking_level`) | native **overrides** derived (§1 merge order) | hard `UnsupportedParamsError` (400, client-side) |

So the layered idiom (`{"reasoning_effort": "high", "thinking": {"type": "adaptive", "display":
"summarized"}}`) is safe on Anthropic; repointing an agent carrying a native Gemini depth param
plus `reasoning_effort` at a Gemini mind fails — remove, don't layer, there.

## 6. Failure layers

| Bad input | Fails where |
|---|---|
| Unmapped effort value on Anthropic path | client-side `BadRequestError` (LiteLLM validates) |
| `xhigh`/`max` on Gemini path | client-side `ValueError` |
| `xhigh` on OpenAI models without support (GPT-5.1 and earlier) | **provider** 400 (LiteLLM passes through unvalidated) |
| `max` on OpenAI models before GPT-5.6 | **provider** 400 (LiteLLM passes through unvalidated) |
| `xhigh`/`max` on xAI | **provider** 400 (LiteLLM passes through unvalidated) |
| Stale-map Claude model → `budget_tokens` arm | **provider** 400 |

Either way the error surfaces as a model-call failure the session log records — consistent with
the fail-hard stance; nothing in aios pre-validates these.

## 7. Re-verification checklist on any LiteLLM bump

The canonical comments for the three harness workarounds live in `harness/completion.py`; this
list is the audit recipe, not a second source of truth.

1. **Thinking-drop guard** (upstream issue #18926): confirm whether
   `last_assistant_with_tool_calls_has_no_thinking_blocks` is still consulted in
   `litellm.llms.anthropic.chat.transformation` — if it's gone, the module-level monkeypatch in
   `completion.py` can be deleted.
2. **`finish_reason` last-wins**: check `litellm_core_utils/streaming_chunk_builder_utils.py`'s
   assembly loop — if it preserves `content_filter`, the sticky-refusal re-assert in
   `stream_litellm` can be simplified.
3. **Model map currency**: confirm the newest Claude models carry `supports_adaptive_thinking`
   (plus `xhigh`/`max` flags) in `model_prices_and_context_window_backup.json`.
4. **Merge-order invariant** (§1): confirm `add_provider_specific_params_to_optional_params` still
   runs after `map_openai_params` in `get_optional_params`, and `reasoning_effort` still precedes
   `thinking` in its signature — the documented explicit-beats-derived precedence depends on both.
5. Update the "verified against" version in this file's header.
