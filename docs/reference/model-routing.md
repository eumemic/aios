# Model routing authority

AIOS uses LiteLLM as a translation client, but its bundled metadata is not the source of truth for provider behavior.

| LiteLLM metadata/check | AIOS posture | Reason |
|---|---|---|
| Request parameter capability map | **Provider-authoritative passthrough.** Standard OpenAI-shaped parameters in `litellm_extra` are centrally added to `allowed_openai_params`. Agent PUT reports a `Warning` header when the installed map would reject one. | Capability tables inevitably lag new models. The provider's response is authoritative. |
| Model list / model existence | **Provider-authoritative passthrough.** AIOS accepts free-form provider-prefixed model strings. | Providers release and alias models faster than client releases; a local unknown-model rejection is stale by construction. |
| Context-window metadata | **Local fail-fast only for explicit AIOS/provider body ceilings; otherwise provider-authoritative.** Agent `window_min`/`window_max` govern context construction, while provider errors remain authoritative for actual model limits. | Bundled context maps can be stale. Locally enforced wire-size limits are deliberate transport safety constraints, not claims about model capacity. |
| Cost tables | **Local accounting signal, never an inference gate.** Reported provider/LiteLLM request cost is preferred; unmapped cost logs a warning rather than blocking a turn. Account spend limits remain fail-fast because they are an operator policy. | A stale price table must not prevent inference, but explicit local budget policy must remain enforceable. |

## Parameter save warnings

`PUT /v1/agents/{id}` evaluates the merged model and `litellm_extra` against the installed LiteLLM capability map. If a supplied standard parameter is absent from that map, the save still succeeds and returns an RFC 7234 `Warning: 299` header. The worker passes the parameter onward, so the warning provides early visibility without restoring the stale map as execution authority.

Provider-specific kwargs that are not part of LiteLLM's standard OpenAI parameter vocabulary cannot be classified by the map and are passed through without a local warning.
