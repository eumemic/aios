"""LiteLLM capability-map diagnostics without making the map authoritative."""

from __future__ import annotations

from typing import Any

import litellm
from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES

# These configure LiteLLM itself rather than becoming provider request parameters.
_LITELLM_CONTROL_PARAMS = {
    "allowed_openai_params",
    "additional_drop_params",
    "custom_llm_provider",
    "drop_params",
    "max_retries",
}
OPENAI_REQUEST_PARAMS = frozenset(DEFAULT_CHAT_COMPLETION_PARAM_VALUES) - _LITELLM_CONTROL_PARAMS


def openai_params_in(extra: dict[str, Any] | None) -> set[str]:
    """Return standard OpenAI-shaped request params present in agent extras."""
    return set(extra or {}) & OPENAI_REQUEST_PARAMS


def unsupported_openai_params(model: str, extra: dict[str, Any] | None) -> list[str]:
    """Report params rejected by LiteLLM's current model map.

    This is advisory. The worker centrally permits these parameters so the provider,
    rather than a potentially stale bundled model map, makes the final decision.
    Provider-specific (non-standard) kwargs cannot be classified by this map and are
    intentionally omitted.
    """
    supplied = openai_params_in(extra)
    if not supplied:
        return []
    supported = set(litellm.get_supported_openai_params(model) or [])
    explicitly_allowed = set((extra or {}).get("allowed_openai_params") or [])
    return sorted(supplied - supported - explicitly_allowed)


# LiteLLM control params that make it discard request params and complete anyway.
# ``drop_params`` discards every param the provider map rejects; ``additional_drop_params``
# removes named params in ``_get_non_default_params`` BEFORE provider validation runs, so
# no error is ever raised (verified against litellm 1.96.2). Both produce the #1674 outcome:
# the caller is told the call succeeded while a treatment never reached the wire.
# ``allowed_openai_params`` is NOT here: it widens what is passed through, and a rejected
# value still raises (verified).
SILENT_DROP_CONTROL_PARAMS = frozenset({"drop_params", "additional_drop_params"})


def silent_drop_controls_in(extra: dict[str, Any] | None) -> list[str]:
    """Return silent-drop controls in ``extra`` that would actually discard a param.

    A control that cannot cause a drop is not reported: ``drop_params: False`` agrees
    with the harness, and an empty ``additional_drop_params`` list drops nothing. Only
    values that would make LiteLLM discard a param and still return success are named.
    """
    supplied = extra or {}
    offenders = []
    for name in sorted(SILENT_DROP_CONTROL_PARAMS & set(supplied)):
        value = supplied[name]
        if name == "drop_params" and value is not True:
            continue
        if name == "additional_drop_params" and not value:
            continue
        offenders.append(name)
    return offenders
