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
