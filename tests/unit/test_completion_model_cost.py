"""Tests for claude-fable-5 cost resolution via litellm's native model_cost map.

The pinned litellm ships a native ``claude-fable-5`` entry in its
bundled model-cost map whose pricing/capability fields are identical to the
hand-typed dict the old ``_FABLE5_MODEL_COST`` shim used to ``setdefault`` (#1561
deleted that inert block). These tests pin the behavior the shim used to
guarantee: cost extraction for a fable-5 turn yields a non-null cost from the
native entry, resolved through both the bare id and the ``anthropic/``-prefixed
routing id (litellm strips the provider prefix and falls back to the bare key).
"""

from __future__ import annotations

import litellm

# Import so the rest of the module's import-time setup (e.g. modify_params) runs;
# the fable-5 cost itself now comes from litellm's native map, not a local shim.
import aios.harness.completion  # noqa: F401
from aios.harness.completion import estimate_cost_usd


class TestFable5NativeModelCost:
    def test_native_bare_id_entry_present(self) -> None:
        entry = litellm.model_cost["claude-fable-5"]
        assert entry["input_cost_per_token"] == 1e-05
        assert entry["output_cost_per_token"] == 5e-05
        assert entry["litellm_provider"] == "anthropic"
        assert entry["mode"] == "chat"
        assert entry["supports_reasoning"] is True
        assert entry["supports_function_calling"] is True
        assert entry["supports_prompt_caching"] is True
        assert entry["max_input_tokens"] == 1_000_000
        assert entry["max_output_tokens"] == 128_000

    def test_bare_id_cost_is_non_null(self) -> None:
        cost = estimate_cost_usd("claude-fable-5", {"input_tokens": 1000, "output_tokens": 1000})
        assert cost is not None
        assert cost > 0

    def test_prefixed_id_cost_is_non_null(self) -> None:
        # litellm strips the ``anthropic/`` prefix and falls back to the bare
        # key, so the prefixed routing id resolves the same native entry.
        cost = estimate_cost_usd(
            "anthropic/claude-fable-5", {"input_tokens": 1000, "output_tokens": 1000}
        )
        assert cost is not None
        assert cost > 0

    def test_prefixed_and_bare_costs_match(self) -> None:
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        assert estimate_cost_usd("claude-fable-5", usage) == estimate_cost_usd(
            "anthropic/claude-fable-5", usage
        )
