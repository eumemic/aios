"""Tests for the claude-fable-5 ``litellm.model_cost`` registration.

litellm 1.83.4 predates claude-fable-5, so its model_cost map ships no entry
and ``response_cost`` comes back None — every fable-5 turn would record
``cost_usd=null``. Importing ``aios.harness.completion`` registers pricing +
capabilities explicitly (both the bare id and the ``anthropic/``-prefixed
routing id), so cost extraction works.
"""

from __future__ import annotations

import litellm

# Import for the registration side-effect at module import time.
import aios.harness.completion  # noqa: F401


class TestFable5ModelCostRegistration:
    def test_bare_id_registered_with_pricing(self) -> None:
        entry = litellm.model_cost["claude-fable-5"]
        assert entry["input_cost_per_token"] == 10e-6
        assert entry["output_cost_per_token"] == 50e-6

    def test_prefixed_id_registered_with_pricing(self) -> None:
        entry = litellm.model_cost["anthropic/claude-fable-5"]
        assert entry["input_cost_per_token"] == 10e-6
        assert entry["output_cost_per_token"] == 50e-6

    def test_capabilities_registered(self) -> None:
        entry = litellm.model_cost["claude-fable-5"]
        assert entry["litellm_provider"] == "anthropic"
        assert entry["mode"] == "chat"
        assert entry["supports_reasoning"] is True
        assert entry["supports_function_calling"] is True
        assert entry["supports_prompt_caching"] is True
