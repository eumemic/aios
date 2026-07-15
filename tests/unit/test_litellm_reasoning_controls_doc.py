from __future__ import annotations

from pathlib import Path

import litellm
import pytest

DOC = Path(__file__).resolve().parents[2] / "docs/reference/litellm-reasoning-controls.md"
GPT_56_MODELS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")


@pytest.mark.parametrize("model", GPT_56_MODELS)
def test_reasoning_controls_doc_tracks_gpt_56_xhigh_model_map(model: str) -> None:
    metadata = litellm.model_cost[model]
    assert metadata["supports_xhigh_reasoning_effort"] is True

    doc = DOC.read_text(encoding="utf-8")
    assert f"`{model}`" in doc


def test_reasoning_controls_doc_records_gpt_56_native_max_support() -> None:
    doc = DOC.read_text(encoding="utf-8")
    assert "GPT-5.6" in doc
    assert "`xhigh` and `max`" in doc
    assert "| `xhigh` / `max` |" not in doc
