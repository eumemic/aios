from __future__ import annotations

import json

from aios.harness.request_body_budget import (
    ANTHROPIC_BODY_BUDGET_BYTES,
    ANTHROPIC_MAX_MEDIA_PER_REQUEST,
    MEDIA_OMITTED_TEXT,
    BodyLimits,
    body_limits_for_model,
    enforce_request_body_budget,
    is_request_too_large_error,
)

ANTHROPIC_LIMITS = body_limits_for_model("anthropic/claude-test")


def _image(payload: str) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{payload}"},
    }


def test_evicts_oldest_media_before_text_and_preserves_thinking() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "keep-old"}, _image("a" * 80)]},
        {
            "role": "assistant",
            "content": "keep-new",
            "thinking_blocks": [{"thinking": "reason", "signature": "signed"}],
        },
        {"role": "user", "content": [_image("b" * 80)]},
    ]

    result = enforce_request_body_budget(
        {"model": "anthropic/claude-test", "messages": messages},
        limits=BodyLimits(byte_budget=500, max_media=ANTHROPIC_MAX_MEDIA_PER_REQUEST),
    )

    assert result.media_removed >= 1
    assert json.dumps(messages).find(MEDIA_OMITTED_TEXT) >= 0
    assert "keep-old" in json.dumps(messages)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    assert assistant.get("thinking_blocks") == [{"thinking": "reason", "signature": "signed"}]
    assert result.request_bytes <= 500


def test_caps_media_count_for_anthropic_evicting_oldest() -> None:
    messages = [{"role": "user", "content": [_image(str(i)) for i in range(101)]}]
    result = enforce_request_body_budget(
        {"model": "anthropic/claude-test", "messages": messages}, limits=ANTHROPIC_LIMITS
    )
    assert result.media_removed == 1
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 101
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_media_count_cap_is_not_applied_to_non_anthropic_providers() -> None:
    """#2007 review: the 100-image cap is Anthropic's, not universal.

    An OpenAI-compatible model does not share Anthropic's per-request media
    limit, so a payload with more than 100 images must reach the provider
    untouched rather than silently losing history.
    """
    messages = [{"role": "user", "content": [_image(str(i)) for i in range(150)]}]
    result = enforce_request_body_budget(
        {"model": "openai/test", "messages": messages}, limits=body_limits_for_model("openai/test")
    )
    assert result.media_removed == 0
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert all(part["type"] == "image_url" for part in content)
    assert MEDIA_OMITTED_TEXT not in json.dumps(messages)


def test_non_anthropic_models_get_no_proactive_limits() -> None:
    for model in ("openai/test", "gpt-5", "openrouter/openai/gpt-5", "qwen2.5-coder"):
        limits = body_limits_for_model(model)
        assert limits.byte_budget is None, model
        assert limits.max_media is None, model


def test_anthropic_routes_get_both_limits() -> None:
    for model in (
        "anthropic/claude-opus-5",
        "claude-fable-5",
        "openrouter/anthropic/claude-opus-5",
        "bedrock/anthropic.claude-v2",
        "vertex_ai/claude-opus-5",
    ):
        limits = body_limits_for_model(model)
        assert limits.byte_budget == ANTHROPIC_BODY_BUDGET_BYTES, model
        assert limits.max_media == ANTHROPIC_MAX_MEDIA_PER_REQUEST, model


def test_no_limits_means_no_trimming_even_for_a_huge_body() -> None:
    """Absent limits, a large body is left alone (no byte-budget fallback)."""
    messages = [{"role": "user", "content": [_image("a" * 5000) for _ in range(20)]}]
    payload = {"model": "openai/test", "messages": messages}
    result = enforce_request_body_budget(payload, limits=None)
    assert result.media_removed == 0
    assert MEDIA_OMITTED_TEXT not in json.dumps(messages)


def test_byte_budget_alone_trims_without_a_media_cap() -> None:
    """A provider with a byte ceiling but no media cap still gets byte-trimmed."""
    messages = [{"role": "user", "content": [_image("a" * 200) for _ in range(5)]}]
    result = enforce_request_body_budget(
        {"model": "x/y", "messages": messages}, limits=BodyLimits(byte_budget=600, max_media=None)
    )
    assert result.media_removed >= 1
    assert result.request_bytes <= 600


def test_strip_all_media_is_provider_agnostic_reactive_path() -> None:
    """The reactive retry answers the provider's own 413 — it ignores scoping."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "t"}, _image("a" * 50)]}]
    result = enforce_request_body_budget(
        {"model": "openai/test", "messages": messages}, strip_all_media=True
    )
    assert result.media_removed == 1
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "t"}
    assert content[1]["text"] == MEDIA_OMITTED_TEXT


def test_anthropic_budget_has_margin_under_32mb() -> None:
    assert 24 * 1024 * 1024 <= ANTHROPIC_BODY_BUDGET_BYTES <= 27 * 1024 * 1024


def test_recognizes_provider_size_signatures() -> None:
    assert is_request_too_large_error(
        Exception("request_too_large: Request exceeds the maximum size")
    )
    exc = Exception("payload rejected")
    exc.status_code = 413  # type: ignore[attr-defined]
    assert is_request_too_large_error(exc)
    assert not is_request_too_large_error(Exception("invalid api key"))
