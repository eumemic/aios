from __future__ import annotations

import json

from aios.harness.request_body_budget import (
    ANTHROPIC_BODY_BUDGET_BYTES,
    MEDIA_OMITTED_TEXT,
    enforce_request_body_budget,
    is_request_too_large_error,
)


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
        {"model": "anthropic/claude-test", "messages": messages}, byte_budget=500
    )

    assert result.media_removed >= 1
    assert json.dumps(messages).find(MEDIA_OMITTED_TEXT) >= 0
    assert "keep-old" in json.dumps(messages)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    assert assistant.get("thinking_blocks") == [{"thinking": "reason", "signature": "signed"}]
    assert result.request_bytes <= 500


def test_caps_media_count_at_100_evicting_oldest() -> None:
    messages = [{"role": "user", "content": [_image(str(i)) for i in range(101)]}]
    result = enforce_request_body_budget(
        {"model": "openai/test", "messages": messages}, byte_budget=1_000_000
    )
    assert result.media_removed == 1
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 101
    assert content[0]["type"] == "text"


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
