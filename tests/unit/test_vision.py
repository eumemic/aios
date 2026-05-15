"""Unit coverage for :mod:`aios.harness.vision`.

Pure decision logic — no I/O.  ``supports_vision`` delegates to
``litellm.get_model_info``, which we monkeypatch to keep tests
hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.harness import vision


@pytest.fixture(autouse=True)
def _clear_vision_overrides() -> Any:
    """Wipe the override dict between tests so order doesn't matter."""
    saved = dict(vision._VISION_OVERRIDES)
    vision._VISION_OVERRIDES.clear()
    yield
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES.update(saved)


def _patch_get_model_info(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, dict[str, Any]]
) -> None:
    def fake(model: str, **kwargs: Any) -> dict[str, Any]:
        if model not in mapping:
            raise Exception(f"unknown model: {model}")
        return mapping[model]

    monkeypatch.setattr("aios.harness.vision.litellm.get_model_info", fake)


class TestSupportsVision:
    def test_true_when_litellm_says_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"foo/vision": {"supports_vision": True}})
        assert vision.supports_vision("foo/vision") is True

    def test_false_when_litellm_says_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"foo/text": {"supports_vision": False}})
        assert vision.supports_vision("foo/text") is False

    def test_false_when_litellm_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {})  # any model raises
        assert vision.supports_vision("totally/unknown") is False

    def test_litellm_exception_emits_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare exception path must surface at warn-level so operators can
        grep when vision degrades across a deploy / provider blip."""
        _patch_get_model_info(monkeypatch, {})  # any model raises

        warned: list[tuple[str, dict[str, Any]]] = []

        class _Recorder:
            def warning(self, event: str, **kwargs: Any) -> None:
                warned.append((event, kwargs))

        monkeypatch.setattr("aios.harness.vision.log", _Recorder())
        assert vision.supports_vision("totally/unknown") is False
        assert len(warned) == 1
        event, kwargs = warned[0]
        assert event == "vision.litellm_lookup_failed"
        assert kwargs["model"] == "totally/unknown"
        assert "error" in kwargs

    def test_override_wins_over_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even if litellm reports True, an explicit False override takes effect.
        _patch_get_model_info(monkeypatch, {"foo/vision": {"supports_vision": True}})
        vision._VISION_OVERRIDES["foo/vision"] = False
        assert vision.supports_vision("foo/vision") is False


class TestCanInlineImage:
    def test_image_under_cap_with_vision_mind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"vision/m": {"supports_vision": True}})
        assert vision.can_inline_image(
            model="vision/m", content_type="image/jpeg", size_bytes=500_000
        )

    def test_image_at_exact_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"vision/m": {"supports_vision": True}})
        assert vision.can_inline_image(
            model="vision/m",
            content_type="image/png",
            size_bytes=vision.INLINE_SIZE_CAP_BYTES,
        )

    def test_image_over_cap_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"vision/m": {"supports_vision": True}})
        assert not vision.can_inline_image(
            model="vision/m",
            content_type="image/jpeg",
            size_bytes=vision.INLINE_SIZE_CAP_BYTES + 1,
        )

    def test_non_vision_mind_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"text/m": {"supports_vision": False}})
        assert not vision.can_inline_image(
            model="text/m", content_type="image/jpeg", size_bytes=100
        )

    def test_non_image_blocked_even_with_vision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"vision/m": {"supports_vision": True}})
        assert not vision.can_inline_image(
            model="vision/m", content_type="application/pdf", size_bytes=100
        )

    def test_audio_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_get_model_info(monkeypatch, {"vision/m": {"supports_vision": True}})
        assert not vision.can_inline_image(
            model="vision/m", content_type="audio/ogg", size_bytes=100
        )


class TestMakeImageUrlPart:
    def test_shape(self) -> None:
        part = vision.make_image_url_part(content_type="image/png", data_b64="ZmFrZQ==")
        assert part == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
        }

    def test_self_corrects_mismatched_mime(self) -> None:
        """Single construction point reconciles declared mime against the
        bytes' magic.  Covers every caller (renderer, read tool, future)
        without each having to remember to wire correction.
        """
        import base64 as _b64

        jpeg_b64 = _b64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode("ascii")
        part = vision.make_image_url_part(content_type="image/png", data_b64=jpeg_b64)
        assert part["image_url"]["url"] == f"data:image/jpeg;base64,{jpeg_b64}"

    def test_keeps_declared_mime_when_unrecognized(self) -> None:
        """Sniff returns None on random/unknown bytes — declared mime wins."""
        part = vision.make_image_url_part(content_type="image/png", data_b64="ZmFrZQ==")
        assert part["image_url"]["url"].startswith("data:image/png;base64,")


class TestTextMarker:
    def test_image_with_path(self) -> None:
        record = {
            "filename": "photo.jpg",
            "content_type": "image/jpeg",
            "size": 524_288,
            "in_sandbox_path": "/mnt/attachments/echo/evt-1-photo.jpg",
        }
        assert vision.text_marker(record) == (
            "[image: photo.jpg (image/jpeg, 512.0KB) at /mnt/attachments/echo/evt-1-photo.jpg]"
        )

    def test_non_image_falls_back_to_attachment(self) -> None:
        record = {
            "filename": "doc.pdf",
            "content_type": "application/pdf",
            "size": 1024,
            "in_sandbox_path": "/mnt/attachments/sig/x-doc.pdf",
        }
        assert "attachment: doc.pdf" in vision.text_marker(record)

    def test_legacy_stub_no_path(self) -> None:
        record = {"filename": "old.jpg", "content_type": "image/jpeg", "size": 100}
        marker = vision.text_marker(record)
        assert "old.jpg" in marker
        assert "at " not in marker
