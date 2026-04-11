"""Unit tests for the output decode/truncate helper in sandbox.container."""

from __future__ import annotations

from aios.sandbox.container import _decode_and_truncate


class TestDecodeAndTruncate:
    def test_short_output_not_truncated(self) -> None:
        raw = b"hello world"
        text, truncated = _decode_and_truncate(raw, max_bytes=100)
        assert text == "hello world"
        assert truncated is False

    def test_output_exactly_at_limit_not_truncated(self) -> None:
        raw = b"x" * 100
        text, truncated = _decode_and_truncate(raw, max_bytes=100)
        assert text == "x" * 100
        assert truncated is False

    def test_output_over_limit_is_truncated(self) -> None:
        raw = b"x" * 1000
        text, truncated = _decode_and_truncate(raw, max_bytes=100)
        assert truncated is True
        assert text.startswith("x" * 100)
        assert "[output truncated]" in text

    def test_invalid_utf8_replaced_not_raised(self) -> None:
        # A lone continuation byte is invalid UTF-8; decode('replace')
        # substitutes U+FFFD rather than raising.
        raw = b"hello \xff world"
        text, truncated = _decode_and_truncate(raw, max_bytes=1000)
        assert truncated is False
        assert "hello" in text
        assert "world" in text
        assert "\ufffd" in text

    def test_truncation_on_byte_boundary_safe_for_utf8(self) -> None:
        # Truncating a 3-byte UTF-8 char in the middle should still decode
        # via the 'replace' error handler, not raise.
        raw = "héllo".encode() * 100
        text, truncated = _decode_and_truncate(raw, max_bytes=50)
        assert truncated is True
        # The result should contain at least the first "héllo"s.
        assert "h" in text
