"""Magic-byte image-mime sniffer (#342).

Inbound platform metadata (Signal, Telegram) and extension-based
guesses both occasionally lie about an image's true mime type;
Anthropic's ``/v1/messages`` validates declared mime against actual
bytes and 400s on mismatch.  Sniff at the SDK boundary so every
inbound :class:`Attachment` carries a correct ``content_type`` before
the event is persisted.
"""

from __future__ import annotations

from aios_connector_http.mime import sniff_image_mime

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff\xe0"
GIF87_MAGIC = b"GIF87a"
GIF89_MAGIC = b"GIF89a"
# RIFF chunk id + a 4-byte size + the WEBP form type at offset 8.
WEBP_MAGIC = b"RIFF\x24\x58\x00\x00WEBP"


class TestSniffImageMime:
    def test_png(self) -> None:
        assert sniff_image_mime(PNG_MAGIC + b"trailing-bytes") == "image/png"

    def test_jpeg(self) -> None:
        assert sniff_image_mime(JPEG_MAGIC + b"trailing-bytes") == "image/jpeg"

    def test_gif87a(self) -> None:
        assert sniff_image_mime(GIF87_MAGIC + b"trailing-bytes") == "image/gif"

    def test_gif89a(self) -> None:
        assert sniff_image_mime(GIF89_MAGIC + b"trailing-bytes") == "image/gif"

    def test_webp(self) -> None:
        assert sniff_image_mime(WEBP_MAGIC + b"VP8 trailing") == "image/webp"

    def test_riff_non_webp_returns_none(self) -> None:
        # A RIFF container that isn't WebP (e.g. WAV audio, AVI video) carries a
        # different form type at offset 8 and must not sniff as an image.
        assert sniff_image_mime(b"RIFF\x24\x58\x00\x00WAVEfmt ") is None

    def test_truncated_riff_returns_none(self) -> None:
        # RIFF chunk id present but the form type at offset 8 is missing.
        assert sniff_image_mime(b"RIFF\x24\x58\x00\x00") is None

    def test_short_bytes_returns_none(self) -> None:
        assert sniff_image_mime(b"") is None
        assert sniff_image_mime(b"\x89P") is None

    def test_unknown_magic_returns_none(self) -> None:
        assert sniff_image_mime(b"%PDF-1.4\n%...") is None
        assert sniff_image_mime(b"\x00\x00\x00\x00garbage") is None
