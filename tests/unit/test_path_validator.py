"""Pure-unit tests for the shared path/prefix validator.

Used by inbound-message path validation (``allow_empty=False``) and
routing-rule prefix validation (``allow_empty=True``) — the latter
accepts ``""`` as the per-connection catch-all.
"""

from __future__ import annotations

import pytest

from aios.models._paths import validate_path_segments


class TestValid:
    @pytest.mark.parametrize(
        "s",
        [
            "A",
            "A/B",
            "A/B/C",
            "chat-abc",
            "chat_abc",
            "01HQR2K7VXBZ9MNPL3WYCT8F",
            "uuid/thread-ts",
            "a.b.c",  # dots inside a segment are fine; only literal '..' segment is rejected
            ".",  # single-dot segment is not '..' — permitted
        ],
    )
    def test_non_empty_variants(self, s: str) -> None:
        # Accepted regardless of allow_empty.
        validate_path_segments(s, allow_empty=False)
        validate_path_segments(s, allow_empty=True)

    def test_empty_allowed(self) -> None:
        validate_path_segments("", allow_empty=True)


class TestInvalid:
    @pytest.mark.parametrize(
        "s",
        [
            "/A",
            "A/",
            "/",
            "A//B",
            "..",
            "A/..",
            "../B",
            "A/../B",
            "A/B/..",
        ],
    )
    def test_malformed_rejected_always(self, s: str) -> None:
        with pytest.raises(ValueError):
            validate_path_segments(s, allow_empty=False)
        with pytest.raises(ValueError):
            validate_path_segments(s, allow_empty=True)

    def test_empty_rejected_when_not_allowed(self) -> None:
        with pytest.raises(ValueError):
            validate_path_segments("", allow_empty=False)
