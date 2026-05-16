"""Unit tests for the segment-glob matcher used by ``http_request``."""

from __future__ import annotations

import pytest

from aios.tools._glob_match import match_glob


class TestLiteralPatterns:
    def test_exact_match(self) -> None:
        assert match_glob("/devices", "/devices")

    def test_exact_match_no_leading_slash(self) -> None:
        assert match_glob("devices", "/devices")
        assert match_glob("/devices", "devices")

    def test_exact_mismatch(self) -> None:
        assert not match_glob("/devices", "/lights")

    def test_segment_count_mismatch(self) -> None:
        assert not match_glob("/devices", "/devices/123")
        assert not match_glob("/devices/123", "/devices")


class TestSingleStar:
    def test_matches_one_segment(self) -> None:
        assert match_glob("/devices/*", "/devices/123")

    def test_does_not_match_zero_segments(self) -> None:
        assert not match_glob("/devices/*", "/devices")

    def test_does_not_match_multiple_segments(self) -> None:
        assert not match_glob("/devices/*", "/devices/123/state")

    def test_middle_position(self) -> None:
        assert match_glob("/devices/*/state", "/devices/123/state")
        assert not match_glob("/devices/*/state", "/devices/123/x/state")


class TestDoubleStar:
    def test_matches_zero_segments(self) -> None:
        assert match_glob("/devices/**", "/devices")

    def test_matches_one_segment(self) -> None:
        assert match_glob("/devices/**", "/devices/123")

    def test_matches_many_segments(self) -> None:
        assert match_glob("/devices/**", "/devices/a/b/c/d")

    def test_double_star_with_trailing_literal(self) -> None:
        assert match_glob("/devices/**/state", "/devices/123/state")
        assert match_glob("/devices/**/state", "/devices/a/b/state")
        assert not match_glob("/devices/**/state", "/devices/state/x")


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("/", "/", True),
        ("**", "/anything/at/all", True),
        ("**", "/", True),
        ("/api/v1/*", "/api/v1/users", True),
        ("/api/v1/*", "/api/v2/users", False),
    ],
)
def test_table(pattern: str, path: str, expected: bool) -> None:
    assert match_glob(pattern, path) is expected
