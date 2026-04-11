"""Tests for the fuzzy matching module.

Ported from hermes-agent tests/tools/test_fuzzy_match.py with only the
import path updated. The fuzzy_match module is pure — no I/O — so
identical behavior is expected.
"""

from __future__ import annotations

from aios.vendor.hermes_files.fuzzy_match import fuzzy_find_and_replace


class TestExactMatch:
    def test_single_replacement(self) -> None:
        content = "hello world"
        new, count, err = fuzzy_find_and_replace(content, "hello", "hi")
        assert err is None
        assert count == 1
        assert new == "hi world"

    def test_no_match(self) -> None:
        content = "hello world"
        new, count, err = fuzzy_find_and_replace(content, "xyz", "abc")
        assert count == 0
        assert err is not None
        assert new == content

    def test_empty_old_string(self) -> None:
        _new, count, err = fuzzy_find_and_replace("abc", "", "x")
        assert count == 0
        assert err is not None

    def test_identical_strings(self) -> None:
        _new, count, err = fuzzy_find_and_replace("abc", "abc", "abc")
        assert count == 0
        assert err is not None
        assert "identical" in err

    def test_multiline_exact(self) -> None:
        content = "line1\nline2\nline3"
        new, count, err = fuzzy_find_and_replace(content, "line1\nline2", "replaced")
        assert err is None
        assert count == 1
        assert new == "replaced\nline3"


class TestWhitespaceDifference:
    def test_extra_spaces_match(self) -> None:
        content = "def  foo(  x,  y  ):"
        new, count, _err = fuzzy_find_and_replace(content, "def foo( x, y ):", "def bar(x, y):")
        assert count == 1
        assert "bar" in new


class TestIndentDifference:
    def test_different_indentation(self) -> None:
        content = "    def foo():\n        pass"
        new, count, _err = fuzzy_find_and_replace(
            content, "def foo():\n    pass", "def bar():\n    return 1"
        )
        assert count == 1
        assert "bar" in new


class TestReplaceAll:
    def test_multiple_matches_without_flag_errors(self) -> None:
        content = "aaa bbb aaa"
        _new, count, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=False)
        assert count == 0
        assert err is not None
        assert "Found 2 matches" in err

    def test_multiple_matches_with_flag(self) -> None:
        content = "aaa bbb aaa"
        new, count, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=True)
        assert err is None
        assert count == 2
        assert new == "ccc bbb ccc"
