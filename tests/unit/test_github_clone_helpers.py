"""Unit tests for the pure helpers in :mod:`aios.sandbox.github_clone`.

The async clone/fetch functions need a real ``git`` binary and disk;
those live in ``tests/e2e/test_github_repositories.py``. This file
covers only the logic-layer helpers: URL building, hashing, redaction.
"""

from __future__ import annotations

import pytest

from aios.sandbox.github_clone import (
    GithubCloneError,
    _build_auth_url,
    _redact_token_from_message,
    url_hash,
)


class TestBuildAuthUrl:
    def test_https_canonical(self) -> None:
        out = _build_auth_url("https://github.com/octocat/Hello-World", "ghp_xyz")
        assert out == "https://x-access-token:ghp_xyz@github.com/octocat/Hello-World"

    def test_https_with_dot_git_suffix(self) -> None:
        out = _build_auth_url("https://github.com/octocat/Hello-World.git", "ghp_xyz")
        assert out == "https://x-access-token:ghp_xyz@github.com/octocat/Hello-World.git"

    def test_http_allowed_for_self_hosted(self) -> None:
        # Some self-hosted GH servers expose HTTP only; we don't actively
        # block it, just leave the choice to the user.
        out = _build_auth_url("http://gh.internal/o/r", "tok")
        assert out == "http://x-access-token:tok@gh.internal/o/r"

    def test_ssh_url_rejected(self) -> None:
        with pytest.raises(GithubCloneError, match="https://"):
            _build_auth_url("git@github.com:o/r.git", "tok")

    def test_git_protocol_rejected(self) -> None:
        with pytest.raises(GithubCloneError, match="https://"):
            _build_auth_url("git://github.com/o/r.git", "tok")


class TestUrlHash:
    def test_stable_for_same_input(self) -> None:
        a = url_hash("https://github.com/o/r")
        b = url_hash("https://github.com/o/r")
        assert a == b

    def test_differs_for_different_urls(self) -> None:
        assert url_hash("https://github.com/o/r1") != url_hash("https://github.com/o/r2")

    def test_returns_hex_string(self) -> None:
        h = url_hash("https://github.com/o/r")
        assert len(h) == 64
        int(h, 16)  # would raise if not hex

    def test_distinguishes_dot_git_suffix(self) -> None:
        # Different cache buckets — we don't normalize, since git treats them
        # equivalently but the user might have meant either form deliberately.
        assert url_hash("https://github.com/o/r") != url_hash("https://github.com/o/r.git")


class TestRedactToken:
    def test_redacts_inline_token(self) -> None:
        msg = "fatal: Authentication failed for ghp_secret123"
        assert _redact_token_from_message(msg, "ghp_secret123") == (
            "fatal: Authentication failed for <redacted>"
        )

    def test_no_redaction_if_token_absent(self) -> None:
        msg = "fatal: not a git repository"
        assert _redact_token_from_message(msg, "ghp_secret") == msg

    def test_empty_token_passthrough(self) -> None:
        # Belt-and-suspenders: empty token should never reach this path,
        # but if it does, don't naive-replace into mush.
        assert _redact_token_from_message("hello world", "") == "hello world"
