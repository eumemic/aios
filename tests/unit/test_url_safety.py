"""Unit tests for ``is_cleartext_credential_target`` (SECURITY-02).

Pure scheme+allowlist predicate — no DNS. Callers AND it with "a credential is
actually being attached"; here we test the predicate in isolation.
"""

from __future__ import annotations

from aios.tools.url_safety import is_cleartext_credential_target


class TestIsCleartextCredentialTarget:
    def test_https_is_never_cleartext(self) -> None:
        assert is_cleartext_credential_target("https://api.example.com/token") is False

    def test_https_ignores_allow_hosts(self) -> None:
        # https is safe regardless of the allow-list membership.
        assert (
            is_cleartext_credential_target(
                "https://api.example.com", allow_hosts=frozenset({"api.example.com"})
            )
            is False
        )

    def test_plain_http_is_cleartext(self) -> None:
        assert is_cleartext_credential_target("http://api.example.com/token") is True

    def test_allow_listed_bare_host_bypasses(self) -> None:
        assert (
            is_cleartext_credential_target(
                "http://api.example.com", allow_hosts=frozenset({"api.example.com"})
            )
            is False
        )

    def test_allow_listed_host_port_form_bypasses(self) -> None:
        assert (
            is_cleartext_credential_target(
                "http://workspace-mcp:8000/mcp", allow_hosts=frozenset({"workspace-mcp:8000"})
            )
            is False
        )

    def test_allow_listed_bare_host_matches_host_with_port(self) -> None:
        # A bare-host allow-list entry matches the hostname even when the URL
        # carries an explicit port (mirrors _guard_url / PinnedTransport).
        assert (
            is_cleartext_credential_target(
                "http://workspace-mcp:8000/mcp", allow_hosts=frozenset({"workspace-mcp"})
            )
            is False
        )

    def test_plain_http_with_unrelated_allow_hosts_is_cleartext(self) -> None:
        assert (
            is_cleartext_credential_target(
                "http://api.example.com", allow_hosts=frozenset({"other-host"})
            )
            is True
        )
