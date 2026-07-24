"""Unit tests for the write-boundary outbound-target validator (#861 / PR #1931).

Covers the adversarial-review bypass matrix: decimal/hex/short-form IPv4
loopback encodings, DNS names resolving to private/metadata ranges, the
unresolvable-name fail-open (deliberate — PinnedTransport is the fail-closed
connection-time authority), and the single shared operator allowlist
(``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` — same variable PinnedTransport honors).
"""

from __future__ import annotations

import pytest

from aios.models import target_urls
from aios.models.target_urls import validate_outbound_target_url


def _reject(url: str) -> None:
    with pytest.raises(ValueError, match="private, internal, or runtime-local"):
        validate_outbound_target_url(url)


class TestNumericEncodings:
    """Loopback-equivalent IPv4 literal encodings are normalized (AI_NUMERICHOST)
    and rejected deterministically — the review's confirmed-by-execution bypasses."""

    def test_rejects_decimal_ipv4_loopback(self) -> None:
        _reject("http://2130706433/mcp")

    def test_rejects_hex_ipv4_loopback(self) -> None:
        _reject("http://0x7f000001/mcp")

    def test_rejects_shortform_ipv4_loopback(self) -> None:
        _reject("http://127.1/mcp")

    def test_rejects_dotted_quad_loopback(self) -> None:
        _reject("http://127.0.0.1:8080/mcp")

    def test_rejects_ipv6_loopback(self) -> None:
        _reject("http://[::1]:8080/mcp")

    def test_rejects_link_local_metadata_ip(self) -> None:
        _reject("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_and_cgnat_ranges(self) -> None:
        _reject("http://10.0.0.5/mcp")
        _reject("http://192.168.1.1/mcp")
        _reject("http://172.16.0.1/mcp")
        _reject("http://100.64.1.1/mcp")


class TestHostnames:
    def test_rejects_localhost_names(self) -> None:
        _reject("http://localhost:9000/mcp")
        _reject("http://foo.localhost/mcp")

    def test_rejects_metadata_hostname(self) -> None:
        _reject("http://metadata.google.internal/computeMetadata/v1/")

    def test_rejects_runtime_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_URL", "https://runtime.example/v1")
        _reject("https://runtime.example/mcp")

    def test_rejects_dns_name_resolving_to_private(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(target_urls, "_resolve_host_ips", lambda host: ["10.0.0.7"])
        _reject("https://internal.example.com/mcp")

    def test_rejects_dns_name_resolving_to_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(target_urls, "_resolve_host_ips", lambda host: ["127.0.0.1"])
        _reject("https://rebind.example.com/mcp")

    def test_accepts_dns_name_resolving_to_public(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(target_urls, "_resolve_host_ips", lambda host: ["93.184.216.34"])
        url = "https://mcp.example.com/mcp"
        assert validate_outbound_target_url(url) == url

    def test_unresolvable_name_is_admitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deliberate declaration-time fail-open: nothing reachable to protect,
        and every later resolution is re-checked fail-closed by PinnedTransport."""
        monkeypatch.setattr(target_urls, "_resolve_host_ips", lambda host: None)
        url = "https://not-yet-registered.example.com/mcp"
        assert validate_outbound_target_url(url) == url

    def test_mixed_resolution_any_blocked_ip_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            target_urls, "_resolve_host_ips", lambda host: ["93.184.216.34", "10.0.0.7"]
        )
        _reject("https://sneaky.example.com/mcp")


class TestShape:
    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ValueError, match="absolute http"):
            validate_outbound_target_url("ftp://example.com/x")

    def test_rejects_userinfo(self) -> None:
        with pytest.raises(ValueError, match="userinfo"):
            validate_outbound_target_url("https://user@example.com/mcp")

    def test_rejects_invalid_port(self) -> None:
        with pytest.raises(ValueError, match="invalid port"):
            validate_outbound_target_url("https://example.com:99999/mcp")


class TestOperatorAllowlist:
    """The SAME allowlist PinnedTransport enforces at connection time
    (``AIOS_OAUTH_ALLOW_INSECURE_HOSTS``) — no declaration/connection drift."""

    def test_allowlisted_loopback_host_admitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "127.0.0.1")
        url = "http://127.0.0.1:9000/mcp"
        assert validate_outbound_target_url(url) == url

    def test_allowlisted_host_port_admitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "mcp.internal:8080")
        url = "http://mcp.internal:8080/mcp"
        assert validate_outbound_target_url(url) == url

    def test_allowlist_is_host_scoped_not_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "mcp.internal:8080")
        _reject("http://127.0.0.1:9000/mcp")

    def test_no_stale_private_allow_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The PR's earlier parallel ``AIOS_TARGET_URL_ALLOW_HOSTS`` variable is
        gone — it must NOT bypass the validator (that was the drift finding)."""
        monkeypatch.setenv("AIOS_TARGET_URL_ALLOW_HOSTS", "127.0.0.1")
        _reject("http://127.0.0.1:9000/mcp")
