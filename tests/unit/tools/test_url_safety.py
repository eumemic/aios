"""The public SSRF-classification wrappers reused by the secret-egress proxy.

``is_safe_url`` (the resolving pre-flight) is exercised elsewhere; these pin the
pure classifiers the proxy's resolve-time gate calls on the IPs it has already
resolved.
"""

from __future__ import annotations

import pytest

from aios.tools.url_safety import is_blocked_hostname, is_blocked_ip


class TestIsBlockedIp:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "::1",  # loopback v6
            "169.254.169.254",  # link-local / cloud metadata
            "fe80::1",  # link-local v6
            "10.0.0.5",  # RFC-1918
            "172.16.0.1",  # RFC-1918
            "192.168.1.1",  # RFC-1918
            "fd00::1",  # IPv6 ULA (is_private)
            "100.64.0.1",  # CGNAT (not is_private)
            "0.0.0.0",  # unspecified
            "224.0.0.1",  # multicast
            "203.0.113.5",  # TEST-NET-3 — is_private in Python's special registry
            "not-an-ip",  # unparseable → fail closed
        ],
    )
    def test_blocked(self, ip: str) -> None:
        assert is_blocked_ip(ip) is True

    @pytest.mark.parametrize("ip", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
    def test_public_allowed(self, ip: str) -> None:
        assert is_blocked_ip(ip) is False


class TestIsBlockedHostname:
    @pytest.mark.parametrize(
        "host", ["metadata.google.internal", "metadata.goog", "METADATA.GOOGLE.INTERNAL"]
    )
    def test_blocked(self, host: str) -> None:
        assert is_blocked_hostname(host) is True

    @pytest.mark.parametrize("host", ["api.github.com", "example.com"])
    def test_allowed(self, host: str) -> None:
        assert is_blocked_hostname(host) is False
