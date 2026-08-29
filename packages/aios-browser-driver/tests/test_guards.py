"""The navigate guard's URL matrix: schemes, resolved addresses, the
hermetic-test knob, and resolution failures — every refusal is
``navigation_failed``."""

from __future__ import annotations

import socket
from typing import Any

import pytest
from aios_browser_driver.errors import ActionError
from aios_browser_driver.guards import check_url


def _resolving_to(*addrs: str) -> Any:
    # The stand-in for socket.getaddrinfo — the loop's executor passes all
    # six positionals.
    def fake_getaddrinfo(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
            for a in addrs
        ]

    return fake_getaddrinfo


async def _expect_blocked(url: str, *, allow_private: bool = False) -> str:
    with pytest.raises(ActionError) as info:
        await check_url(url, allow_private=allow_private)
    assert info.value.code == "navigation_failed"
    return info.value.message


async def test_public_addresses_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _resolving_to("93.184.216.34"))
    await check_url("https://example.com/page", allow_private=False)
    await check_url("http://example.com", allow_private=False)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "chrome://settings",
        "javascript:alert(1)",
        "data:text/html,<b>x</b>",
        "example.com/no-scheme",
        "https://",
    ],
)
async def test_non_http_schemes_and_hostless_urls_are_blocked_even_with_the_knob(
    url: str,
) -> None:
    # No getaddrinfo patch on purpose: these must be refused before any
    # resolution — and the knob must not bypass the scheme check.
    await _expect_blocked(url)
    await _expect_blocked(url, allow_private=True)


@pytest.mark.parametrize(
    "addr",
    [
        "10.0.0.8",
        "192.168.1.1",
        "172.16.3.4",
        "127.0.0.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.7",  # CGNAT
        "0.0.0.0",
        "::1",
        "fd00::1",
        "fe80::1%eth0",  # link-local with a scope suffix
        "::ffff:10.0.0.1",  # v4-mapped private (is_global delegates to the v4)
        "::ffff:169.254.169.254",  # v4-mapped metadata — a base-image regression guard
    ],
)
async def test_non_public_addresses_are_blocked(monkeypatch: pytest.MonkeyPatch, addr: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _resolving_to(addr))
    message = await _expect_blocked("https://internal.example/")
    assert "non-public" in message


async def test_one_private_addr_among_public_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    # DNS-rebinding shape: a name resolving to both a public and a private
    # address must be refused outright.
    monkeypatch.setattr(socket, "getaddrinfo", _resolving_to("93.184.216.34", "10.0.0.8"))
    await _expect_blocked("https://rebind.example/")


async def test_resolution_failure_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> list[Any]:
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", failing)
    message = await _expect_blocked("https://nxdomain.example/")
    assert "could not resolve" in message


async def test_the_knob_allows_private_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    def exploding(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("the knob must skip resolution entirely")

    monkeypatch.setattr(socket, "getaddrinfo", exploding)
    await check_url("http://127.0.0.1:8000/fixture", allow_private=True)
