"""Deny third-party network access from the unit-test process."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any


class ExternalEgressBlocked(RuntimeError):
    """Raised when a unit test attempts to contact a non-loopback host."""


def guard_external_host(host: Any) -> None:
    """Allow only loopback IPs/names, without resolving hostnames."""
    if host is None:
        return
    if isinstance(host, bytes):
        if len(host) in (4, 16):
            address = ipaddress.ip_address(host)
            if address.is_loopback:
                return
        else:
            host = host.decode("ascii")
    if isinstance(host, str):
        normalized = host.rstrip(".").lower()
        if normalized == "localhost":
            return
        try:
            if ipaddress.ip_address(normalized).is_loopback:
                return
        except ValueError:
            pass
    raise ExternalEgressBlocked(
        f"unit test attempted external egress to {host!r}; use a mock transport or loopback server"
    )


class GuardedSocket(socket.socket):
    """Socket that permits local test servers but rejects external peers."""

    def connect(self, address: Any) -> None:
        if self.family != socket.AF_UNIX:
            guard_external_host(address[0])
        super().connect(address)

    def connect_ex(self, address: Any) -> int:
        if self.family != socket.AF_UNIX:
            guard_external_host(address[0])
        return super().connect_ex(address)


def install_socket_guard(monkeypatch: Any) -> None:
    """Reject external DNS and socket connections while preserving loopback tests."""
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        flags = kwargs.get("flags", args[4] if len(args) > 4 else 0)
        # AI_NUMERICHOST is a local parser used by URL validation, not DNS.
        if flags & socket.AI_NUMERICHOST:
            return real_getaddrinfo(host, *args, **kwargs)
        try:
            guard_external_host(host)
        except ExternalEgressBlocked:
            # URL-validation tests use invented public names. Give them a stable
            # globally routed address without consulting the machine's DNS; the
            # socket guard below still prevents any connection to that address.
            port = args[0] if args else kwargs.get("port")
            socktype = kwargs.get("type", args[2] if len(args) > 2 else socket.SOCK_STREAM)
            return [(socket.AF_INET, socktype, 6, "", ("93.184.216.34", port or 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "socket", GuardedSocket)
