"""Deny third-party network access from the unit-test process."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any


class ExternalEgressBlocked(OSError):
    """Raised when a unit test attempts to contact a non-loopback host.

    This is an ``OSError`` so application code that deliberately handles DNS
    failure can follow that degraded path while an unhandled lookup still
    fails the test loudly.
    """



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

    def _guard_address(self, address: Any) -> None:
        if self.family != socket.AF_UNIX:
            guard_external_host(address[0])

    def connect(self, address: Any) -> None:
        self._guard_address(address)
        super().connect(address)

    def connect_ex(self, address: Any) -> int:
        self._guard_address(address)
        return super().connect_ex(address)

    def _guard_peer(self) -> None:
        if self.family != socket.AF_UNIX:
            self._guard_address(self.getpeername())

    def send(self, data: Any, flags: int = 0) -> int:
        self._guard_peer()
        return super().send(data, flags)

    def sendall(self, data: Any, flags: int = 0) -> None:
        self._guard_peer()
        super().sendall(data, flags)

    def sendfile(self, file: Any, offset: int = 0, count: int | None = None) -> int:
        self._guard_peer()
        return super().sendfile(file, offset, count)

    def sendto(self, data: Any, *args: Any) -> int:
        """Guard the destination in both sendto(data, address) call forms."""
        self._guard_address(args[-1])
        return super().sendto(data, *args)

    def sendmsg(
        self,
        buffers: Any,
        ancdata: Any = (),
        flags: int = 0,
        address: Any = None,
    ) -> int:
        """Guard a connectionless sendmsg destination when one is supplied."""
        if address is not None:
            self._guard_address(address)
            return super().sendmsg(buffers, ancdata, flags, address)
        self._guard_peer()
        return super().sendmsg(buffers, ancdata, flags)


def install_socket_guard(monkeypatch: Any) -> None:
    """Reject external DNS and socket connections while preserving loopback tests."""
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        flags = kwargs.get("flags", args[4] if len(args) > 4 else 0)
        # AI_NUMERICHOST is a local parser used by URLvalidation, not DNS.
        if flags & socket.AI_NUMERICHOST:
            return real_getaddrinfo(host, *args, **kwargs)
        guard_external_host(host)
        return real_getaddrinfo(host, *args, **kwargs)

    real_gethostbygjg˜[YHHÛØÚÙ]™Ù]Üİ[˜[YBˆ™X[ÙÙ]Üİ[˜[YWÙ^HÛØÚÙ]™Ù]ÜİYÚ™æÖUöW€¢&VÅövWF†÷7F'–FG"Ò6ö6¶WBævWF†÷7F'–FG ¢&VÅövWFæÖV–æfòÒ6ö6¶WBævWFæÖV–æfğ ¢FVbwV&FVEövWF†÷7F'–æÖR††÷7C¢ç’’Óâç“ ¢wV&EöW‡FW&æÅö†÷7B††÷7B¢&WGW&â&VÅövWF†÷7F'–æÖR††÷7B ¢FVbwV&FVEövWF†÷7F'–v¦y…µ•}•à¡¡½ÍĞè¹ä¤€´ø¹äè(€€€€€€€Õ…É‘}•áÑ•É¹…±}¡½ÍĞ¡¡½ÍĞ¤(€€€€€€€É•ÑÕÉ¸É•…±}•Ñ¡½ÍÑ‰å©ame_ex(host)

    def guarded_gethostbyaddr(host: Any) -> Any:
        guard_external_host(host)
        return real_gethostbyaddr(host)

    def guarded_getnameinfo(sockaddr: Any, flags: int) -> Any:
        # NI_NUMERICHOST formats an address locally rather than resolving it.
        if not flags & socket.NI_NUMERICHOST:
            guard_external_host(sockaddr[0])
        return real_getnameinfo(sockaddr, flags)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbygjg˜[YH‹İX\™YÙÙ]ÜİYÚ™æÖR¢Ööæ¶W—F6‚ç6WFGG"‡6ö6¶WBÂ&vWF†÷7F'–v¦y…µ•}•àˆ°Õ…É‘•‘}•Ñ¡½ÍÑ‰å©ame_ex)
    monkeypatch.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)
    monkeypatch.setattr(socket, "getnameinfo", guarded_getnameinfo)
    monkeypatch.setattr(socket, "socket", GuardedSocket)
