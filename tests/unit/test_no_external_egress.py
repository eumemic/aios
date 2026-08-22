from __future__ import annotations

import socket

import pytest

from tests.unit.no_external_egress import (
    ExternalEgressBlocked,
    GuardedSocket,
    guard_external_host,
    install_socket_guard,
)


def test_external_host_is_rejected_before_dns_or_connect() -> None:
    with pytest.raises(ExternalEgressBlocked, match="unit test attempted external egress"):
        guard_external_host("api.vendor.example")


def test_external_dns_lookup_is_rejected() -> None:
    with pytest.raises(ExternalEgressBlocked, match=r"api\.vendor\.example"):
        socket.getaddrinfo("api.vendor.example", 443)


def test_loopback_hosts_remain_available_to_local_server_tests() -> None:
    for host in ("localhost", "127.0.0.1", "::1"):
        guard_external_host(host)


def test_guarded_dns_fails_loudly_but_remains_an_os_error() -> None:
    monkeypatch = pytest.MonkeyPatch()
    install_socket_guard(monkeypatch)
    try:
        with pytest.raises(ExternalEgressBlocked):
            socket.getaddrinfo("api.vendor.example", 443)
        assert issubclass(ExternalEgressBlocked, OSError)
    finally:
        monkeypatch.undo()


def test_ip_address_objects_are_classified_without_live_dns() -> None:
    guard_external_host(socket.inet_pton(socket.AF_INET, "127.0.0.1"))


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("sendto", (b"", ("0.0.0.0", 9))),
        ("sendto", (b"", 0, ("192.0.2.1", 9))),
        ("sendmsg", ([b""], (), 0, ("192.0.2.1", 9))),
    ],
)
def test_connectionless_external_send_is_rejected_before_os_call(
    operation: str, args: tuple[object, ...]
) -> None:
    with (
        GuardedSocket(socket.AF_INET, socket.SOCK_DGRAM) as guarded_socket,
        pytest.raises(ExternalEgressBlocked),
    ):
        getattr(guarded_socket, operation)(*args)


def test_connectionless_loopback_send_remains_available() -> None:
    with GuardedSocket(socket.AF_INET, socket.SOCK_DGRAM) as guarded_socket:
        assert guarded_socket.sendto(b"", ("127.0.0.1", 9)) == 0


def test_unix_socket_operations_remain_available() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(b"local")
        assert right.recv(5) == b"local"
    finally:
        left.close()
        right.close()
