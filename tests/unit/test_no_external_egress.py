from __future__ import annotations

import socket

import pytest

from tests.unit.no_external_egress import ExternalEgressBlocked, guard_external_host


def test_external_host_is_rejected_before_dns_or_connect() -> None:
    with pytest.raises(ExternalEgressBlocked, match="unit test attempted external egress"):
        guard_external_host("api.vendor.example")


def test_loopback_hosts_remain_available_to_local_server_tests() -> None:
    for host in ("localhost", "127.0.0.1", "::1"):
        guard_external_host(host)


def test_ip_address_objects_are_classified_without_live_dns() -> None:
    guard_external_host(socket.inet_pton(socket.AF_INET, "127.0.0.1"))
