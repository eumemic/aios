from __future__ import annotations

import socket
import struct

from aios.sandbox.credential_dns import CREDENTIAL_HOST_IP, credential_answer


def _query(host: str, qtype: int = 1) -> bytes:
    question = b"".join(bytes([len(label)]) + label.encode() for label in host.split("."))
    return (
        b"\x12\x34"
        + struct.pack("!HHHHH", 0x0100, 1, 0, 0, 0)
        + question
        + b"\0"
        + struct.pack("!HH", qtype, 1)
    )


def test_credential_name_gets_stable_synthetic_address() -> None:
    answer = credential_answer(_query("api.secret.com"), {"api.secret.com"})
    assert answer is not None
    assert answer[-4:] == socket.inet_aton(CREDENTIAL_HOST_IP)


def test_unrelated_name_is_forwarded() -> None:
    assert credential_answer(_query("example.com"), {"api.secret.com"}) is None


def test_non_a_query_is_forwarded() -> None:
    assert credential_answer(_query("api.secret.com", qtype=28), {"api.secret.com"}) is None
