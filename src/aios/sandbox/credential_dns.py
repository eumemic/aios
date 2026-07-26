"""Per-sandbox name policy for credential-bearing HTTPS destinations."""

from __future__ import annotations

import argparse
import asyncio
import socket
import struct
from collections.abc import Iterable

CREDENTIAL_HOST_IP = "192.0.2.1"
CREDENTIAL_DNS_IP = "127.0.0.53"


def _question_name(packet: bytes) -> tuple[str, int]:
    labels: list[str] = []
    offset = 12
    while offset < len(packet):
        size = packet[offset]
        offset += 1
        if size == 0:
            return ".".join(labels).lower(), offset + 4
        if size & 0xC0 or offset + size > len(packet):
            raise ValueError("compressed or invalid DNS question")
        labels.append(packet[offset : offset + size].decode("ascii"))
        offset += size
    raise ValueError("unterminated DNS question")


def credential_answer(packet: bytes, hosts: Iterable[str]) -> bytes | None:
    """Return a synthetic A response when ``packet`` asks for a credential host."""
    if len(packet) < 12 or struct.unpack("!H", packet[4:6])[0] != 1:
        return None
    host, question_end = _question_name(packet)
    qtype, qclass = struct.unpack("!HH", packet[question_end - 4 : question_end])
    if host not in hosts or qtype != 1 or qclass != 1:
        return None
    flags = 0x8180
    header = packet[:2] + struct.pack("!HHHHH", flags, 1, 1, 0, 0)
    answer = b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 0, 4) + socket.inet_aton(CREDENTIAL_HOST_IP)
    return header + packet[12:question_end] + answer


class CredentialDnsProtocol(asyncio.DatagramProtocol):
    """Authoritative for credential names; forwards every other DNS packet."""

    def __init__(self, hosts: Iterable[str], upstream: tuple[str, int]) -> None:
        self.hosts = frozenset(host.lower() for host in hosts)
        self.upstream = upstream
        self.transport: asyncio.DatagramTransport | None = None
        self.tasks: set[asyncio.Task[None]] = set()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.DatagramTransport)
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            answer = credential_answer(data, self.hosts)
        except (ValueError, UnicodeDecodeError):
            return
        if answer is not None:
            assert self.transport is not None
            self.transport.sendto(answer, addr)
            return
        task = asyncio.create_task(self._forward(data, addr))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _forward(self, data: bytes, addr: tuple[str, int]) -> None:
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        try:
            await loop.sock_sendto(sock, data, self.upstream)
            response = await asyncio.wait_for(loop.sock_recv(sock, 65535), timeout=5)
            if self.transport is not None:
                self.transport.sendto(response, addr)
        finally:
            sock.close()


async def _serve(hosts: list[str], upstream: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        lambda: CredentialDnsProtocol(hosts, (upstream, 53)),
        local_addr=("0.0.0.0", 53),
    )
    await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("hosts", nargs="+")
    args = parser.parse_args()
    asyncio.run(_serve(args.hosts, args.upstream))


if __name__ == "__main__":
    main()
